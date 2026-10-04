# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "audio8-asr-infinite @ git+https://github.com/Edge0-AI/Audio8-ASR-Infinite@c8ba8eea829be0339e8d7757f8ca52dac06e1e32",
#   "torch==2.13.0", "transformers==5.13.0", "numpy>=2.1,<3",
# ]
# ///
"""Weight-free controller tests. Run: uv run audio8_streaming_test.py.

Mocks test exact scheduling/storage/backpressure. Tiny random-weight models
exercise the actual pinned Torch tower, Qwen delay decoder and SDPA masks;
these are NOT an ASR accuracy/performance validation of the real checkpoint.
"""
from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import DynamicCache, Qwen2Config
from transformers.models.voxtral_realtime.configuration_voxtral_realtime import VoxtralRealtimeEncoderConfig

import audio8_streaming as streaming
from audio8_asr_infinite.simulated_streaming_audio import iter_realtime_audio_windows


SPECIAL_IDS = {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
               "streaming_pad_token_id": 3, "streaming_word_token_id": 4}
BYTE_ENCODER = {value: char for char, value in streaming.BYTE_DECODER.items()}


class Tokenizer:
    all_special_ids = list(SPECIAL_IDS.values()) + [30, 31]

    def __init__(self, pieces=None):
        self.pieces = pieces or {6: b" hello", 7: "你".encode(), 8: b" world"}

    def convert_ids_to_tokens(self, token):
        return "".join(BYTE_ENCODER[value] for value in self.pieces[token])

    def decode(self, ids, **kwargs):
        return b"".join(self.pieces[token] for token in ids if token not in self.all_special_ids).decode("utf-8", errors="replace")


class Tower:
    def __init__(self):
        self.calls = []

    def embedder(self, features):
        return torch.zeros((1, (features.shape[-1] + 1) // 2, 4))

    def __call__(self, *, inputs_embeds, position_ids, attention_mask, past_key_values, **kwargs):
        self.calls.append((position_ids.clone(), attention_mask.clone()))
        states = position_ids.float().reshape(1, 1, -1, 1)
        past_key_values.update(states, states, 0)
        return types.SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=past_key_values)


class Model:
    def __init__(self, tokens=None):
        self.audio_tower = Tower()
        self.tokens = tokens or [6, 7, 8, 3]
        self.inputs, self.calls, self.condition_calls = [], [], []
        self.config = types.SimpleNamespace(max_frame_len=8,
            audio_config=types.SimpleNamespace(hidden_size=4, sliding_window=750))

    def multi_modal_projector(self, grouped):
        return grouped[..., :4]

    def build_t_cond(self, delay, **kwargs):
        self.condition_calls.append((delay, kwargs))
        return torch.zeros((1, 1, 4))

    def build_text_inputs_embeds(self, *, input_ids, audio_embeds):
        self.inputs.append(input_ids.tolist()[0])
        return audio_embeds

    def forward_language_model_with_delay(self, *, inputs_embeds, position_ids,
            cache_position, attention_mask, past_key_values, **kwargs):
        self.calls.append((position_ids.clone(), cache_position.clone(), attention_mask.clone(), kwargs))
        states = position_ids.float().reshape(1, 1, -1, 1)
        past_key_values.update(states, states, 0)
        logits = torch.zeros((1, 1, 32))
        logits[0, 0, SPECIAL_IDS["eos_token_id"]] = 100  # must be suppressed
        logits[0, 0, self.tokens[(len(self.calls) - 1) % len(self.tokens)]] = 50
        return types.SimpleNamespace(logits=logits, past_key_values=past_key_values)


def runtime(tokens=None, pieces=None):
    return types.SimpleNamespace(torch=torch, np=np, device=torch.device("cpu"), dtype=torch.float32,
        model=Model(tokens), tokenizer=Tokenizer(pieces), extractor=object(), special_ids=SPECIAL_IDS,
        language_ids={"en": 31, "zh": 30})


def decoder(runtime=None, window=32, pcm_capacity_samples=None):
    result = streaming.StreamingDecoder(runtime or globals()["runtime"](), "en", window,
                                       pcm_capacity_samples=pcm_capacity_samples)
    result.windows = []

    def features(*, audio_windows, **kwargs):
        result.windows.append(audio_windows[0].copy())
        return torch.zeros((1, 8, len(audio_windows[0]) // 160))

    result._features = features
    return result


def feed(decoder, waveform, sizes=(400, 2560, 1024, 3900)):
    events, offset, index = [], 0, 0
    while offset < len(waveform):
        count = sizes[index % len(sizes)]
        events.extend(decoder.push_pcm(waveform[offset:offset + count].astype("<f4").tobytes()))
        offset += count
        index += 1
    return events


def text(events):
    return "".join(event["text"] for event in events if event["type"] == "segment")


class CacheTests(unittest.TestCase):
    def test_crop_keeps_newest_not_first_and_releases_storage(self):
        cache = DynamicCache()
        for layer in range(3):
            values = torch.arange(40).reshape(1, 1, 40, 1).float() + layer * 100
            cache.update(values, values, layer)
        streaming.crop_recent(cache, 7)
        for index, layer in enumerate(cache.layers):
            self.assertEqual(layer.keys.flatten().tolist(), list(range(index * 100 + 33, index * 100 + 40)))
            self.assertEqual(layer.keys.untyped_storage().nbytes(), layer.keys.numel() * 4)
        self.assertEqual(cache.get_seq_length(), 7)

    def test_prefix_and_newest_tail_zero_tail(self):
        cache = DynamicCache()
        cache.update(torch.arange(12).reshape(1, 1, 12, 1), torch.arange(12).reshape(1, 1, 12, 1), 0)
        streaming.crop_recent(cache, 5, prefix=2)
        self.assertEqual(cache.layers[0].keys.flatten().tolist(), [0, 1, 9, 10, 11])
        streaming.crop_recent(cache, 2, prefix=2)
        self.assertEqual(cache.layers[0].keys.flatten().tolist(), [0, 1])
        streaming.crop_recent(cache, 0)
        self.assertEqual(cache.get_seq_length(), 0)

    def test_unsupported_cache_is_visible(self):
        from transformers.cache_utils import DynamicSlidingWindowLayer
        cache = types.SimpleNamespace(layers=[DynamicSlidingWindowLayer(sliding_window=4)])
        with self.assertRaisesRegex(TypeError, "plain dynamic"):
            streaming.crop_recent(cache, 4)


class ControllerTests(unittest.TestCase):
    def test_frontend_windows_match_pinned_batch_schedule(self):
        waveform = np.sin(np.arange(32 * 2560 + 77, dtype=np.float32) / 50) * 0.6
        d = decoder()
        feed(d, waveform)
        final = d.finish()
        config = types.SimpleNamespace(raw_audio_samples_per_token=2560,
            sampling_rate=16000, streaming_n_left_pad_tokens=9)
        # Upstream batch builds its schedule on ceil-rounded PCM, and stores
        # the actual waveform in that padded timeline.
        rounded = np.pad(waveform, (0, (-len(waveform)) % 2560))
        expected, _ = iter_realtime_audio_windows(rounded, audio_config=config,
            num_delay_tokens=3, right_pad_text_tokens=10, streaming_look_ahead_ms=2.5,
            streaming_look_back_ms=52.5, initial_prefill_tokens=13)
        expected = list(expected)
        self.assertEqual(len(d.windows), len(expected))
        for observed, window in zip(d.windows, expected):
            np.testing.assert_array_equal(observed, window["audio"])
        self.assertEqual(d._steps, 33 + 10)
        self.assertEqual(final[-1]["type"], "result")
        self.assertEqual(d.finish(), [])
        self.assertIsNone(d.audio_cache)
        self.assertEqual(d.buffered_samples, 0)

    def test_first_token_needs_delay_prefill_and_lookahead(self):
        d = decoder()
        self.assertEqual(d.push_pcm(np.zeros(4 * 2560, dtype="<f4").tobytes()), [])
        self.assertEqual(len(d.runtime.model.calls), 0)
        events = d.push_pcm(np.zeros(40, dtype="<f4").tobytes())
        self.assertEqual(events[0]["type"], "partial")
        self.assertEqual(d.runtime.model.inputs[0], [0, 31] + [3] * 11)
        self.assertEqual(d.absolute_text_position, 13)
        self.assertEqual(d.absolute_audio_position, 104)
        self.assertEqual(len(d.runtime.model.condition_calls), 1)
        d.push_pcm(np.zeros(2560, dtype="<f4").tobytes())
        self.assertEqual(d.runtime.model.inputs[1], [6])
        self.assertEqual(d.runtime.model.calls[1][3]["num_delay_tokens"], 3)
        self.assertEqual(d.runtime.model.calls[1][3]["frame_len"], 8)

    def test_long_session_bounds_prefix_tail_and_absolute_positions(self):
        d = decoder()
        # Keep test records bounded too: the actual controller stores neither
        # these captured windows nor the fake model's forward-call history.
        for _ in range(150):
            d.push_pcm(np.zeros(2560, dtype="<f4").tobytes())
            self.assertLessEqual(d.buffered_samples, d.pcm_capacity_samples)
            self.assertLessEqual(d.language_cache.get_seq_length(), 32)
            self.assertLessEqual(d.audio_cache.get_seq_length(), 256)
            self.assertLess(len(d._text), 400)
        self.assertGreater(d.absolute_text_position, 32)
        positions = d.language_cache.layers[0].keys.flatten().tolist()
        self.assertEqual(positions[:13], list(range(13)))
        self.assertEqual(positions[13:], list(range(d.absolute_text_position - 19, d.absolute_text_position)))
        audio_positions = d.audio_cache.layers[0].keys.flatten().tolist()
        self.assertEqual(audio_positions, list(range(d.absolute_audio_position - 256, d.absolute_audio_position)))
        # Every new audio query sees all old keys, plus only the causal portion
        # of the eight newly appended frames, even after hundreds of crops.
        mask = d.runtime.model.audio_tower.calls[-1][1][0, 0]
        self.assertEqual(tuple(mask.shape), (8, 264))
        self.assertEqual(torch.isfinite(mask[0]).sum().item(), 257)
        self.assertEqual(torch.isfinite(mask[-1]).sum().item(), 264)
        self.assertTrue(torch.equal(d.runtime.model.calls[-1][0].flatten(),
                                    torch.tensor([d.absolute_text_position - 1])))

    def test_audio_native_sliding_mask_is_not_bypassed(self):
        r = runtime()
        r.model.config.audio_config.sliding_window = 24
        d = decoder(r, window=32)
        feed(d, np.zeros(20 * 2560, dtype=np.float32))
        mask = d.runtime.model.audio_tower.calls[-1][1][0, 0]
        self.assertEqual(torch.isfinite(mask).sum(dim=-1).tolist(), [24] * 8)
        self.assertEqual(d.audio_cache.get_seq_length(), 24)

    def test_utf8_split_across_tokens_and_segment_commit(self):
        # First 31 ticks are ASCII. A Chinese codepoint's first byte lands on
        # the commit tick, its continuation arrives in the next segment.
        pieces = {6: b"a", 7: b"\xe4", 8: b"\xbd\xa0", 9: b"\xf0\x9f", 10: b"\x98\x80"}
        tokens = [6] * 31 + [7, 8, 9, 10] + [3] * 40
        d = decoder(runtime(tokens, pieces))
        events = feed(d, np.zeros(36 * 2560, dtype=np.float32))
        events.extend(d.finish())
        self.assertEqual(text(events), "a" * 31 + "你😀")
        self.assertEqual(len([e for e in events if e["type"] == "segment"]), 2)
        self.assertNotIn("�", text(events))
        partials = [event for event in events if event["type"] == "partial"]
        self.assertEqual(partials[-1]["text"], "你😀")
        self.assertEqual(partials[-1]["segment_index"], 1)
        self.assertEqual(partials[-1]["committed_characters"], 31)
        self.assertLessEqual(len(d._utf8.getstate()[0]), 3)

    def test_partial_text_monotone_per_segment(self):
        d = decoder()
        events = feed(d, np.zeros(75 * 2560, dtype=np.float32)) + d.finish()
        previous = {}
        for event in events:
            if event["type"] == "partial":
                index = event["segment_index"]
                self.assertTrue(event["text"].startswith(previous.get(index, "")))
                previous[index] = event["text"]
            if event["type"] == "segment":
                self.assertEqual(event["text"], previous[event["segment_index"]])
                self.assertLessEqual(event["end"], 75 * 16)
                self.assertLessEqual(event["start"], event["end"])
        self.assertEqual(d._text, "")
        self.assertNotEqual(d._next_token, SPECIAL_IDS["eos_token_id"])

    def test_packetisation_does_not_change_text_or_end_drain(self):
        wave = np.zeros(55 * 2560 + 7, dtype=np.float32)
        first, second = decoder(), decoder()
        a = feed(first, wave, (320,)) + first.finish()
        b = feed(second, wave, (14000, 23, 5120)) + second.finish()
        self.assertEqual(a, b)
        self.assertEqual(first._steps, 56 + 10)

    def test_invalid_and_overflow_push_are_atomic(self):
        d = decoder()
        for raw, error in ((b"x", ValueError), (np.array([float("nan")], dtype="<f4").tobytes(), ValueError),
                           (np.zeros(d.pcm_capacity_samples + 1, dtype="<f4").tobytes(), BufferError)):
            with self.assertRaises(error):
                d.push_pcm(raw)
            self.assertEqual(d.accepted_samples, 0)
            self.assertEqual(d.absolute_text_position, 0)
        d.push_pcm(np.zeros(5000, dtype="<f4").tobytes())
        with self.assertRaises(BufferError):
            d.push_pcm(np.zeros(d.pcm_capacity_samples - 4999, dtype="<f4").tobytes())
        self.assertEqual(d.accepted_samples, 5000)
        self.assertFalse(d._failed)

    def test_empty_finish_no_inference_and_push_after_finish(self):
        d = decoder()
        self.assertEqual(d.finish(), [{"type": "result", "segments": 0, "samples": 0}])
        self.assertEqual(d.runtime.model.calls, [])
        self.assertEqual(d.finish(), [])
        with self.assertRaisesRegex(RuntimeError, "after finish"):
            d.push_pcm(b"")

    def test_inference_failure_poison_and_concurrent_backpressure(self):
        d = decoder()
        d._lock.acquire()
        with self.assertRaisesRegex(RuntimeError, "backpressure"):
            d.push_pcm(b"")
        d._lock.release()
        with patch.object(d, "_forward", side_effect=RuntimeError("device failed")):
            with self.assertRaisesRegex(RuntimeError, "device failed"):
                d.push_pcm(np.zeros(10280, dtype="<f4").tobytes())
        self.assertEqual(d.accepted_samples, 10280)
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            d.finish()

    def test_incomplete_utf8_fails_visibly(self):
        d = decoder(runtime([6] + [3] * 25, {6: b"\xe4"}))
        d.push_pcm(np.zeros(1, dtype="<f4").tobytes())
        with self.assertRaises(UnicodeDecodeError):
            d.finish()
        self.assertTrue(d._failed)

    def test_bad_tokenizer_and_window_fail_visible(self):
        with self.assertRaises(ValueError):
            streaming.StreamingDecoder(runtime(), "auto")
        with self.assertRaises(ValueError):
            streaming.StreamingDecoder(runtime(), "en", 0)
        r = runtime()
        r.tokenizer.backend_tokenizer = types.SimpleNamespace(to_str=lambda: '{"decoder":{"type":"Metaspace"}}')
        with self.assertRaisesRegex(ValueError, "ByteLevel"):
            streaming.StreamingDecoder(r, "en")


class ActualTorchTests(unittest.TestCase):
    def test_tiny_random_pinned_architecture_matches_reference_before_rolling(self):
        from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
        from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import Audio8ASRInfiniteForConditionalGeneration
        from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch
        from transformers import VoxtralRealtimeFeatureExtractor
        torch.manual_seed(4)
        torch.set_num_threads(2)
        audio = VoxtralRealtimeEncoderConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            head_dim=8, num_mel_bins=8, sliding_window=750)
        config = Audio8ASRInfiniteConfig(audio_config=audio.to_dict(),
            text_config=Qwen2Config(hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
                vocab_size=32, bos_token_id=0, eos_token_id=1, pad_token_id=2).to_dict(),
            supported_frame_lens=[8], max_frame_len=8, projection_size=256,
            semantic_vad_horizons_seconds=None)
        config._attn_implementation = "sdpa"
        model = Audio8ASRInfiniteForConditionalGeneration(config).eval()
        extractor = VoxtralRealtimeFeatureExtractor(feature_size=8, sampling_rate=16000,
            n_fft=400, hop_length=160, win_length=400, global_log_mel_max=1.5)
        r = runtime()
        r.model, r.extractor = model, extractor
        r.tokenizer = Tokenizer({token: b"a" for token in range(32)})
        waveform = np.sin(np.arange(7 * 2560 + 2, dtype=np.float32) / 65) * 0.1
        d = streaming.StreamingDecoder(r, "en", 128)
        tokens, original = [], d._emit_token
        def capture(token):
            tokens.append(token)
            return original(token)
        d._emit_token = capture
        events = feed(d, waveform) + d.finish()
        ac = types.SimpleNamespace(raw_audio_samples_per_token=2560, sampling_rate=16000,
            streaming_n_left_pad_tokens=9)
        expected = simulated_streaming_greedy_decode_batch(model=model, tokenizer=r.tokenizer,
            feature_extractor=extractor, waveforms=[waveform], language_token_ids=[31],
            special_ids=SPECIAL_IDS, audio_config=ac, num_delay_tokens=3, frame_len=8,
            right_pad_text_tokens=10, dtype=torch.float32, device=torch.device("cpu"))
        self.assertEqual(tokens, expected[0]["generated_token_ids"])
        self.assertEqual(text(events), expected[0]["final_text"])
        # Continue across rolling boundaries with the actual Torch modules:
        # cache shape and absolute counters, not ASR quality, are asserted.
        d = streaming.StreamingDecoder(r, "en", 16)
        for _ in range(45):
            d.push_pcm(np.zeros(2560, dtype="<f4").tobytes())
        self.assertEqual(d.language_cache.get_seq_length(), 16)
        self.assertEqual(d.audio_cache.get_seq_length(), 128)
        self.assertGreater(d.absolute_text_position, 16)
        self.assertEqual(d.absolute_audio_position, d.absolute_text_position * 8)
        d.finish()


class CalibrationTests(unittest.TestCase):
    def test_saturated_trials_select_largest_eligible_context(self):
        # Exercise the real calibration controller/masks and timing protocol
        # with tiny fake tensors; no checkpoint weight load is involved.
        r = runtime()
        original = streaming.StreamingDecoder
        def make(*args):
            d = original(*args)
            d._features = lambda *, audio_windows, **kwargs: torch.zeros((1, 8, len(audio_windows[0]) // 160))
            return d
        with patch.object(streaming, "StreamingDecoder", side_effect=make):
            report = streaming.calibrate(r)
        self.assertEqual(report["window_tokens"], 256)
        self.assertTrue(report["realtime"])
        self.assertFalse(report["accuracy_validated"])
        self.assertGreaterEqual(report["pcm_capacity_samples"], streaming.MIN_PCM_CAPACITY_SAMPLES)
        self.assertLessEqual(report["pcm_capacity_samples"], streaming.MAX_PCM_CAPACITY_SAMPLES)
        self.assertEqual(report["pcm_capacity_samples"] % 2560, 0)
        self.assertEqual(original(r, "en").pcm_capacity_samples, report["pcm_capacity_samples"])
        self.assertIn("threads=", report["hardware_key"])
        self.assertEqual(len(report["trials"]), 4)
        self.assertEqual(report["trials"][0]["cache_bytes"], (32 + 256) * 2 * 4)

    def test_ring_sizing_observed_jitter_bounds_and_rounding(self):
        self.assertEqual(streaming._pcm_capacity(0, 0, 0), (12800, True))
        # Actual cached-checkpoint MPS timings measured in the validation slot.
        self.assertEqual(streaming._pcm_capacity(0.24186837498564273,
            0.16428687499137595, 0.1608001668122597), (15360, True))
        self.assertEqual(streaming._pcm_capacity(10, 0.1, 0.1),
                         (streaming.MAX_PCM_CAPACITY_SAMPLES, False))
        waveform = np.sin(np.arange(36 * 2560 + 1, dtype=np.float32) / 43) * 0.5
        reference = decoder()
        expected_events = feed(reference, waveform, (2560,)) + reference.finish()
        for capacity in (12800, 15360):
            d = decoder(pcm_capacity_samples=capacity)
            self.assertEqual(feed(d, waveform, (2560,)) + d.finish(), expected_events)
            self.assertEqual(len(d.windows), len(reference.windows))
            for observed, expected in zip(d.windows, reference.windows):
                np.testing.assert_array_equal(observed, expected)
        with self.assertRaises(ValueError):
            streaming.StreamingDecoder(runtime(), "en", pcm_capacity_samples=10280)

    def test_calibration_cancellation_between_windows(self):
        r = runtime()
        original = streaming.StreamingDecoder
        calls = [0]
        def check():
            calls[0] += 1
            return calls[0] == 3
        def make(*args):
            d = original(*args)
            d._features = lambda *, audio_windows, **kwargs: torch.zeros((1, 8, len(audio_windows[0]) // 160))
            return d
        with patch.object(streaming, "StreamingDecoder", side_effect=make):
            with self.assertRaisesRegex(InterruptedError, "cancelled"):
                streaming.calibrate(r, abort_check=check)
        self.assertEqual(len(r.model.calls), 2)
        self.assertFalse(hasattr(r, "streaming_pcm_capacity_samples"))

    def test_slow_steps_or_memory_budget_reject_realtime(self):
        r = runtime()
        r.streaming_memory_budget_bytes = 1
        original = streaming.StreamingDecoder
        def make(*args):
            d = original(*args)
            d._features = lambda *, audio_windows, **kwargs: torch.zeros((1, 8, len(audio_windows[0]) // 160))
            return d
        with patch.object(streaming, "StreamingDecoder", side_effect=make):
            report = streaming.calibrate(r)
        self.assertFalse(report["realtime"])
        self.assertTrue(all(not trial["realtime"] for trial in report["trials"]))
        # Even plenty of memory cannot make a 130ms step pass a 128ms budget.
        r.streaming_memory_budget_bytes = 1024 ** 3
        clock = [0.0]
        def now():
            clock[0] += 0.130
            return clock[0]
        with patch.object(streaming, "StreamingDecoder", side_effect=make), patch.object(streaming.time, "perf_counter", side_effect=now):
            self.assertFalse(streaming.calibrate(r)["realtime"])


if __name__ == "__main__":
    unittest.main()
