# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy>=2.1,<3"]
# ///
"""Portable scheduler contract tests; deliberately no Torch/Transformers deps.

Run: uv run --script server/crates/vibe-server/src/audio8_portable_streaming_test.py
This uses fake NumPy KV caches, never checkpoint weights or backend inference.
"""
import builtins
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import audio8_streaming as streaming

SPECIAL = {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
           "streaming_pad_token_id": 3, "streaming_word_token_id": 4}
ENCODER = {value: char for char, value in streaming.BYTE_DECODER.items()}


class Tokenizer:
    all_special_ids = list(SPECIAL.values())
    backend_tokenizer = types.SimpleNamespace(to_str=lambda: '{"decoder":{"type":"ByteLevel"}}')
    pieces = {6: b" hello", 7: "你".encode(), 8: b" world", 9: b"\xe4", 10: b"\xbd\xa0"}

    def convert_ids_to_tokens(self, token):
        return "".join(ENCODER[byte] for byte in self.pieces[token])


class Step:
    def __init__(self, window, tokens):
        self.window = window
        self.audio_window_frames = min(window * 8, 750)
        self.audio_position = self.text_position = 0
        self.audio_cache = self.text_cache = np.empty(0, dtype=np.int64)
        self.tokens, self.windows, self.ids = tokens, [], []

    def step(self, audio_window, ids):
        self.windows.append(audio_window.copy())
        self.ids.append(list(ids))
        width = len(ids)
        self.audio_cache = np.concatenate((self.audio_cache,
            np.arange(self.audio_position, self.audio_position + width * 8))) [-self.audio_window_frames:].copy()
        self.text_cache = np.concatenate((self.text_cache,
            np.arange(self.text_position, self.text_position + width)))
        if len(self.text_cache) > self.window:
            self.text_cache = np.concatenate((self.text_cache[:13], self.text_cache[-(self.window - 13):]))
        self.text_position += width
        self.audio_position += width * 8
        return np.int64(self.tokens[(len(self.ids) - 1) % len(self.tokens)])

    def cache_stats(self):
        return {"cache_bytes": self.audio_cache.nbytes + self.text_cache.nbytes,
            "audio_frames": len(self.audio_cache), "text_tokens": len(self.text_cache),
            "audio_position": self.audio_position, "text_position": self.text_position}


class Runtime:
    np = np
    tokenizer = Tokenizer()
    special_ids = SPECIAL
    language_ids = {"en": 30, "zh": 31}
    device, dtype = "portable:cpu", "float32"
    hardware_key = "portable-test-machine"
    backend_version = "fake-v1"

    def __init__(self, tokens=None, step_class=Step):
        self.tokens = tokens or [6, 7, 8, 3]
        self.step_class = step_class
        self.steps_created = []
        self.synchronizations = 0
        self.model_loads = 0

    def new_step_decoder(self, window_tokens):
        result = self.step_class(window_tokens, self.tokens)
        self.steps_created.append(result)
        return result

    def synchronize(self):
        self.synchronizations += 1

    def load_model(self):
        self.model_loads += 1
        raise AssertionError("scheduler must not load a model")


def feed(decoder, waveform, packet=2560):
    events = []
    for start in range(0, len(waveform), packet):
        events.extend(decoder.push_pcm(waveform[start:start + packet].astype("<f4").tobytes()))
    return events


class PortableTests(unittest.TestCase):
    def test_no_torch_import_or_model_load_in_stream_and_calibration(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "transformers", "audio8_asr_infinite"):
                raise AssertionError(f"Portable path imported {name}")
            return original(name, *args, **kwargs)
        r = Runtime()
        with patch("builtins.__import__", side_effect=guarded):
            decoder = streaming.StreamingDecoder(r, "en", 32, 15360)
            feed(decoder, np.zeros(8 * 2560 + 9, dtype=np.float32))
            decoder.finish()
            report = streaming.calibrate(r)
        self.assertEqual(r.model_loads, 0)
        self.assertEqual(report["hardware_key"], r.hardware_key)
        self.assertTrue(report["realtime"])
        self.assertEqual(report["window_tokens"], 256)
        self.assertFalse(any(name.split(".")[0] in ("torch", "transformers", "audio8_asr_infinite") for name in sys.modules))

    def test_exact_frontend_clock_feedback_and_tail(self):
        wave = np.sin(np.arange(18 * 2560 + 27, dtype=np.float32) / 45) * 0.7
        r = Runtime()
        decoder = streaming.StreamingDecoder(r, "zh", 32, 15360)
        events = feed(decoder, wave, 1337)
        live_steps = decoder._steps
        events += decoder.finish()
        step = r.steps_created[0]
        self.assertEqual(len(step.ids), 19 + 10)
        self.assertLess(live_steps, len(step.ids))
        self.assertEqual(step.ids[0], [0, 31] + [3] * 11)
        for index, ids in enumerate(step.ids[1:], 1):
            self.assertEqual(ids, [r.tokens[(index - 1) % len(r.tokens)]])
        timeline = np.zeros((9 + 19 + 3 + 1 + 10) * 2560, dtype=np.float32)
        timeline[9 * 2560:9 * 2560 + len(wave)] = wave
        start, end, expected = 0, 13 * 2560, []
        while end + 40 <= len(timeline):
            expected.append(timeline[max(0, start - 840):end + 40])
            start, end = end, end + 2560
        self.assertEqual(len(step.windows), len(expected))
        for observed, reference in zip(step.windows, expected):
            np.testing.assert_array_equal(observed, reference)
        self.assertTrue(any(event["type"] == "partial" for event in events))
        self.assertTrue(any(event["type"] == "segment" for event in events))
        self.assertEqual(events[-1], {"type": "result", "segments": decoder.segment_count, "samples": len(wave)})
        self.assertEqual(decoder.finish(), [])
        self.assertIsNone(decoder._backend_step)
        self.assertEqual(decoder.cache_stats()["cache_bytes"], 0)

    def test_calibration_reuses_runtime_fills_real_physical_cache(self):
        r = Runtime()
        report = streaming.calibrate(r)
        self.assertEqual(len(r.steps_created), 4)
        for window, step in zip((32, 64, 128, 256), r.steps_created):
            self.assertEqual([len(ids) for ids in step.ids], [13, window - 13] + [1] * 8)
            self.assertEqual(step.ids[0], [0, 30] + [3] * 11)
            self.assertEqual(step.ids[1], [6] * (window - 13))
            self.assertEqual(step.cache_stats()["text_tokens"], window)
            self.assertEqual(step.cache_stats()["audio_frames"], min(window * 8, 750))
        self.assertEqual(r.model_loads, 0)
        self.assertGreater(r.synchronizations, 20)
        self.assertEqual(report["pcm_capacity_samples"] % 2560, 0)
        decoder = streaming.StreamingDecoder(r, "en", report["window_tokens"])
        self.assertEqual(decoder.pcm_capacity_samples, report["pcm_capacity_samples"])

    def test_rolls_bounds_keeps_positions_and_safe_utf8_across_commit(self):
        r = Runtime([6] * 31 + [9, 10] + [3] * 100)
        decoder = streaming.StreamingDecoder(r, "en", 32, 12800)
        events = feed(decoder, np.zeros(150 * 2560 + 1, dtype=np.float32))
        stats = decoder.cache_stats()
        self.assertEqual(stats["text_tokens"], 32)
        self.assertEqual(stats["audio_frames"], 256)
        self.assertGreater(stats["text_position"], 32)
        self.assertEqual(stats["audio_position"], stats["text_position"] * 8)
        cache = r.steps_created[0].text_cache
        np.testing.assert_array_equal(cache[:13], np.arange(13))
        np.testing.assert_array_equal(cache[13:], np.arange(stats["text_position"] - 19, stats["text_position"]))
        self.assertLessEqual(decoder.buffered_samples, 12800)
        events += decoder.finish()
        committed = "".join(event["text"] for event in events if event["type"] == "segment")
        self.assertIn("你", committed)
        self.assertNotIn("�", committed)
        self.assertLess(len(decoder._text), 4096)

    def test_eos_failure_is_visible_and_poisoned(self):
        r = Runtime([1])
        decoder = streaming.StreamingDecoder(r, "en", 32)
        with self.assertRaisesRegex(RuntimeError, "suppress EOS"):
            decoder.push_pcm(np.zeros(10280, dtype="<f4").tobytes())
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            decoder.finish()

    def test_calibration_rejects_unsaturated_cache(self):
        class Unsaturated(Step):
            def cache_stats(self):
                stats = super().cache_stats()
                stats["text_tokens"] = min(stats["text_tokens"], self.window - 1)
                return stats
        with self.assertRaisesRegex(RuntimeError, "not genuinely saturated"):
            streaming.calibrate(Runtime(step_class=Unsaturated))

    def test_bad_cache_position_and_missing_stats_fail(self):
        class BadPosition(Step):
            def cache_stats(self):
                return {**super().cache_stats(), "audio_position": 0}
        decoder = streaming.StreamingDecoder(Runtime(step_class=BadPosition), "en", 32)
        with self.assertRaisesRegex(RuntimeError, "absolute positions"):
            decoder.push_pcm(np.zeros(10280, dtype="<f4").tobytes())
        class Missing(Step):
            def cache_stats(self):
                return {"cache_bytes": 1}
        decoder = streaming.StreamingDecoder(Runtime(step_class=Missing), "en", 32)
        with self.assertRaisesRegex(ValueError, "audio_frames"):
            decoder.push_pcm(np.zeros(10280, dtype="<f4").tobytes())

    def test_statistics_aliases_and_sync_optional(self):
        class Aliases(Step):
            def cache_stats(self):
                return {"total_cache_bytes": self.audio_cache.nbytes + self.text_cache.nbytes,
                    "audio_cache_frames": len(self.audio_cache), "text_cache_tokens": len(self.text_cache),
                    "absolute_audio_position": self.audio_position, "absolute_text_position": self.text_position}
        r = Runtime(step_class=Aliases)
        r.synchronize = None
        decoder = streaming.StreamingDecoder(r, "en", 32, 15360)
        feed(decoder, np.zeros(10 * 2560, dtype=np.float32))
        self.assertGreater(decoder.cache_stats()["cache_bytes"], 0)
        decoder.finish()
        self.assertEqual(r.synchronizations, 0)
        streaming._synchronize(r)


if __name__ == "__main__":
    unittest.main()
