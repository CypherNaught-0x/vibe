# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "audio8-asr-infinite @ git+https://github.com/Edge0-AI/Audio8-ASR-Infinite@c8ba8eea829be0339e8d7757f8ca52dac06e1e32",
#   "torch==2.13.0", "transformers==5.13.0", "onnx==1.22.0",
#   "onnxruntime==1.23.2", "numpy==2.4.4", "safetensors==0.8.0", "tokenizers==0.22.2",
# ]
# ///
"""Executable ORT/Torch parity harness. Default: tiny untrained architecture.

uv run tests.py --checkpoint <pinned local snapshot> --output <new workspace dir>

No full pretrained weights are read by default. --pretrained-ort writes an ORT
fixture from a full conversion and MUST have the exclusive heavy RAM slot.
--pretrained-torch checks that fixture in a separate process; never hold both
full ORT and Torch models in RAM at once. A fixture mismatch raises, and a JSON
report records whether weights were pretrained rather than claiming tiny ASR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import hashlib
import resource
import sys
import time
import wave
import subprocess

import numpy as np
import onnxruntime as ort
import torch
from safetensors.torch import save_file
from transformers import VoxtralRealtimeFeatureExtractor
from transformers.cache_utils import DynamicCache
from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import Audio8ASRInfiniteForConditionalGeneration

from backend import Audio8OnnxBackend, MappedInitializers, TokenizerAdapter, features, run_cached, session_options
from export import ExternalWeights, CHECKPOINT, REVISION, UPSTREAM, sha256, build_all, validate_shapes, verify_package, weight_header


def compare(report, name, actual, expected, tolerance=2e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance, err_msg=name)
    report[name] = {"max_absolute_error": error, "atol": tolerance, "rtol": tolerance,
                    "shape": list(actual.shape)}


def torch_encoder(model, conv, state, seen, attention_mask=None):
    positions = torch.arange(seen, seen + conv.shape[1])[None].expand(conv.shape[0], -1)
    out = model.audio_tower(inputs_embeds=torch.from_numpy(conv), position_ids=positions,
                            attention_mask=attention_mask, past_key_values=state, use_cache=True, return_dict=True)
    grouped = model.group_audio_hidden_states(out.last_hidden_state, frame_len=8)
    return model.multi_modal_projector(grouped), out.past_key_values


def torch_decoder(model, ids, audio, state, seen, delay, attention_mask=None):
    ids_tensor = torch.from_numpy(ids)
    fused = model.build_text_inputs_embeds(input_ids=ids_tensor, audio_embeds=audio)
    pos = torch.arange(seen, seen + ids.shape[1])[None].expand(ids.shape[0], -1)
    out = model.forward_language_model_with_delay(
        inputs_embeds=fused, position_ids=pos, past_key_values=state, use_cache=True,
        attention_mask=attention_mask if attention_mask is not None else torch.ones((ids.shape[0], seen + ids.shape[1]), dtype=torch.long),
        logits_to_keep=1, num_delay_tokens=torch.from_numpy(delay), frame_len=8)
    return out.logits, out.past_key_values


def fixture_inputs(cfg, batch=1):
    rng = np.random.default_rng(8901)
    # Shapes deliberately exercise prefill, nonempty past, and unequal chunks.
    for tokens in (3, 1, 2, 1):
        conv = rng.normal(0, 0.1, (batch, tokens * 8, cfg.audio_config.hidden_size)).astype(np.float32)
        ids = rng.integers(0, cfg.text_config.vocab_size, (batch, tokens), dtype=np.int64)
        yield conv, ids


def rolling_mask(seen, past, count, prefix=0, sliding=None):
    head = min(prefix, past)
    keys = torch.cat((torch.arange(head), torch.arange(seen - (past - head), seen + count)))
    queries = torch.arange(seen, seen + count)
    allowed = keys[None] <= queries[:, None]
    if sliding:
        allowed &= keys[None] > queries[:, None] - sliding
    return torch.where(allowed, 0.0, torch.finfo(torch.float32).min)[None, None]


def crop_torch(cache, limit, prefix=0):
    tail = limit - prefix
    for layer in cache.layers:
        if layer.keys.shape[2] > limit:
            for attr in ("keys", "values"):
                value = getattr(layer, attr)
                if prefix:
                    value = torch.cat((value[:, :, :prefix], value[:, :, -tail:]), 2)
                else:
                    value = value[:, :, -tail:].clone()
                setattr(layer, attr, value)


def check_rolling(model, backend, report):
    decoder = backend.new_step_decoder(14)
    audio_cache, text_cache = DynamicCache(), DynamicCache()
    seen_audio = seen_text = 0
    captured = {}
    original = backend.decoder_step
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["logits"] = result[0]
        return result
    backend.decoder_step = record
    rng = np.random.default_rng(321)
    generated = []
    try:
        with torch.inference_mode():
            for tick in range(41):
                count = 13 if tick == 0 else 1
                window = rng.normal(0, 0.1, count * 2560 + (40 if tick == 0 else 880)).astype(np.float32)
                ids = rng.integers(0, 128, count).tolist()
                actual = decoder.step(window, ids)
                conv = model.audio_tower.embedder(torch.from_numpy(features(window, backend.mel_filters)))[:, -count * 8:]
                audio, audio_cache = torch_encoder(model, conv.numpy(), audio_cache, seen_audio,
                    rolling_mask(seen_audio, audio_cache.get_seq_length(), count * 8, sliding=25))
                logits, text_cache = torch_decoder(model, np.array([ids], np.int64), audio, text_cache, seen_text,
                    np.array([3], np.float32), rolling_mask(seen_text, text_cache.get_seq_length(), count, prefix=13))
                compare(report, f"rolling_logits_{tick}", captured["logits"], logits.numpy())
                logits[0, -1, backend.special_ids["eos_token_id"]] = -torch.inf
                assert actual == int(logits[0, -1].argmax()), "Rolling greedy token mismatch"
                generated.append(actual)
                seen_audio += count * 8
                seen_text += count
                crop_torch(audio_cache, 25)
                crop_torch(text_cache, 14, prefix=13)
                stats = decoder.cache_stats()
                assert stats["audio_cache_frames"] <= 25 and stats["text_cache_tokens"] <= 14
                assert stats["absolute_audio_position"] == seen_audio and stats["absolute_text_position"] == seen_text
                if tick >= 2:
                    assert all(v.base is None for v in decoder.text_state.values.values()), "Crop retained oversized storage"
            report["rolling_cache_parity"] = {"tokens": generated, "equal": True, "stats": decoder.cache_stats(), "pretrained": False}
            session_ids = {k: id(v) for k, v in backend.sessions.items()}
            decoder.reset()
            assert decoder.cache_stats()["cache_bytes"] == 0
            assert session_ids == {k: id(v) for k, v in backend.sessions.items()}
    finally:
        backend.decoder_step = original


def check_tiny(checkpoint, output):
    verify_package()
    unvalidated = {"format_version": 1, "engine": "audio8", "backend": "onnx", "checkpoint": CHECKPOINT,
                   "revision": REVISION, "upstream_revision": UPSTREAM, "package_revision": UPSTREAM,
                   "source_git_verified": True, "host": {"frame_len": 8}, "dtype": "float32",
                   "validation": {"pretrained_numerical_parity": False}}
    gate = output / "unvalidated_descriptor.json"
    gate.write_text(json.dumps(unvalidated))
    try:
        Audio8OnnxBackend(gate)
    except ValueError as error:
        assert "have not passed" in str(error)
    else:
        raise AssertionError("Default runtime accepted unvalidated artifacts")
    torch.set_num_threads(2)
    torch.manual_seed(8901)
    cfg = Audio8ASRInfiniteConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.semantic_vad_horizons_seconds = None
    cfg.audio_config.hidden_size, cfg.audio_config.intermediate_size = 24, 48
    cfg.audio_config.num_hidden_layers, cfg.audio_config.num_attention_heads, cfg.audio_config.head_dim = 2, 4, 8
    cfg.audio_config.sliding_window = 25  # forces cache rollover during test
    cfg.text_config.hidden_size, cfg.text_config.intermediate_size = 32, 64
    cfg.text_config.num_hidden_layers, cfg.text_config.num_attention_heads, cfg.text_config.num_key_value_heads = 2, 4, 2
    cfg.text_config.vocab_size = 128
    cfg.text_config.layer_types = ["full_attention"] * 2
    cfg.text_config.sliding_window = None
    cfg.text_config.use_sliding_window = False
    cfg.hidden_size, cfg.projection_size = 32, 24 * 8
    cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id = 1, 2, 0
    cfg.text_config.bos_token_id, cfg.text_config.eos_token_id, cfg.text_config.pad_token_id = 1, 2, 0
    cfg.audio_config._attn_implementation = cfg.text_config._attn_implementation = "eager"
    model = Audio8ASRInfiniteForConditionalGeneration(cfg).eval()
    for parameter in model.parameters():
        parameter.data = parameter.data.bfloat16().float()
    state = {name: value.bfloat16().contiguous() for name, value in model.state_dict().items()
             if name != "language_model.lm_head.weight"}
    save_file(state, str(output / "tiny.safetensors"), metadata={"format": "pt"})
    header, start = weight_header(output / "tiny.safetensors")
    weights = ExternalWeights(output / "tiny.safetensors", header, start, output / "weights.bin")
    graphs = build_all(cfg.to_dict(), weights, output)
    mapped = MappedInitializers(output / "weights.bin")
    sessions = {name: mapped.session((output / data["file"]).read_bytes(), 2) for name, data in graphs.items()}
    assert len(mapped.values) == len(header)
    assert all(value.data_ptr() == mapped.views[name].ctypes.data for name, value in mapped.values.items())
    assert all(not value.flags.writeable for value in mapped.views.values())
    report = {"pretrained": False, "fixture": "untrained tiny Audio8; BF16 parameters promoted to FP32", "revision": REVISION}
    pinned_header, _ = weight_header(checkpoint / "model.safetensors")
    for name, mutate in (("missing", lambda h: h.pop(next(iter(h)))),
                         ("extra", lambda h: h.update({"semantic_vad_heads.fake.weight": next(iter(h.values()))})),
                         ("shape", lambda h: h.update({next(iter(h)): {**h[next(iter(h))], "shape": [1]}}))):
        invalid = dict(pinned_header)
        mutate(invalid)
        try:
            validate_shapes(checkpoint, invalid)
        except ValueError as error:
            assert "Strict ASR weight mismatch" in str(error)
        else:
            raise AssertionError(f"Strict metadata accepted {name} weights")
    report["strict_metadata_rejections"] = ["missing", "extra", "shape"]
    report["unvalidated_runtime_rejected"] = True
    report["mapped_initializers"] = {"count": len(mapped.values), "zero_copy_pointer_equal": True,
                                     "read_only": True, "runtime_dependencies_added": False}
    from tokenizers import Tokenizer
    adapted = TokenizerAdapter(Tokenizer.from_file(str(checkpoint / "tokenizer.json")))
    assert json.loads(adapted.backend_tokenizer.to_str())["decoder"]["type"] == "ByteLevel"
    assert adapted.token_to_id("<|im_end|>") in adapted.all_special_ids
    assert adapted.convert_ids_to_tokens(151645) == "<|im_end|>"
    assert adapted.convert_ids_to_tokens([151645]) == ["<|im_end|>"]
    report["tokenizer_adapter"] = True
    extractor = VoxtralRealtimeFeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
    waveform = np.random.default_rng(17).normal(0, 0.1, 13320).astype(np.float32)
    expected = extractor._torch_extract_fbank_features(torch.from_numpy(waveform)[None]).numpy()
    mel = features(waveform, extractor.mel_filters)
    compare(report, "numpy_frontend", mel, expected, 2e-5)
    with torch.inference_mode():
        conv_torch = model.audio_tower.embedder(torch.from_numpy(mel)).numpy()
        conv_ort = sessions["embedder"].run(["conv_embeds"], {"input_features": mel})[0]
        compare(report, "embedder", conv_ort, conv_torch)
        tokens_ort, tokens_torch = [], []
        for batch in (1, 2):
            audio_state, text_state, audio_cache, text_cache = None, None, None, None
            seen_audio, seen_text = 0, 0
            delay = np.arange(batch, dtype=np.float32) + 3
            for index, (conv, ids) in enumerate(fixture_inputs(cfg, batch)):
                audio_out, audio_state = run_cached(sessions["encoder"], cfg.to_dict()["audio_config"],
                                                    {"conv_embeds": conv}, audio_state, conv.shape[1], sliding=True)
                audio_torch, audio_cache = torch_encoder(model, conv, audio_cache, seen_audio)
                compare(report, f"encoder_b{batch}_step{index}", audio_out["audio_embeds"], audio_torch.numpy())
                decoder_out, text_state = run_cached(sessions["decoder"], cfg.to_dict()["text_config"],
                    {"input_ids": ids, "audio_embeds": audio_out["audio_embeds"], "num_delay_tokens": delay},
                    text_state, ids.shape[1])
                # Independent Torch encoder output tests accumulated end-to-end error.
                logits, text_cache = torch_decoder(model, ids, audio_torch, text_cache, seen_text, delay)
                compare(report, f"decoder_b{batch}_step{index}", decoder_out["logits"], logits.numpy())
                for layer in range(2):
                    for kind in ("key", "value"):
                        torch_kv = getattr(text_cache.layers[layer], "keys" if kind == "key" else "values").numpy()
                        compare(report, f"kv_b{batch}_step{index}_{layer}_{kind}",
                                text_state.values[f"past_{kind}_{layer}"], torch_kv)
                tokens_ort.extend(decoder_out["logits"].argmax(-1).reshape(-1).tolist())
                tokens_torch.extend(logits.argmax(-1).reshape(-1).tolist())
                seen_audio += conv.shape[1]
                seen_text += ids.shape[1]
        assert tokens_ort == tokens_torch, "Tiny greedy token mismatch"
        report["greedy_token_parity"] = {"tokens": tokens_ort, "equal": True, "pretrained": False}
    # Exercise the actual NumPy controller against upstream's Torch controller.
    # Tiny IDs/tokenizer are explicitly fixture-only; this is not an ASR claim.
    from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode
    class FixtureTokenizer:
        def decode(self, ids, skip_special_tokens=True, **kwargs):
            return " ".join(str(v) for v in ids if not skip_special_tokens or v not in range(8))
    backend = Audio8OnnxBackend.__new__(Audio8OnnxBackend)
    backend.sessions, backend.audio_config, backend.text_config = sessions, cfg.to_dict()["audio_config"], cfg.to_dict()["text_config"]
    backend.mel_filters, backend.tokenizer = extractor.mel_filters, FixtureTokenizer()
    backend.descriptor = {"host": {"max_chunk_samples": 384000}}
    backend.ids = {"bos": 1, "eos": 2, "pad": 0, "streaming_pad": 3, "streaming_word": 4, "en": 6, "zh": 7}
    backend.special_ids = {k + "_token_id": v for k, v in backend.ids.items() if k not in ("en", "zh")}
    backend.language_ids = {k: backend.ids[k] for k in ("en", "zh")}
    backend.np, backend.device, backend.dtype = np, "onnx:cpu", "float32"
    check_rolling(model, backend, report)
    short_audio = waveform[:2577]  # partial final frame shorter than STFT lookahead
    generated_trace = []
    decoder_step = backend.decoder_step
    def recording_decoder(*args, **kwargs):
        logits, state = decoder_step(*args, **kwargs)
        scores = logits[0, -1].copy()
        scores[backend.ids["eos"]] = -np.inf
        generated_trace.append(int(scores.argmax()))
        return logits, state
    backend.decoder_step = recording_decoder
    numpy_text = backend.transcribe_chunk(short_audio, "en")
    reference = simulated_streaming_greedy_decode(
        model=model, tokenizer=backend.tokenizer, feature_extractor=extractor,
        waveform=np.pad(short_audio, (0, (-len(short_audio)) % 2560)),
        language_token_id=6, special_ids={k + "_token_id": v for k, v in backend.ids.items() if k not in ("en", "zh")},
        audio_config=SimpleNamespace(raw_audio_samples_per_token=2560, streaming_n_left_pad_tokens=9, sampling_rate=16000),
        num_delay_tokens=3, frame_len=8, right_pad_text_tokens=10, dtype=torch.float32, device=torch.device("cpu"))
    assert numpy_text == reference["final_text"], "Streaming host-controller token text mismatch"
    assert generated_trace == reference["generated_token_ids"], "Streaming controller greedy token mismatch"
    report["streaming_controller_parity"] = {"equal": True, "tokens": list(generated_trace),
                                             "fixture_token_text": numpy_text, "pretrained": False}
    assert backend.decode(short_audio.astype("<f4").tobytes(), "en") == numpy_text
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(report) - 3, "report": str(output / "report.json"), "pretrained": False}))


def fingerprint(descriptor):
    d = json.loads(Path(descriptor).read_text())
    d.pop("validation", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def read_wave(path):
    with wave.open(str(path)) as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Fixture WAV must be mono PCM16 at 16kHz")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def check_shared_scheduler(backend, samples, expected, fixture, output):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audio8_streaming import StreamingDecoder
    decoder = StreamingDecoder(backend, "en", window_tokens=64)
    stepper = decoder._backend_step
    original = stepper.step
    tokens, observations = [], []
    def record(audio, ids):
        value = original(audio, ids)
        tokens.append(value)
        observations.append(stepper.cache_stats())
        return value
    stepper.step = record
    events = []
    for first in range(0, len(samples), 2560):
        events.extend(decoder.push_pcm(samples[first:first + 2560].astype("<f4").tobytes()))
    events.extend(decoder.finish())
    text = "".join(v["text"] for v in events if v["type"] == "segment")
    assert tokens == expected["generated_token_ids"], "Shared scheduler differs from canonical token clock"
    assert text == expected["final_text"], "Shared scheduler transcript mismatch"
    fixture["shared_tokens"] = np.array(tokens, np.int64)
    fixture["shared_text"] = np.array(text)
    report = {"token_parity": True, "text_parity": True, "text": text, "tokens": tokens,
              "window_tokens": 64, "events": events, "final_step_cache": observations[-1]}
    (output / "shared_scheduler.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Shared live scheduler token/text parity passed", flush=True)
    return report


def pretrained_ort(descriptor, output, audio_path, calibrate_runtime=False):
    started = time.perf_counter()
    d = json.loads(descriptor.read_text())
    if sha256(descriptor.parent / "weights.bin") != d["weights"]["sha256"]:
        raise ValueError("Converted external weight SHA256 mismatch")
    print("Loading full ORT graphs", flush=True)
    backend = Audio8OnnxBackend(descriptor, threads=4, allow_unvalidated=True)
    cfg = Audio8ASRInfiniteConfig.from_pretrained(backend.root, local_files_only=True)
    wave = np.random.default_rng(19).normal(0, 0.1, 13320).astype(np.float32)
    mel = features(wave, backend.mel_filters)
    conv_embedder = backend.sessions["embedder"].run(["conv_embeds"], {"input_features": mel})[0]
    fixture = {"waveform": wave, "mel": mel, "conv_embedder": conv_embedder}
    audio_state, text_state = None, None
    for index, (conv, ids) in enumerate(fixture_inputs(cfg)):
        audio_out, audio_state = run_cached(backend.sessions["encoder"], backend.audio_config,
                                            {"conv_embeds": conv}, audio_state, conv.shape[1], sliding=True)
        logits, text_state = backend.decoder_step(ids, audio_out["audio_embeds"], text_state)
        fixture.update({f"conv_{index}": conv, f"ids_{index}": ids,
                        f"audio_{index}": audio_out["audio_embeds"], f"logits_{index}": logits})
        fixture.update({f"kv_{index}_{name}": value for name, value in text_state.values.items()})
    print("Full ORT numerical fixture generated; decoding repository short.wav", flush=True)
    actual_wave = read_wave(audio_path)
    decoded = backend.decode_waveform(actual_wave, "en")
    if not decoded["final_text"].strip():
        raise AssertionError("Pretrained ONNX fixture produced no speech text")
    fixture.update({"speech_waveform": actual_wave, "speech_tokens": np.array(decoded["generated_token_ids"], np.int64),
                    "speech_text": np.array(decoded["final_text"]), "descriptor_fingerprint": np.array(fingerprint(descriptor))})
    # Include a final partial frame shorter than lookahead to catch lost tail clocks.
    partial = actual_wave[:2560 + 17]
    partial_decoded = backend.decode_waveform(partial, "en")
    fixture.update({"partial_waveform": partial, "partial_tokens": np.array(partial_decoded["generated_token_ids"], np.int64),
                    "partial_text": np.array(partial_decoded["final_text"])})
    shared = check_shared_scheduler(backend, actual_wave, decoded, fixture, output)
    calibration = {"status": "not_run", "reason": "optional heavy calibration requires --calibrate"}
    if calibrate_runtime:
        print("Running real saturated-cache ORT calibration", flush=True)
        from audio8_streaming import calibrate
        calibration = calibrate(backend)
        (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    np.savez(output / "pretrained_fixture.npz", **fixture)
    report = {"pretrained": True, "revision": REVISION, "descriptor": str(descriptor),
              "shared_scheduler": shared, "calibration": calibration,
              "speech_source": str(audio_path), "speech_source_sha256": sha256(audio_path),
              "speech_text": decoded["final_text"], "speech_tokens": decoded["generated_token_ids"],
              "partial_token_count": len(partial_decoded["generated_token_ids"]),
              "elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": peak_rss(),
              "weights_bytes": backend.descriptor["weights"]["size"]}
    (output / "ort_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    print(output / "pretrained_fixture.npz", flush=True)


def pretrained_torch(checkpoint, fixture_path, output, descriptor):
    started = time.perf_counter()
    fixture = np.load(fixture_path, allow_pickle=False)
    if str(fixture["descriptor_fingerprint"]) != fingerprint(descriptor):
        raise ValueError("Fixture provenance does not match the graph descriptor")
    if sha256(checkpoint / "model.safetensors") != json.loads(descriptor.read_text())["checkpoint_sha256"]:
        raise ValueError("Torch reference checkpoint SHA256 mismatch")
    verify_package()
    header, _ = weight_header(checkpoint / "model.safetensors")
    validate_shapes(checkpoint, header)
    cfg = Audio8ASRInfiniteConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.semantic_vad_horizons_seconds = None
    cfg.audio_config._attn_implementation = cfg.text_config._attn_implementation = "eager"
    torch.set_num_threads(4)
    print("Loading serial full FP32 Torch reference", flush=True)
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        checkpoint, config=cfg, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True).eval()
    report = {"pretrained": True, "revision": REVISION, "fixture": str(fixture_path)}
    audio_cache, text_cache = None, None
    seen_audio, seen_text = 0, 0
    with torch.inference_mode():
        extractor = VoxtralRealtimeFeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
        mel = extractor._torch_extract_fbank_features(torch.from_numpy(fixture["waveform"])[None]).numpy()
        compare(report, "numpy_frontend", fixture["mel"], mel, 2e-5)
        conv_embedder = model.audio_tower.embedder(torch.from_numpy(fixture["mel"]))
        compare(report, "embedder", fixture["conv_embedder"], conv_embedder.numpy(), 2e-4)
        for index in range(4):
            conv, ids = fixture[f"conv_{index}"], fixture[f"ids_{index}"]
            audio, audio_cache = torch_encoder(model, conv, audio_cache, seen_audio)
            compare(report, f"encoder_step{index}", fixture[f"audio_{index}"], audio.numpy(), 2e-3)
            logits, text_cache = torch_decoder(model, ids, audio, text_cache, seen_text, np.array([3], np.float32))
            compare(report, f"decoder_step{index}", fixture[f"logits_{index}"], logits.numpy(), 5e-3)
            assert fixture[f"logits_{index}"].argmax(-1).tolist() == logits.argmax(-1).tolist(), "Pretrained greedy token mismatch"
            for layer in range(cfg.text_config.num_hidden_layers):
                for kind in ("key", "value"):
                    torch_kv = getattr(text_cache.layers[layer], "keys" if kind == "key" else "values").numpy()
                    compare(report, f"kv_step{index}_{layer}_{kind}",
                            fixture[f"kv_{index}_past_{kind}_{layer}"], torch_kv, 2e-3)
            seen_audio += conv.shape[1]
            seen_text += ids.shape[1]
    print("Full frontend/encoder/decoder/KV numerical parity passed; comparing speech clocks", flush=True)
    from transformers import AutoTokenizer
    from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch
    from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
        resolve_qwen_streaming_special_token_ids, resolve_qwen_language_token_id)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    special_ids = resolve_qwen_streaming_special_token_ids(tokenizer)
    for label in ("speech", "partial"):
        expected = simulated_streaming_greedy_decode_batch(
            model=model, tokenizer=tokenizer, feature_extractor=extractor,
            waveforms=[fixture[f"{label}_waveform"]], language_token_ids=[resolve_qwen_language_token_id(tokenizer, "en")],
            special_ids=special_ids,
            audio_config=SimpleNamespace(raw_audio_samples_per_token=2560, streaming_n_left_pad_tokens=9, sampling_rate=16000),
            num_delay_tokens=[3], frame_len=8, right_pad_text_tokens=10, dtype=torch.float32,
            device=torch.device("cpu"), max_new_tokens=None)[0]
        assert expected["generated_token_ids"] == fixture[f"{label}_tokens"].tolist(), f"{label}: pretrained generated-token mismatch"
        assert expected["final_text"] == str(fixture[f"{label}_text"]), f"{label}: pretrained transcript mismatch"
        report[f"{label}_parity"] = {"token_parity": True, "text_parity": True,
                                     "text": expected["final_text"], "tokens": expected["generated_token_ids"]}
    report.update({"greedy_token_parity": True, "elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": peak_rss()})
    report_path = output / "pretrained_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    d = json.loads(descriptor.read_text())
    d["validation"].update({"pretrained_numerical_parity": True, "pretrained_token_parity": True,
                            "pretrained_text_parity": True, "report_sha256": sha256(report_path),
                            "fixture_sha256": sha256(fixture_path)})
    staged = descriptor.with_suffix(".validated.json")
    staged.write_text(json.dumps(d, indent=2) + "\n")
    staged.replace(descriptor)
    # Selector marker is deliberately small, strict, and written only AFTER parity.
    marker = descriptor.parent / "Audio8-ASR-Infinite.audio8.json"
    marker.write_text(json.dumps({"engine": "audio8", "checkpoint": CHECKPOINT, "revision": REVISION,
                                  "backend": "onnx", "artifacts": "."}, indent=2) + "\n")
    print(json.dumps({"passed": True, "pretrained": True, "report": str(report_path), "marker": str(marker)}), flush=True)


def rust_bridge(descriptor, audio_path, output):
    d = json.loads(descriptor.read_text())
    if d.get("validation", {}).get("pretrained_numerical_parity") is not True:
        raise ValueError("Rust bridge requires parity-validated ONNX artifacts")
    repo = Path(__file__).resolve().parents[5]
    command = [str(repo / "server/target/debug/vibe-server"), "transcribe",
               str(descriptor.parent / "Audio8-ASR-Infinite.audio8.json"), str(audio_path),
               "--language", "en", "--threads", "4"]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=900)
    (output / "stdout.txt").write_text(result.stdout)
    (output / "stderr.txt").write_text(result.stderr)
    report = {"command": command, "returncode": result.returncode, "elapsed_seconds": time.perf_counter() - started,
              "expected_text": "Experience proves this", "transcript_present": "Experience proves this" in result.stdout}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    assert result.returncode == 0, "Rust bridge failed: " + result.stderr[-2000:]
    assert report["transcript_present"], "Rust bridge did not return expected repository transcript"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new workspace-only artifact directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pretrained-ort", type=Path, metavar="DESCRIPTOR")
    mode.add_argument("--pretrained-torch", type=Path, metavar="FIXTURE_NPZ")
    mode.add_argument("--rust-bridge", action="store_true")
    parser.add_argument("--descriptor", type=Path, help="conversion descriptor required for Torch parity/promotion")
    parser.add_argument("--audio", type=Path, default=Path("server/fixtures/short.wav"))
    parser.add_argument("--calibrate", action="store_true", help="opt-in heavy shared saturated-cache calibration")
    args = parser.parse_args()
    if (args.pretrained_torch or args.rust_bridge) and not args.descriptor:
        parser.error("Torch parity and Rust bridge modes require --descriptor")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.pretrained_ort:
        pretrained_ort(args.pretrained_ort, args.output, args.audio, args.calibrate)
    elif args.pretrained_torch:
        pretrained_torch(args.checkpoint, args.pretrained_torch, args.output, args.descriptor)
    elif args.rust_bridge:
        rust_bridge(args.descriptor, args.audio, args.output)
    else:
        check_tiny(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
