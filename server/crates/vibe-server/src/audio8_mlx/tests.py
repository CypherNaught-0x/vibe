# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mlx==0.32.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-lm==0.31.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-audio==0.4.4; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "transformers==5.5.3", "numpy==2.5.3", "safetensors==0.8.0",
#   "huggingface-hub==1.33.0", "tokenizers==0.22.2",
# ]
# ///
"""Light numerical/contract tests; --full needs parent's serialized heavy slot.

First: uv run reference.py --output-dir plans/audio8_mlx/artifacts/tiny
Then:  uv run tests.py
Full:  uv run reference.py --full --output-dir plans/audio8_mlx/artifacts/parity
       uv run convert.py --output-dir plans/audio8_mlx/artifacts/fp16
       uv run tests.py --full --model-dir plans/audio8_mlx/artifacts/fp16
                       --reference-dir plans/audio8_mlx/artifacts/parity
Only passing full pretrained token AND final-text parity enables the backend.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio8_mlx.common import CHECKPOINT, REVISION, PACKAGE_REVISION, WORKSPACE, require_apple_silicon, verify_dependencies, workspace_output

REFERENCE = WORKSPACE / "plans/audio8_mlx/artifacts/tiny"
METRICS = {}


class Contracts(unittest.TestCase):
    def test_non_apple_rejected_before_import(self):
        # backend.py's top-level import is safe without MLX or Torch.
        from audio8_mlx.backend import MlxStepDecoder
        with patch("audio8_mlx.common.sys.platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "macOS on Apple Silicon"):
                MlxStepDecoder("never-read")
        with patch("audio8_mlx.common.platform.machine", return_value="x86_64"):
            with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                require_apple_silicon()

    def test_unvalidated_model_refused_before_loading(self):
        from audio8_mlx.backend import MlxStepDecoder
        with patch("audio8_mlx.backend.read_descriptor", return_value={"validation": {"status": "unvalidated"}}):
            with self.assertRaisesRegex(RuntimeError, "FP16 fixture"):
                MlxStepDecoder("never-read")

    def test_portable_conversion_path_and_marker_contract(self):
        from audio8_mlx.common import explicit_output_directory, write_validated_marker, MARKER_NAME
        outside = WORKSPACE.parent / "explicit-portable-audio8"
        self.assertEqual(explicit_output_directory(outside), outside.resolve())
        with self.assertRaisesRegex(ValueError, "explicit"):
            explicit_output_directory("")
        with self.assertRaisesRegex(ValueError, "non-root"):
            explicit_output_directory("/")
        with patch("audio8_mlx.common.read_descriptor", return_value={"validation": {"status": "unvalidated"}}):
            with self.assertRaisesRegex(ValueError, "full FP16"):
                write_validated_marker(REFERENCE / "marker-contract")
        validated = {"validation": {"status": "fp16-parity-passed", "token_parity": True, "text_parity": True}}
        # Exercise the publishing contract WITHOUT actually creating a marker:
        # only full pretrained validation is allowed to publish real artifacts.
        with patch("audio8_mlx.common.read_descriptor", return_value=validated), patch.object(Path, "write_text") as writer:
            path = write_validated_marker(REFERENCE / "marker-contract")
            self.assertEqual(path.name, MARKER_NAME)
            self.assertEqual(json.loads(writer.call_args.args[0]), {"engine": "audio8", "checkpoint": CHECKPOINT,
                "revision": REVISION, "backend": "mlx", "artifacts": "."})

    def test_runtime_rejects_unsupported_controls_without_model_loading(self):
        from audio8_mlx.backend import MlxRuntime
        with patch("audio8_mlx.backend.MlxStepDecoder") as loader:
            for kwargs, message in (({"no_gpu": True}, "no_gpu"), ({"gpu_device": 1}, "default Apple Metal"),
                                    ({"threads": -1}, "non-negative")):
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    MlxRuntime("never-read", **kwargs)
            loader.assert_not_called()

    def test_output_must_be_workspace(self):
        with self.assertRaisesRegex(ValueError, "inside"):
            workspace_output("/tmp/audio8-mlx")
        self.assertTrue(workspace_output("plans/audio8_mlx/artifacts/test").is_relative_to(WORKSPACE))

    def test_only_unused_heads_excluded(self):
        from audio8_mlx.model import map_weight
        import mlx.core as mx
        config = json.loads((REFERENCE / "tiny-config.json").read_text())
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            map_weight("semantic_vad_heads.0.weight", mx.zeros((8, 16)), config)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            map_weight("audio_tower.layers.0.unexpected.weight", mx.zeros((16, 16)), config)


class NumericalParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_dependencies()
        import mlx.core as mx
        import numpy as np
        from audio8_mlx.model import Model, map_weight
        cls.mx, cls.np = mx, np
        cls.config = json.loads((REFERENCE / "tiny-config.json").read_text())
        cls.models = {}
        for label in ("fp32", "fp16"):
            model = Model(cls.config)
            weights = {new: value for name, original in mx.load(str(REFERENCE / f"tiny-{label}.safetensors")).items()
                       for new, value in [map_weight(name, original, cls.config)]}
            model.load_weights(list(weights.items()), strict=True)
            model.eval()
            mx.eval(model.parameters())
            cls.models[label] = model

    def check_array(self, name, actual, expected, tolerance):
        self.mx.eval(actual)
        array = self.np.array(actual.astype(self.mx.float32))
        self.assertEqual(array.shape, expected.shape, name)
        error = float(self.np.abs(array - expected).max())
        METRICS[name] = error
        self.assertLessEqual(error, tolerance, f"{name}: max absolute error {error}")

    def test_fp32_and_fp16_cached_layers(self):
        for label, model in self.models.items():
            reference = self.np.load(REFERENCE / f"tiny-{label}-reference.npz")
            dtype = self.mx.float16 if label == "fp16" else self.mx.float32
            tolerance = 0.003 if label == "fp16" else 0.00002
            self.check_array(f"{label}/t_cond", model.build_t_cond(3, 8, dtype), reference["t_cond"], tolerance)
            scales = model.delay_scales(3, 8, dtype)
            audio_cache, text_cache = model.audio_tower.make_cache(), model.make_text_cache()
            for index in range(3):
                features = self.mx.array(reference[f"features_{index}"], dtype=dtype)
                conv = model.audio_tower.embedder(features)
                self.check_array(f"{label}/conv_{index}", conv, reference[f"conv_{index}"], tolerance)
                hidden = model.audio_tower(conv, audio_cache)
                self.check_array(f"{label}/hidden_{index}", hidden, reference[f"hidden_{index}"], tolerance)
                projected = model.project(hidden, 8)
                self.check_array(f"{label}/projector_{index}", projected, reference[f"audio_{index}"], tolerance)
                ids = self.mx.array(reference[f"ids_{index}"])
                fused = model.language_model.model.embed_tokens(ids) + projected
                logits = model.language_model(fused, scales, text_cache)
                self.check_array(f"{label}/logits_{index}", logits, reference[f"logits_{index}"], tolerance)
                self.assertEqual(int(self.mx.argmax(logits).item()), int(reference[f"logits_{index}"].argmax()))
            self.assertEqual(audio_cache[0].offset, 32)
            self.assertEqual(audio_cache[0].size(), 11)
            self.assertEqual(text_cache[0].offset, 4)
            for frame_len in (4, 6, 8):
                actual = model.project(self.mx.array(reference[f"group_input_{frame_len}"], dtype=dtype), frame_len)
                self.check_array(f"{label}/frame_group_{frame_len}", actual, reference[f"group_project_{frame_len}"], tolerance)

    def test_frontend_against_pinned_transformers(self):
        from mlx_audio.stt.models.voxtral_realtime.audio import compute_mel_filters, compute_mel_spectrogram
        reference = self.np.load(REFERENCE / "tiny-fp32-reference.npz")
        filters = self.mx.array(compute_mel_filters(), dtype=self.mx.float32)
        mel = compute_mel_spectrogram(self.mx.array(reference["samples"]), filters)[None]
        self.check_array("frontend/log_mel", mel, reference["mel"], 0.0001)

    def test_future_audio_is_masked(self):
        model = self.models["fp32"]
        conv = self.mx.arange(16 * 16, dtype=self.mx.float32).reshape(1, 16, 16) / 100
        changed = self.mx.concatenate([conv[:, :8], conv[:, 8:] + 10], axis=1)
        original, perturbed = model.audio_tower(conv), model.audio_tower(changed)
        self.check_array("causal/no_future_audio", original[:, :8], self.np.array(perturbed[:, :8]), 0.000001)

    def test_weight_roundtrip(self):
        from mlx.utils import tree_flatten
        from audio8_mlx.model import Model
        path = REFERENCE / "tiny-mlx-roundtrip.safetensors"
        self.mx.save_safetensors(str(path), dict(tree_flatten(self.models["fp16"].parameters())), metadata={"format": "mlx"})
        restored = Model(self.config)
        restored.load_weights(list(self.mx.load(str(path)).items()), strict=True)
        reference = self.np.load(REFERENCE / "tiny-fp16-reference.npz")
        self.check_array("roundtrip/t_cond", restored.build_t_cond(3, 8, self.mx.float16), reference["t_cond"], 0.003)

    def test_executable_sharded_converter_on_small_model(self):
        from uuid import uuid4
        from audio8_mlx.convert import convert
        from audio8_mlx.common import read_descriptor
        from audio8_mlx.model import Model
        output = REFERENCE / f"converter-{uuid4().hex}"
        files = {"model.safetensors": REFERENCE / "tiny-fp16.safetensors",
                 "config.json": REFERENCE / "tiny-config.json"}
        for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
            files[name] = REFERENCE / "tiny-config.json"  # static-copy test only
        config = json.loads((REFERENCE / "tiny-config.json").read_text())
        with patch("audio8_mlx.convert.inspect_checkpoint", return_value=(config, {"mapping": "tiny-test"})):
            descriptor = convert(files, output, shard_bytes=16384)
        self.assertGreater(len(descriptor["weight_files"]), 1)
        self.assertIsNone(descriptor["model_config"]["semantic_vad_horizons_seconds"])
        self.assertEqual(read_descriptor(output)["backend"], "mlx")
        weights = {}
        for name in descriptor["weight_files"]:
            weights.update(self.mx.load(str(output / name)))
        restored = Model(config)
        restored.load_weights(list(weights.items()), strict=True)
        reference = self.np.load(REFERENCE / "tiny-fp16-reference.npz")
        self.check_array("converter/t_cond", restored.build_t_cond(3, 8, self.mx.float16), reference["t_cond"], 0.003)
        with self.assertRaisesRegex(ValueError, "empty"):
            with patch("audio8_mlx.convert.inspect_checkpoint", return_value=(config, {})):
                convert(files, output)

    def test_actual_backend_step_and_tokenizer_adapter(self):
        from uuid import uuid4
        from tokenizers import Tokenizer, models, decoders
        from audio8_mlx.convert import convert
        from audio8_mlx.backend import MlxStepDecoder, MlxRuntime
        root = REFERENCE / f"backend-{uuid4().hex}"
        source, output = root / "source", root / "mlx"
        source.mkdir(parents=True)
        tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[STREAMING_PAD]", "[STREAMING_WORD]",
                  "[LANGUAGE_EN]", "[LANGUAGE_ZH]"] + [f"word{index}" for index in range(7, 32)]
        tokenizer = Tokenizer(models.WordLevel({token: index for index, token in enumerate(tokens)}, unk_token=tokens[0]))
        tokenizer.add_special_tokens(tokens[:7])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save(str(source / "tokenizer.json"))
        (source / "tokenizer_config.json").write_text("{}")
        (source / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 16000, "feature_size": 128,
            "hop_length": 160, "n_fft": 400, "global_log_mel_max": 1.5}))
        files = {name: source / name for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")}
        files["model.safetensors"] = REFERENCE / "tiny-fp16.safetensors"
        config = json.loads((REFERENCE / "tiny-config.json").read_text())
        config["audio_config"]["sliding_window"] = 112  # canonical prefill, tiny weights
        with patch("audio8_mlx.convert.inspect_checkpoint", return_value=(config, {"mapping": "tiny-test"})):
            convert(files, output)
        decoder = MlxStepDecoder(output, text_window_tokens=16, keep_prompt_tokens=2, allow_unvalidated=True)
        prompt = [decoder.special_ids["bos_token_id"], decoder.language_ids["en"]] + [decoder.special_ids["streaming_pad_token_id"]] * 11
        token = decoder.step(self.np.zeros(13 * 2560 + 40, dtype=self.np.float32), prompt)
        self.assertNotEqual(token, decoder.special_ids["eos_token_id"])
        token = decoder.step(self.np.zeros(2560 + 840 + 40, dtype=self.np.float32), [token])
        self.assertEqual(decoder.cache_stats()["audio_position"], 112)
        self.assertEqual(decoder.cache_stats()["text_position"], 14)
        self.assertEqual(decoder.tokenizer.convert_ids_to_tokens([1, 7]), [tokens[1], tokens[7]])
        self.assertEqual(decoder.decode_visible_text([3, 7, 2, 8]), "word7")
        self.assertEqual(decoder.descriptor["validation"]["status"], "unvalidated")
        result = decoder.decode_waveform(self.np.zeros(1000, dtype=self.np.float32), "en")
        self.assertEqual(len(result["generated_token_ids"]), 11)
        self.assertEqual(decoder.cache_stats()["text_position"], 23)
        self.assertEqual(decoder.cache_stats()["text_kv_tokens"], 16)
        from copy import deepcopy
        checked = deepcopy(decoder.descriptor)
        checked["validation"] = {"status": "fp16-parity-passed", "token_parity": True, "text_parity": True}
        # Simulate the checked descriptor only; don't mutate it or publish a
        # marker for this random fixture. Runtime loads this tiny model ONCE.
        with patch("audio8_mlx.backend.read_descriptor", return_value=checked):
            runtime = MlxRuntime(output, no_gpu=False, gpu_device=0, threads=4)
        self.assertTrue(runtime.source_git_verified)
        self.assertEqual(runtime.backend, "mlx")
        with patch.object(self.mx, "load", side_effect=AssertionError("duplicate weights load")):
            first, second = runtime.new_step_decoder(16), runtime.new_step_decoder(32)
            self.assertIs(first.model, runtime.model)
            self.assertIs(second.model, first.model)
            self.assertIsNot(first.audio_cache[0], second.audio_cache[0])
            self.assertIsNot(first.text_cache[0], second.text_cache[0])
            next_token = first.step(self.np.zeros(13 * 2560 + 40, dtype=self.np.float32), prompt)
            # Calibration's synthetic multi-token ticks after a live prefill.
            for width in (3, 4, 8):
                next_token = first.step(self.np.zeros(width * 2560 + 840 + 40, dtype=self.np.float32), [next_token] * width)
            first.synchronize()
            stats = first.cache_stats()
            self.assertEqual(stats["absolute_text_position"], 28)
            self.assertEqual(stats["absolute_audio_position"], 224)
            self.assertLessEqual(stats["text_cache_tokens"], 16)
            self.assertLessEqual(stats["audio_cache_frames"], 112)
            self.assertEqual(stats["cache_bytes"], stats["audio_cache_bytes"] + stats["text_cache_bytes"])
            self.assertGreater(stats["cache_bytes"], 0)
            self.assertEqual(second.cache_stats()["absolute_text_position"], 0)
            self.assertEqual(second.cache_stats()["cache_bytes"], 0)
            self.assertIsInstance(runtime.decode(self.np.zeros(1000, dtype="<f4").tobytes(), "en"), str)
            self.assertEqual(runtime.decode(b"", "en"), "")
            with self.assertRaisesRegex(ValueError, "float32"):
                runtime.decode(b"bad", "en")
            with self.assertRaisesRegex(ValueError, "finite"):
                runtime.decode(self.np.array([float("nan")], dtype="<f4").tobytes(), "en")
            runtime.set_threads(0)
            runtime.synchronize()

    def test_explicit_rolling_text_cache_preserves_positions(self):
        model = self.models["fp32"]
        cache = model.make_text_cache(max_tokens=5, keep=2)
        scales = model.delay_scales(3, 8, self.mx.float32)
        for index in range(12):
            embeddings = model.language_model.model.embed_tokens(self.mx.array([[index % 32]]))
            logits = model.language_model(embeddings, scales, cache)
            self.mx.eval(logits)
        self.assertEqual(cache[0].offset, 12)
        self.assertEqual(cache[0].size(), 5)
        self.assertEqual(cache[0].keep, 2)
        with self.assertRaisesRegex(ValueError, "window"):
            model.make_text_cache(max_tokens=2, keep=2)


def full_numeric(decoder, samples, reference_dir, generated):
    """Independent teacher-forced first 3 real speech windows, shared weights."""
    import numpy as np
    from audio8_mlx.backend import MlxStepDecoder
    reference = np.load(reference_dir / "full-numerical-reference.npz")
    probe = MlxStepDecoder.from_loaded(decoder)
    spt = 2560
    real = max(1, (len(samples) + spt - 1) // spt)
    stream = np.zeros((9 + real + 3 + 1 + 10) * spt, dtype=np.float32)
    stream[9 * spt:9 * spt + len(samples)] = samples
    start, end = 0, 13 * spt
    ids = [probe.special_ids["bos_token_id"], probe.language_ids["en"]] + [probe.special_ids["streaming_pad_token_id"]] * 11
    metrics = {}
    mx = probe.mx
    for index in range(3):
        window = stream[max(start - 840, 0):end + 40]
        mel = probe.compute_mel(mx.array(window), probe.mel_filters).astype(mx.float16)
        conv = probe.model.audio_tower.embedder(mel[None])[:, -len(ids) * 8:, :]
        hidden = probe.model.audio_tower(conv, probe.audio_cache)
        audio = probe.model.project(hidden, 8)
        embeddings = probe.model.language_model.model.embed_tokens(mx.array([ids])) + audio
        logits = probe.model.language_model(embeddings, probe.scales, probe.text_cache)
        for stage, value in (("conv", conv), ("hidden", hidden), ("logits", logits)):
            mx.eval(value)
            actual = np.array(value.astype(mx.float32))
            expected = reference[f"{stage}_{index}"]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise AssertionError(f"Full numerical {stage}_{index} shape/finiteness mismatch")
            difference = actual.astype(np.float64) - expected
            rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
            nrmse = float(np.sqrt(np.mean(difference ** 2)) / max(rms, 1e-12))
            cosine = float(np.dot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)) /
                           max(np.linalg.norm(actual.astype(np.float64)) * np.linalg.norm(expected.astype(np.float64)), 1e-12))
            metrics[f"{stage}_{index}"] = {"max_absolute_error": float(np.abs(difference).max()),
                                           "normalized_rmse": nrmse, "cosine_similarity": cosine,
                                           "passed": nrmse <= 0.01 and cosine >= 0.9999}
        start, end = end, end + spt
        ids = [generated[index]]  # reference feedback prevents divergent-ID comparisons
    (reference_dir / "full-mlx-numerical.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def full(model_dir, reference_dir):
    from audio8_mlx.backend import MlxStepDecoder
    import numpy as np
    reference = json.loads((reference_dir / "full-reference.json").read_text())
    if (reference.get("checkpoint"), reference.get("revision"), reference.get("package_revision")) != (CHECKPOINT, REVISION, PACKAGE_REVISION):
        raise ValueError("Full parity reference must come from the pinned pretrained checkpoint")
    print(json.dumps({"stage": "mlx-full-parity", "pid": os.getpid()}), flush=True)
    started = time.perf_counter()
    decoder = MlxStepDecoder(model_dir, allow_unvalidated=True)
    load_seconds = time.perf_counter() - started
    with wave.open(str(WORKSPACE / reference["fixture"]), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("Full parity fixture must be mono 16kHz PCM16")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    numeric = full_numeric(decoder, samples, reference_dir, reference["generated_token_ids"])
    numeric_passed = all(item["passed"] for item in numeric.values())
    started = time.perf_counter()
    actual = decoder.decode_waveform(samples, "en")
    decoder.mx.synchronize()
    inference_seconds = time.perf_counter() - started
    expected_tokens, actual_tokens = reference["generated_token_ids"], actual["generated_token_ids"]
    matched = sum(left == right for left, right in zip(expected_tokens, actual_tokens))
    report = {"checkpoint": CHECKPOINT, "revision": REVISION, "weights": "pinned-pretrained-fp16",
              "fixture": reference["fixture"], "expected_tokens": len(expected_tokens), "actual_tokens": len(actual_tokens),
              "matching_tokens": matched, "token_parity": actual_tokens == expected_tokens,
              "text_parity": actual["final_text"] == reference["final_text"], "numerical_parity": numeric_passed,
              "actual_text": actual["final_text"], "actual_token_ids": actual_tokens,
              "timing": "warm: 3 teacher-forced speech windows before full fixture decode",
              "load_seconds": load_seconds, "inference_seconds": inference_seconds,
              "audio_seconds": len(samples) / 16000, "rtf": inference_seconds / (len(samples) / 16000),
              "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "mx_peak_memory_bytes": decoder.mx.get_peak_memory(), "cache": decoder.cache_stats()}
    (reference_dir / "full-mlx-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    # Fail closed. No quality claims or enablement on numerical-only validation.
    if not actual["final_text"].strip() or not report["text_parity"] or not report["token_parity"] or not numeric_passed:
        raise AssertionError("MLX FP16 pretrained token/text parity failed; see full-mlx-parity.json")
    descriptor = decoder.descriptor
    descriptor["validation"] = {"status": "fp16-parity-passed", "fixture": reference["fixture"],
                                "token_parity": True, "text_parity": True, "numerical_parity": True}
    (model_dir / "config.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    from audio8_mlx.common import write_validated_marker
    report["marker"] = str(write_validated_marker(model_dir))
    print(json.dumps(report, indent=2))


def main():
    global REFERENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    REFERENCE = workspace_output(args.reference_dir)
    if args.full:
        if args.model_dir is None:
            parser.error("--full requires --model-dir")
        from audio8_mlx.common import explicit_output_directory
        full(explicit_output_directory(args.model_dir), REFERENCE)
        return 0
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Contracts),
                                unittest.defaultTestLoader.loadTestsFromTestCase(NumericalParity)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    (REFERENCE / "small-mlx-parity.json").write_text(json.dumps({"tests": result.testsRun, "passed": result.wasSuccessful(),
        "max_absolute_errors": METRICS, "full_checkpoint_loaded": False, "torch_imported": "torch" in sys.modules}, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
