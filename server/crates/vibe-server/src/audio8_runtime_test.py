# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Dependency-free tests: uv run --script server/crates/vibe-server/src/audio8_runtime_test.py."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RUNTIME_PATH = Path(__file__).with_name("audio8_runtime.py")
spec = importlib.util.spec_from_file_location("audio8_runtime", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def frame(samples, language="en", **fields):
    header = {"type": "transcribe", "sample_count": len(samples), "language": language, **fields}
    return json.dumps(header).encode() + b"\n" + struct.pack(f"<{len(samples)}f", *samples)


class FakeRuntime:
    def __init__(self):
        self.chunks = []
        self.threads = []

    def set_threads(self, threads):
        self.threads.append(threads)

    def decode(self, raw, language):
        self.chunks.append((len(raw) // 4, struct.unpack("<f", raw[-4:])[0], language))
        return "tail" if language == "en" else "中文"


class RuntimeTests(unittest.TestCase):
    def test_public_checkpoint_omits_only_untrained_semantic_vad_heads(self):
        config = SimpleNamespace(semantic_vad_horizons_seconds=[0.5, 1.0, 2.0, 3.0],
                                 weight_format_version=2, hidden_size=2048, semantic_vad_num_classes=8)
        self.assertIs(runtime.transcription_config(config), config)
        self.assertIsNone(config.semantic_vad_horizons_seconds)
        self.assertEqual(config.weight_format_version, 2)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.semantic_vad_num_classes, 8)

    def test_marker_pins_and_no_executable_fields(self):
        valid = {"engine": "audio8", "checkpoint": runtime.CHECKPOINT, "revision": runtime.REVISION}
        for manifest, accepted in [(valid, True), ({**valid, "revision": "main"}, False),
                                   ({**valid, "executable": "malicious"}, False), ({**valid, "checkpoint": "other/model"}, False)]:
            with patch.object(Path, "open", return_value=io.BytesIO(json.dumps(manifest).encode())):
                if accepted:
                    runtime.validate_manifest("test.audio8.json")
                else:
                    with self.assertRaises(ValueError):
                        runtime.validate_manifest("test.audio8.json")
        with self.assertRaises(ValueError):
            runtime.validate_manifest("test.json")

    def test_source_provenance(self):
        valid = {"url": "https://github.com/Edge0-AI/Audio8-ASR-Infinite", "vcs_info": {"commit_id": runtime.PACKAGE_REVISION}}
        distribution = SimpleNamespace(read_text=lambda _: json.dumps(valid))
        with patch.object(runtime.importlib.metadata, "distribution", return_value=distribution):
            runtime.verify_package()
            valid["vcs_info"]["commit_id"] = "main"
            with self.assertRaises(RuntimeError):
                runtime.verify_package()

    def test_binary_frames_preserve_newlines_and_tail(self):
        sample = struct.unpack("<f", b"\n\x00\x00\x3f")[0]
        stream = io.BytesIO(frame([sample, -0.5, 0.25], threads=2))
        request, raw = runtime.read_request(stream)
        self.assertEqual(request["threads"], 2)
        self.assertEqual(struct.unpack("<3f", raw), (sample, -0.5, 0.25))
        self.assertIsNone(runtime.read_request(stream))

    def test_persistent_requests_and_empty_audio(self):
        backend = FakeRuntime()
        events = []
        stream = io.BytesIO(frame([0.5]) + frame([], "zh") + frame([-0.25], "zh"))
        self.assertEqual(runtime.serve(backend, stream, events.append), 0)
        self.assertEqual(len([e for e in events if e["type"] == "result"]), 3)
        self.assertEqual(backend.chunks, [(1, 0.5, "en"), (1, -0.25, "zh")])
        self.assertEqual([e["end"] for e in events if e["type"] == "segment"], [1, 1])

    def test_long_audio_is_bounded_and_partial_last_chunk_is_not_dropped(self):
        count = runtime.CHUNK_SAMPLES * 2 + 7
        raw = b"\x00" * ((count - 1) * 4) + struct.pack("<f", 0.75)
        backend = FakeRuntime()
        events = []
        runtime.transcribe(backend, {"sample_count": count, "language": "en"}, raw, events.append)
        self.assertEqual([c[0] for c in backend.chunks], [runtime.CHUNK_SAMPLES, runtime.CHUNK_SAMPLES, 7])
        self.assertEqual(backend.chunks[-1][1], 0.75)
        segments = [e for e in events if e["type"] == "segment"]
        self.assertEqual([e["start"] for e in segments], [0, 2400, 4800])
        self.assertEqual(segments[-1]["end"], 4801)
        self.assertEqual("".join(e["text"] for e in segments), "tail tail tail")
        progress = [e["value"] for e in events if e["type"] == "progress"]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(progress[-1], 100)
        self.assertEqual(events[-1], {"type": "result", "segments": 3})
        # Include both pad sides and worst-case audio-clock rounding in context.
        self.assertLess(24 + (9 + 3 + 1 + 10 + 1) * 0.16, 30)

    def test_malformed_and_truncated_frames_stop_without_resynchronising(self):
        bad = [b"{bad}\n", b"[]\n", b"x" * (runtime.HEADER_LIMIT + 1),
               frame([0.5])[:-1], frame([], "auto"), frame([], threads=-1),
               b'{"type":"transcribe","sample_count":true,"language":"en"}\n']
        for payload in bad:
            events = []
            self.assertEqual(runtime.serve(FakeRuntime(), io.BytesIO(payload), events.append), 1)
            self.assertEqual(events[-1]["type"], "error")

    def test_inference_errors_do_not_disclose_audio_or_transcripts(self):
        backend = FakeRuntime()
        backend.decode = lambda *_: (_ for _ in ()).throw(ValueError("private transcript"))
        events = []
        self.assertEqual(runtime.serve(backend, io.BytesIO(frame([0.5])), events.append), 1)
        self.assertNotIn("private transcript", json.dumps(events))
        self.assertIn("ValueError", events[-1]["message"])

    def test_device_dtype_and_no_gpu(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                               is_bf16_supported=lambda: True, device=lambda _: unittest.mock.MagicMock())
        torch = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                                device=lambda value: value, bfloat16="bf16", float16="fp16", float32="fp32")
        self.assertEqual(runtime.select_device(torch, True, -1), ("cpu", "fp32"))
        self.assertEqual(runtime.select_device(torch, False, 1), ("cuda:1", "bf16"))
        cuda.is_bf16_supported = lambda: False
        self.assertEqual(runtime.select_device(torch, False, 0), ("cuda:0", "fp16"))
        with self.assertRaises(ValueError):
            runtime.select_device(torch, False, 2)
        cuda.is_available = lambda: False
        with patch.object(runtime.sys, "platform", "darwin"):
            self.assertEqual(runtime.select_device(torch, False, -1), ("mps", "fp16"))
        with patch.object(runtime.sys, "platform", "linux"):
            self.assertEqual(runtime.select_device(torch, False, -1), ("cpu", "fp32"))

    def test_mps_frontend_moves_input_to_cpu_without_exposing_fast_stft(self):
        windows = unittest.mock.MagicMock()
        windows.detach.return_value.cpu.return_value.numpy.return_value = "cpu waveform"
        extractor = unittest.mock.MagicMock(return_value={"input_features": "features"})
        frontend = runtime.CpuFrontend(extractor)
        self.assertFalse(hasattr(frontend, "n_fft"))
        self.assertEqual(frontend(windows, center=True), {"input_features": "features"})
        extractor.assert_called_once_with("cpu waveform", center=True)

    def test_decoder_uses_optimized_clock_and_complete_tail_without_512_limit(self):
        backend = runtime.Runtime.__new__(runtime.Runtime)
        values = unittest.mock.MagicMock()
        values.astype.return_value = values
        backend.np = SimpleNamespace(frombuffer=lambda *_, **__: values, float32="fp32",
                                     isfinite=lambda _: SimpleNamespace(all=lambda: True), clip=lambda *args: values)
        backend.model, backend.tokenizer, backend.extractor = object(), object(), object()
        backend.language_ids, backend.special_ids = {"en": 7}, {}
        backend.dtype, backend.device = "fp32", "cpu"
        backend.decode_batch = unittest.mock.MagicMock(return_value=[{"final_text": "tail"}])
        self.assertEqual(backend.decode(struct.pack("<f", 0.5), "en"), "tail")
        kwargs = backend.decode_batch.call_args.kwargs
        self.assertEqual(kwargs["frame_len"], 8)
        self.assertEqual(kwargs["num_delay_tokens"], [3])
        self.assertEqual(kwargs["right_pad_text_tokens"], 10)
        self.assertIsNone(kwargs["max_new_tokens"])

    def test_native_and_python_prints_cannot_corrupt_protocol(self):
        code = f'''import importlib.util,os,json
s=importlib.util.spec_from_file_location("r",{str(RUNTIME_PATH)!r}); r=importlib.util.module_from_spec(s); s.loader.exec_module(r)
p=r.reserve_protocol_stdout()
print("private transcript"); os.write(1,b"native audio"); os.write(2,b"private diagnostics")
p.write(json.dumps({{"type":"ready"}})+"\\n"); p.flush()
'''
        child = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, check=True)
        self.assertEqual(child.stdout, b'{"type": "ready"}\n')
        self.assertEqual(child.stderr, b"")

    def test_startup_dependency_failure_is_protocol_json(self):
        code = f'''import importlib.util,sys
s=importlib.util.spec_from_file_location("r",{str(RUNTIME_PATH)!r}); r=importlib.util.module_from_spec(s); s.loader.exec_module(r)
r.validate_manifest=lambda _:None
r.Runtime=lambda _: (_ for _ in ()).throw(ModuleNotFoundError("missing pinned package"))
sys.argv=["runtime","--manifest","test.audio8.json"]
sys.exit(r.main())
'''
        child = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True)
        self.assertEqual(child.returncode, 1)
        event = json.loads(child.stdout)
        self.assertEqual(event["type"], "error")
        self.assertIn("ModuleNotFoundError", event["message"])
        self.assertIn("VIBE_AUDIO8_PYTHON", event["message"])
        self.assertEqual(child.stderr, b"")


if __name__ == "__main__":
    unittest.main(verbosity=2)
