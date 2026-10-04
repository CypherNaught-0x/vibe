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
"""EXCLUSIVE HEAVY SLOT: real shared calibration + prerecorded PCM scheduler.

One full model load; every candidate/session shares its immutable weights.
No microphone, no overlap with Torch/reference/conversion or Rust bridge jobs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio8_mlx.common import WORKSPACE, workspace_output
from audio8_mlx.backend import MlxRuntime
from audio8_streaming import calibrate, StreamingDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=WORKSPACE / "plans/audio8_mlx/artifacts/fp16")
    parser.add_argument("--report-dir", type=Path, default=WORKSPACE / "plans/audio8_mlx/artifacts/parity")
    args = parser.parse_args()
    output = workspace_output(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "mlx-runtime-calibration", "pid": os.getpid()}), flush=True)
    runtime = MlxRuntime(args.model_dir, threads=4)
    started = time.perf_counter()
    calibration = calibrate(runtime)
    calibration["total_seconds"] = time.perf_counter() - started
    (output / "full-calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps({"stage": "calibration-complete", **calibration}), flush=True)
    with wave.open(str(WORKSPACE / "server/fixtures/short.wav"), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("Scheduler fixture must be mono PCM16/16kHz")
        samples = runtime.np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(runtime.np.float32) / 32768
    reference = json.loads((output / "full-reference.json").read_text())
    tokens = []
    factory = runtime.new_step_decoder

    def recording_factory(window):
        backend = factory(window)
        original = backend.step
        def record(audio_window, ids):
            token = original(audio_window, ids)
            tokens.append(token)
            return token
        backend.step = record
        return backend

    runtime.new_step_decoder = recording_factory
    decoder = StreamingDecoder(runtime, "en", calibration["window_tokens"])
    events = []
    started = time.perf_counter()
    # Deliberately not 160ms-aligned; the final partial clock frame must survive.
    for offset in range(0, len(samples), 1024):
        events.extend(decoder.push_pcm(samples[offset:offset + 1024].astype("<f4").tobytes()))
    before_finish = len(tokens)
    events.extend(decoder.finish())
    runtime.synchronize()
    elapsed = time.perf_counter() - started
    text = "".join(event["text"] for event in events if event["type"] == "segment")
    report = {"fixture": "server/fixtures/short.wav", "weights": "pinned-pretrained-fp16",
              "physical_capture_tested": False, "calibration_realtime": calibration["realtime"],
              "window_tokens": calibration["window_tokens"], "samples": len(samples),
              "partial_clock_samples": len(samples) % 2560, "tokens_before_finish": before_finish,
              "tokens_after_finish": len(tokens), "generated_token_ids": tokens,
              "token_parity": tokens == reference["generated_token_ids"], "text_parity": text == reference["final_text"],
              "final_text": text, "inference_seconds": elapsed, "events": events,
              "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "mx_peak_memory_bytes": runtime.mx.get_peak_memory(), "torch_imported": "torch" in sys.modules}
    (output / "full-scheduler-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["token_parity"] or not report["text_parity"] or report["torch_imported"]:
        raise AssertionError("Actual shared-scheduler fixture parity failed")
    print(json.dumps({"stage": "scheduler-parity-passed", "tokens": len(tokens), "final_text": text,
                      "inference_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
