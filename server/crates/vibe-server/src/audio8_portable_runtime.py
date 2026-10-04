"""Entry point for built-in validated converted backends (no Torch imports)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import audio8_runtime as protocol


def converted_directory(path, backend):
    data = Path(path).read_bytes()
    if len(data) > 4096:
        raise ValueError("Audio8 marker is too large")
    marker = json.loads(data)
    expected = {"engine", "checkpoint", "revision", "backend", "artifacts"}
    if set(marker) != expected or marker.get("backend") != backend:
        raise ValueError("Unsupported converted Audio8 marker")
    if marker["checkpoint"] != protocol.CHECKPOINT or marker["revision"] != protocol.REVISION or marker["engine"] != "audio8":
        raise ValueError("Converted Audio8 requires the pinned checkpoint")
    relative = Path(marker["artifacts"])
    parent = Path(path).resolve().parent
    directory = (parent / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not directory.is_dir() or not directory.is_relative_to(parent):
        raise ValueError("Audio8 artifacts must be a relative directory beside the marker")
    return directory


def main(backend):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--gpu-device", type=int, default=0)
    args = parser.parse_args()
    protocol_out, protocol_in = protocol.reserve_protocol_stdout(), sys.stdin.buffer
    def emit(event):
        protocol_out.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        protocol_out.flush()
    try:
        directory = converted_directory(args.manifest, backend)
        if backend == "mlx" and (args.no_gpu or args.gpu_device > 0):
            emit({"type": "error", "message":
                "Audio8 MLX requires the default Apple Metal device. Disable Transcribe on CPU or choose the Torch/ONNX backend."})
            return 1
        # The protocol owns a separate FD; native/Python stdout and stderr are
        # already redirected to /dev/null by reserve_protocol_stdout().
        if backend == "onnx":
            from audio8_onnx.backend import OnnxRuntime
            runtime = OnnxRuntime(directory / "audio8_onnx.json", no_gpu=args.no_gpu,
                gpu_device=max(0, args.gpu_device))
        elif backend == "mlx":
            from audio8_mlx.backend import MlxRuntime
            runtime = MlxRuntime(directory, no_gpu=args.no_gpu, gpu_device=max(0, args.gpu_device))
        else:
            raise ValueError("Unsupported Audio8 backend")
        emit({"type": "ready", "checkpoint": protocol.CHECKPOINT,
            "revision": protocol.REVISION, "package_revision": protocol.PACKAGE_REVISION,
            "source_git_verified": True, "backend": backend,
            "device": str(runtime.device), "dtype": str(runtime.dtype)})
    except Exception as exc:
        # Suppress raw exceptions/paths, which may include private data or auth.
        emit({"type": "error", "message":
            "Audio8 converted backend startup failed (" + type(exc).__name__ +
            "). Check platform, validated conversion files, package versions and device memory."})
        return 1
    return protocol.serve(runtime, protocol_in, emit)
