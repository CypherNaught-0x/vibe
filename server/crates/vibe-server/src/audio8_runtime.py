# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "audio8-asr-infinite @ git+https://github.com/Edge0-AI/Audio8-ASR-Infinite@c8ba8eea829be0339e8d7757f8ca52dac06e1e32",
#   "torch==2.13.0",
#   "transformers==5.13.0",
#   "numpy>=2.1,<3",
#   "safetensors>=0.8,<1",
# ]
# ///
"""Local Audio8 worker. Stdin: JSON line, then sample_count little-endian f32s.

Stdout: NDJSON ready/error, then progress/segment/result/error per request.
Audio never goes to disk or the network. Only checkpoint downloads use HTTP.
File transcription uses independent <=24-second chunks. Live transcription
uses bounded PCM and rolling KV caches via stream_start/chunk/end frames.
Segment times are coarse emission-clock bounds, not word/speech alignment.
Clock: frame_len=8 (160ms), delay=480ms.

Sources verified via HF/GitHub APIs on 2026-09-30:
https://huggingface.co/api/models/Edge0/Audio8-ASR-Infinite
https://github.com/Edge0-AI/Audio8-ASR-Infinite/commit/c8ba8eea829be0339e8d7757f8ca52dac06e1e32
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

CHECKPOINT = "Edge0/Audio8-ASR-Infinite"
REVISION = "7476824bc222e4ad509d286e8cae8b8d3f371129"
PACKAGE_REVISION = "c8ba8eea829be0339e8d7757f8ca52dac06e1e32"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 24 * SAMPLE_RATE
HEADER_LIMIT = 65536
ATTENTION_IMPLEMENTATION = "sdpa"  # PyTorch's fused, platform-dispatched attention kernels.


class ProtocolError(Exception):
    pass


class AudioConfig:
    sampling_rate = SAMPLE_RATE
    raw_audio_samples_per_token = 2560  # 160ms = eight 20ms tower frames
    streaming_n_left_pad_tokens = 9


def validate_manifest(path: str) -> None:
    marker = Path(path)
    if not str(marker).endswith(".audio8.json"):
        raise ValueError("Audio8 model marker must end in .audio8.json")
    with marker.open("rb") as handle:
        data = handle.read(HEADER_LIMIT + 1)
    if len(data) > HEADER_LIMIT:
        raise ValueError("Audio8 model marker is too large")
    manifest = json.loads(data)
    if isinstance(manifest, dict) and manifest.get("backend") == "torch":
        manifest.pop("backend")
    if manifest != {"engine": "audio8", "checkpoint": CHECKPOINT, "revision": REVISION}:
        raise ValueError("Audio8 marker must select the supported pinned checkpoint revision")


def verify_package() -> None:
    """Installed-environment overrides must also use the reviewed upstream source."""
    distribution = importlib.metadata.distribution("audio8-asr-infinite")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    url = direct_url.get("url", "").removesuffix(".git")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if url != "https://github.com/Edge0-AI/Audio8-ASR-Infinite" or commit != PACKAGE_REVISION:
        raise RuntimeError("Install audio8-asr-infinite from the pinned Git commit; unverified installed source is refused")


def select_device(torch, no_gpu: bool, gpu_device: int):
    if not no_gpu and torch.cuda.is_available():
        index = max(0, gpu_device)
        if index >= torch.cuda.device_count():
            raise ValueError("Audio8 CUDA device index is unavailable")
        device = torch.device(f"cuda:{index}")
        with torch.cuda.device(device):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return device, dtype
    mps = getattr(torch.backends, "mps", None)
    if not no_gpu and sys.platform == "darwin" and mps and mps.is_available():
        return torch.device("mps"), torch.float16
    # CPU fp32 avoids unsupported half/bfloat16 kernels; it needs ~16GB for weights.
    return torch.device("cpu"), torch.float32


class CpuFrontend:
    """Use the canonical CPU extractor on MPS (complex STFT is unsupported).

    Deliberately do not expose n_fft/mel_filters: upstream's optimized frontend
    would otherwise run torch.stft on MPS. Audio/text tower inference stays on MPS.
    """

    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, windows, **kwargs):
        if hasattr(windows, "detach"):
            windows = windows.detach().cpu().numpy()
        return self.extractor(windows, **kwargs)


def transcription_config(config):
    """The pinned public ASR weights contain no semantic_vad_heads parameters.

    Their config advertises training-only VAD horizons. Construct no such heads:
    the exact upstream weight-key validation still checks every ASR parameter.
    The simulated-streaming ASR decoder never consumes semantic VAD logits.
    """
    config.semantic_vad_horizons_seconds = None
    return config


class Runtime:
    def __init__(self, args):
        verify_package()
        import numpy as np
        import torch
        from transformers import AutoFeatureExtractor, AutoTokenizer
        from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
            Audio8ASRInfiniteForConditionalGeneration,
            resolve_qwen_language_token_id,
            resolve_qwen_streaming_special_token_ids,
        )
        from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch
        from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig

        self.np = np
        self.torch = torch
        self.device, self.dtype = select_device(torch, args.no_gpu, args.gpu_device)
        # Rust sets a local default HF_HOME, but explicit HF_HOME/HF_HUB_CACHE wins.
        home = os.environ.get("HF_HOME", str(Path(args.manifest).parent / ".audio8-runtime" / "hf"))
        cache = os.environ.get("HF_HUB_CACHE", str(Path(home) / "hub"))
        common = dict(revision=REVISION, cache_dir=cache, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, **common)
        extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT, **common)
        self.extractor = CpuFrontend(extractor) if self.device.type == "mps" else extractor
        # Import the pinned package class explicitly: never execute HF auto_map code.
        config = transcription_config(Audio8ASRInfiniteConfig.from_pretrained(CHECKPOINT, **common))
        self.model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
            CHECKPOINT,
            **common,
            config=config,
            torch_dtype=self.dtype,
            use_safetensors=True,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        ).to(self.device).eval()
        self.model.config.use_cache = True
        self.model.gradient_checkpointing_disable()
        self.special_ids = resolve_qwen_streaming_special_token_ids(self.tokenizer)
        self.language_ids = {language: resolve_qwen_language_token_id(self.tokenizer, language) for language in ("en", "zh")}
        self.decode_batch = simulated_streaming_greedy_decode_batch

    def set_threads(self, threads: int):
        if threads > 0:
            self.torch.set_num_threads(threads)

    def decode(self, raw: bytes, language: str) -> str:
        waveform = self.np.frombuffer(raw, dtype="<f4").astype(self.np.float32, copy=False)
        if not self.np.isfinite(waveform).all():
            raise ValueError("Audio8 samples must be finite")
        waveform = self.np.clip(waveform, -1.0, 1.0)
        results = self.decode_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.extractor,
            waveforms=[waveform],
            language_token_ids=[self.language_ids[language]],
            special_ids=self.special_ids,
            audio_config=AudioConfig(),
            num_delay_tokens=[3],
            frame_len=8,
            right_pad_text_tokens=10,
            dtype=self.dtype,
            device=self.device,
            # Do NOT use 512: upstream extends right silence to reach this limit.
            # None follows the complete audio clock + delay + right padding,
            # suppresses EOS, and drains the tail without inflating the context.
            max_new_tokens=None,
        )
        return results[0]["final_text"]


def read_exact(stream, count: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        part = stream.read(min(count - len(result), 1048576))
        if not part:
            raise ProtocolError("Truncated Audio8 binary sample frame")
        result.extend(part)
    return bytes(result)


def read_request(stream):
    line = stream.readline(HEADER_LIMIT + 1)
    if not line:
        return None
    if len(line) > HEADER_LIMIT or not line.endswith(b"\n"):
        raise ProtocolError("Invalid Audio8 request header frame")
    try:
        request = json.loads(line)
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("Invalid Audio8 request JSON") from exc
    if not isinstance(request, dict):
        raise ProtocolError("Expected Audio8 request object")
    kind = request.get("type")
    if kind not in ("transcribe", "stream_start", "stream_chunk", "stream_end"):
        raise ProtocolError("Expected Audio8 transcription or live request")
    count = request.get("sample_count", 0 if kind in ("stream_start", "stream_end") else None)
    if type(count) is not int or count < 0:
        raise ProtocolError("Invalid Audio8 sample count")
    if kind == "stream_chunk" and not 0 < count <= 2560:
        raise ProtocolError("Audio8 live frames must contain 1..2560 samples")
    if kind in ("stream_start", "stream_end") and count != 0:
        raise ProtocolError("Audio8 control frames cannot contain PCM")
    if kind in ("transcribe", "stream_start") and request.get("language") not in ("en", "zh"):
        raise ProtocolError("Audio8 requires language en or zh; auto detection is unsupported")
    threads = request.get("threads", 0)
    if type(threads) is not int or threads < 0:
        raise ProtocolError("Invalid Audio8 thread count")
    # Complete input is drained before decoding: parent can write without an
    # inference/output-pipe deadlock. The request buffer is RAM only; model/KV
    # allocations remain bounded by CHUNK_SAMPLES regardless of input duration.
    return request, read_exact(stream, count * 4)


def transcribe(runtime, request, raw: bytes, emit):
    total = request["sample_count"]
    runtime.set_threads(request.get("threads", 0))
    emit({"type": "progress", "value": 0})
    segment_count = 0
    for start in range(0, total, CHUNK_SAMPLES):
        end = min(start + CHUNK_SAMPLES, total)
        text = runtime.decode(memoryview(raw)[start * 4:end * 4], request["language"])
        if text.strip():
            # Leading separator is required by TranscribeResult::text(), which
            # concatenates segments. Preserve Chinese joins without adding spaces.
            if segment_count and request["language"] == "en" and not text[0].isspace():
                text = " " + text
            emit({"type": "segment", "start": start // 160, "end": (end + 159) // 160, "text": text})
            segment_count += 1
        emit({"type": "progress", "value": end * 100 // max(1, total)})
    if total == 0:
        emit({"type": "progress", "value": 100})
    emit({"type": "result", "segments": segment_count})


def local_module(name):
    """Load only the server-embedded sibling; -I deliberately ignores sys.path."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def serve(runtime, stream, emit):
    live = None
    while True:
        try:
            frame = read_request(stream)
        except ProtocolError as exc:
            emit({"type": "error", "message": str(exc)})
            return 1  # Cannot resynchronize a malformed binary frame.
        if frame is None:
            return 0
        request, raw = frame
        try:
            if request["type"] == "transcribe":
                if live is not None and live.decoder is not None:
                    raise ValueError("Cannot transcribe a file during a live session")
                transcribe(runtime, request, raw, emit)
            else:
                if live is None:
                    protocol = local_module("audio8_stream_protocol")
                    live = protocol.LiveProtocol(runtime, local_module("audio8_streaming"), emit)
                if not live.dispatch(request, raw):
                    return 1
        except Exception as exc:
            # Library exceptions may include tokens/audio: never echo their values.
            message = f"Audio8 inference failed ({type(exc).__name__}); check device memory and runtime dependencies"
            if isinstance(exc, BufferError):
                message = "Audio8 live PCM overflow; stop and restart. No untranscribed audio was silently dropped."
            emit({"type": "error", "message": message})
            return 1


def reserve_protocol_stdout():
    # Preserve a dedicated protocol FD; silence both Python and native library
    # prints, including stderr. They may contain transcripts. uv's installation
    # diagnostics occur before Python starts and remain available to the parent.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    os.close(sink)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
    return protocol


def main():
    protocol = reserve_protocol_stdout()

    def emit(event):
        protocol.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        protocol.flush()

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--gpu-device", type=int, default=-1)
    try:
        args = parser.parse_args()
        validate_manifest(args.manifest)
        runtime = Runtime(args)
    except Exception as exc:
        # Hub failures may include signed URLs or credentials: do not echo values.
        emit({"type": "error", "message": f"Audio8 startup failed ({type(exc).__name__}). Check checkpoint access/HF login, device memory, uv/Python 3.12 and pinned dependencies, or set VIBE_AUDIO8_PYTHON to a prepared interpreter"})
        return 1
    emit({"type": "ready", "device": str(runtime.device), "dtype": str(runtime.dtype),
          "chunk_seconds": 24, "streaming": True, "live_benchmark_required": True, "timestamps": False})
    return serve(runtime, sys.stdin.buffer, emit)


if __name__ == "__main__":
    raise SystemExit(main())
