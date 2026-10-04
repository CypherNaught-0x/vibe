# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.4.4", "onnxruntime==1.23.2", "tokenizers==0.22.2"]
# ///
"""Torch-free Audio8 ONNX CPU runtime, including the streaming ASR controller.

Integration API: Audio8OnnxBackend(descriptor_path, threads=4), then
transcribe_chunk(float32_mono_16khz, "en"|"zh") -> str. The caller retains
existing segmentation/resampling/progress/cancellation policy. This class does
not require transformers, torch, the upstream package, vLLM or Triton.

The three graph APIs encode_window/decoder_step expose explicit NumPy KV state
for numerical fixtures and alternative controllers. Host windowing follows the
reviewed upstream simulated_streaming_greedy_decode, with EOS suppression and
160ms gears (frame_len=8, delay=3, 52.5ms back/2.5ms look-ahead). Audio cache is
bounded to sliding_window-1; text cache is bounded by the caller's chunk size.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

CHECKPOINT = "Edge0/Audio8-ASR-Infinite"
REVISION = "7476824bc222e4ad509d286e8cae8b8d3f371129"
UPSTREAM = "c8ba8eea829be0339e8d7757f8ca52dac06e1e32"


@dataclass
class KVState:
    values: dict[str, np.ndarray]
    seen: int = 0


class TokenizerAdapter:
    """Small HF-compatible surface used by the shared ByteLevel scheduler."""
    def __init__(self, tokenizer):
        self.backend_tokenizer = tokenizer
        self.all_special_ids = [v["id"] for v in json.loads(tokenizer.to_str()).get("added_tokens", [])
                                if v.get("special")]

    def token_to_id(self, token):
        return self.backend_tokenizer.token_to_id(token)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, (int, np.integer)):
            return self.backend_tokenizer.id_to_token(int(ids))
        return [self.backend_tokenizer.id_to_token(int(v)) for v in ids]

    def decode(self, ids, skip_special_tokens=True, **_kwargs):
        return self.backend_tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


def features(waveform, mel_filters):
    """NumPy equivalent of VoxtralRealtime's default centered Torch STFT."""
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1 or x.size <= 200 or not np.isfinite(x).all():
        raise ValueError("Feature extraction needs finite mono audio longer than 200 samples")
    x = np.pad(x, (200, 200), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160][:-1]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    stft = np.fft.rfft(frames * window, axis=-1)
    power = (stft.real ** 2 + stft.imag ** 2).astype(np.float32)
    mel = np.asarray(mel_filters, dtype=np.float32).T @ power.T
    log = np.log10(np.maximum(mel, np.float32(1e-10)))
    return ((np.maximum(log, np.float32(-6.5)) + 4) / 4)[None].astype(np.float32)


def causal_mask(seen, past, count, window=None, batch=1, prefix=0):
    queries = np.arange(seen, seen + count)[:, None]
    head = min(prefix, past)
    tail = past - head
    # Prefix entries retain their original RoPE positions after physical crop.
    keys = np.concatenate((np.arange(head), np.arange(seen - tail, seen + count)))[None, :]
    allowed = keys <= queries
    if window is not None:
        allowed &= keys > queries - window
    mask = np.where(allowed, np.float32(0), np.finfo(np.float32).min)
    return np.broadcast_to(mask[None, None], (batch, 1, count, past + count)).copy()


def empty_state(cfg, batch=1):
    heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    return KVState({f"past_{kind}_{i}": np.empty((batch, heads, 0, dim), np.float32)
                    for i in range(cfg["num_hidden_layers"]) for kind in ("key", "value")})


def crop_state(state, max_length, prefix=0):
    if max_length < 0 or not 0 <= prefix <= max_length:
        raise ValueError("Invalid rolling KV bound")
    if state.values["past_key_0"].shape[2] <= max_length:
        return state
    tail = max_length - prefix
    cache = {}
    for name, value in state.values.items():
        if prefix and tail:
            cache[name] = np.concatenate((value[:, :, :prefix], value[:, :, -tail:]), axis=2)
        elif prefix:
            cache[name] = value[:, :, :prefix].copy()
        else:
            cache[name] = value[:, :, -tail:].copy() if tail else value[:, :, :0].copy()
    return KVState(cache, state.seen)


def run_cached(session, cfg, inputs, state, count, sliding=False, cache_limit=None, prefix=0):
    batch = next(iter(inputs.values())).shape[0]
    state = state if state is not None else empty_state(cfg, batch)
    past = state.values["past_key_0"].shape[2]
    window = cfg.get("sliding_window") if sliding else None
    inputs = dict(inputs)
    inputs.update(state.values)
    inputs["position_ids"] = np.broadcast_to(
        np.arange(state.seen, state.seen + count, dtype=np.int64), (batch, count)).copy()
    inputs["attention_mask"] = causal_mask(state.seen, past, count, window, batch, prefix)
    outputs = dict(zip((v.name for v in session.get_outputs()), session.run(None, inputs)))
    cache = {name.replace("present_", "past_", 1): outputs.pop(name)
             for name in tuple(outputs) if name.startswith("present_")}
    state = KVState(cache, state.seen + count)
    limit = cache_limit if cache_limit is not None else window - 1 if window else None
    if limit is not None:
        state = crop_state(state, limit, prefix)
    return outputs, state


def session_options(threads):
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Otherwise ORT may allocate another full copy of fp32 weights for prepack.
    opts.add_session_config_entry("session.disable_prepacking", "1")
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return opts


def _varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("Truncated ONNX metadata varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("Oversized ONNX metadata varint")


def _proto_fields(data):
    """Read protobuf wire fields only, not tensors; no protobuf/onnx dependency.

    Field numbers below are ONNX ModelProto/GraphProto/TensorProto's stable
    schema, checked against installed onnx 1.22.0. Unknown scalar/message fields
    are skipped; deprecated group wire types fail closed.
    """
    data, offset = memoryview(data), 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("Invalid ONNX metadata field number")
        if wire == 0:
            value, offset = _varint(data, offset)
        else:
            if wire == 2:
                size, offset = _varint(data, offset)
            elif wire in (1, 5):
                size = 8 if wire == 1 else 4
            else:
                raise ValueError("Unsupported ONNX metadata wire type")
            if size > len(data) - offset:
                raise ValueError("Truncated ONNX metadata field")
            value, offset = data[offset:offset + size], offset + size
        yield number, wire, value


def external_initializers(model_bytes):
    """Exact external FP32 tensor metadata from hash-validated graph bytes."""
    if len(model_bytes) > 16 * 1024 * 1024:
        raise ValueError("Audio8 ONNX graph must contain metadata, not embedded large weights")
    graphs = [v for n, w, v in _proto_fields(model_bytes) if n == 7 and w == 2]
    if len(graphs) != 1:
        raise ValueError("ONNX metadata requires exactly one graph")
    result = {}
    for field, wire, message in _proto_fields(graphs[0]):
        if field != 5 or wire != 2:  # GraphProto.initializer
            continue
        fields = list(_proto_fields(message))
        def one(number, expected_wire):
            values = [v for n, w, v in fields if n == number and w == expected_wire]
            if len(values) != 1:
                raise ValueError(f"Invalid ONNX tensor metadata field {number}")
            return values[0]
        location = [v for n, w, v in fields if n == 14 and w == 0]
        if location != [1]:  # internal small constants have DEFAULT location
            if location and location != [0]:
                raise ValueError("Unsupported ONNX tensor data location")
            continue
        if one(2, 0) != 1:  # TensorProto.FLOAT
            raise ValueError("Audio8 external initializers must be FP32")
        name = bytes(one(8, 2)).decode("utf-8")
        shape = []
        for n, w, value in fields:
            if n == 1:
                if w == 0:
                    shape.append(value)
                elif w == 2:
                    offset = 0
                    while offset < len(value):
                        dim, offset = _varint(value, offset)
                        shape.append(dim)
                else:
                    raise ValueError("Invalid ONNX tensor dimensions")
        metadata = {}
        for n, w, entry in fields:
            if n == 13 and w == 2:
                pair = list(_proto_fields(entry))
                keys = [bytes(v).decode("utf-8") for f, t, v in pair if f == 1 and t == 2]
                values = [bytes(v).decode("utf-8") for f, t, v in pair if f == 2 and t == 2]
                if len(keys) != 1 or len(values) != 1 or keys[0] in metadata:
                    raise ValueError("Duplicate or invalid ONNX external-data entry")
                metadata[keys[0]] = values[0]
        if (not name or name in result or not shape or any(v <= 0 for v in shape)
                or set(metadata) != {"location", "offset", "length"}
                or metadata["location"] != "weights.bin"
                or not metadata["offset"].isdigit() or not metadata["length"].isdigit()):
            raise ValueError("Invalid Audio8 external initializer metadata")
        result[name] = (tuple(shape), int(metadata["offset"]), int(metadata["length"]))
    return result


class MappedInitializers:
    """Read-only shared backing storage; CPU OrtValues and views stay alive.

    ORT 1.23.2 add_external_initializers COPIES; add_initializer does not. Each
    session also retains this owner, protecting callers that keep a session
    after releasing the backend. No session/thread-count reload copies weights.
    """
    def __init__(self, path):
        self.path, self.size = Path(path), Path(path).stat().st_size
        self.mapping = None
        self.views, self.values, self.metadata = {}, {}, {}

    def add(self, options, graph_bytes):
        import math
        import sys
        if sys.byteorder != "little":
            raise ValueError("Zero-copy ONNX FP32 weights require a little-endian CPU")
        for name, (shape, offset, length) in external_initializers(graph_bytes).items():
            if offset % 4 or length != math.prod(shape) * 4 or offset + length > self.size:
                raise ValueError(f"Out-of-bounds or incompatible external initializer: {name}")
            signature = (shape, offset, length)
            if name in self.metadata and self.metadata[name] != signature:
                raise ValueError(f"Graphs disagree on shared initializer metadata: {name}")
            if name not in self.values:
                if self.mapping is None:
                    self.mapping = np.memmap(self.path, dtype=np.uint8, mode="r")
                view = np.ndarray(shape, dtype="<f4", buffer=self.mapping, offset=offset)
                value = ort.OrtValue.ortvalue_from_numpy(view, "cpu")
                if view.flags.writeable or value.data_ptr() != view.ctypes.data:
                    raise ValueError("ORT CPU initializer must borrow read-only mapped data")
                self.views[name], self.values[name], self.metadata[name] = view, value, signature
            options.add_initializer(name, self.values[name])

    def session(self, graph_bytes, threads):
        options = session_options(threads)
        self.add(options, graph_bytes)
        # All external tensors are supplied, so ORT need not resolve/read the
        # weights file itself. Passing checked bytes also avoids graph TOCTOU.
        session = ort.InferenceSession(graph_bytes, options, providers=["CPUExecutionProvider"])
        session._audio8_initializer_owner = self
        return session


class Audio8OnnxBackend:
    def __init__(self, descriptor_path, threads=4, *, allow_unvalidated=False):
        path = Path(descriptor_path).resolve()
        self.root = path.parent
        self.descriptor = json.loads(path.read_text(encoding="utf-8"))
        d = self.descriptor
        if (d.get("format_version") != 1 or d.get("backend") != "onnx" or d.get("engine") != "audio8"
                or d.get("checkpoint") != CHECKPOINT or d.get("revision") != REVISION
                or d.get("upstream_revision") != UPSTREAM or d.get("package_revision") != UPSTREAM
                or d.get("source_git_verified") is not True
                or d.get("host", {}).get("frame_len") != 8 or d.get("dtype") != "float32"):
            raise ValueError("Not a supported pinned Audio8 ONNX descriptor")
        if not allow_unvalidated and d.get("validation", {}).get("pretrained_numerical_parity") is not True:
            raise ValueError("Audio8 ONNX artifacts have not passed pretrained numerical/token parity")
        self.np, self.device, self.dtype = np, "onnx:cpu", "float32"
        self.backend, self.checkpoint, self.revision = "onnx", CHECKPOINT, REVISION
        self.backend_version = ort.__version__
        self.package_revision, self.source_git_verified = UPSTREAM, True
        for name, digest in d["assets_sha256"].items():
            if Path(name).name != name:
                raise ValueError("Descriptor assets must stay inside the model directory")
            if hashlib.sha256((self.root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"ONNX model asset SHA256 mismatch: {name}")
        w = d["weights"]
        if w["file"] != "weights.bin" or (self.root / w["file"]).stat().st_size != w["size"]:
            raise ValueError("Missing or truncated ONNX external weights")
        self.audio_config, self.text_config = d["audio_config"], d["text_config"]
        self.sessions = {}
        self.threads = threads
        self._mapped_initializers = MappedInitializers(self.root / w["file"])
        for name in ("embedder", "encoder", "decoder"):
            graph = d["graphs"][name]
            if graph["file"] != name + ".onnx":
                raise ValueError("Invalid graph filename")
            graph_bytes = (self.root / graph["file"]).read_bytes()
            if hashlib.sha256(graph_bytes).hexdigest() != graph["sha256"]:
                raise ValueError(f"ONNX graph SHA256 mismatch: {name}")
            self.sessions[name] = self._mapped_initializers.session(graph_bytes, threads)
        from tokenizers import Tokenizer
        self.tokenizer = TokenizerAdapter(Tokenizer.from_file(str(self.root / "tokenizer.json")))
        self.mel_filters = np.load(self.root / "mel_filters.npy", allow_pickle=False)
        strings = {"bos": "<|im_start|>", "eos": "<|im_end|>", "pad": "<|endoftext|>",
                   "streaming_pad": "[STREAMING_PAD]", "streaming_word": "[STREAMING_WORD]",
                   "en": "[LANGUAGE_EN]", "zh": "[LANGUAGE_ZH]"}
        self.ids = {name: self.tokenizer.token_to_id(token) for name, token in strings.items()}
        if any(value is None for value in self.ids.values()):
            raise ValueError("Tokenizer lacks required streaming/language special tokens")
        self.special_ids = {k + "_token_id": self.ids[k] for k in
                            ("bos", "eos", "pad", "streaming_pad", "streaming_word")}
        self.language_ids = {k: self.ids[k] for k in ("en", "zh")}
        if d.get("special_ids") != self.special_ids or d.get("language_ids") != self.language_ids:
            raise ValueError("Descriptor tokenizer IDs disagree with tokenizer assets")

    def new_step_decoder(self, window_tokens):
        return NumpyStepDecoder(self, window_tokens)

    def encode_window(self, waveform, num_tokens, state=None, *, cache_limit=None):
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer")
        mel = features(waveform, self.mel_filters)
        conv = self.sessions["embedder"].run(["conv_embeds"], {"input_features": mel})[0]
        count = num_tokens * 8
        if conv.shape[1] < count:
            raise ValueError("Audio window is shorter than its declared token count")
        conv = np.ascontiguousarray(conv[:, -count:])
        output, state = run_cached(self.sessions["encoder"], self.audio_config,
                                   {"conv_embeds": conv}, state, count, sliding=True, cache_limit=cache_limit)
        return output["audio_embeds"], state

    def decoder_step(self, input_ids, audio_embeds, state=None, num_delay_tokens=3, *, cache_limit=None, prefix=0):
        ids = np.asarray(input_ids, dtype=np.int64)
        audio = np.asarray(audio_embeds, dtype=np.float32)
        if ids.ndim != 2 or audio.shape[:2] != ids.shape:
            raise ValueError("Token IDs and audio embeddings must have aligned batch/token dimensions")
        output, state = run_cached(self.sessions["decoder"], self.text_config,
                                   {"input_ids": ids, "audio_embeds": audio,
                                    "num_delay_tokens": np.full(ids.shape[0], num_delay_tokens, np.float32)},
                                   state, ids.shape[1], cache_limit=cache_limit, prefix=prefix)
        return output["logits"], state

    def decode_waveform(self, waveform, language="en"):
        """Fixture convenience: canonical ceil-padded batch=1 clock and tokens."""
        if language not in ("en", "zh"):
            raise ValueError("Audio8 supports only English ('en') and Chinese ('zh')")
        wave = np.asarray(waveform, dtype=np.float32)
        if wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError("Audio must be finite mono float32 at 16kHz")
        if wave.size > self.descriptor["host"]["max_chunk_samples"]:
            raise ValueError("Segment audio into chunks of at most 24 seconds before calling ONNX")
        if not wave.size:
            return {"final_text": "", "generated_token_ids": []}
        samples_per_token, prefill = 2560, 13  # 9 left-pad + 3 delay + 1
        real_tokens = max(1, (wave.size + samples_per_token - 1) // samples_per_token)
        stream = np.zeros((9 + real_tokens + 14) * samples_per_token, np.float32)
        stream[9 * samples_per_token:9 * samples_per_token + wave.size] = wave
        generated = []
        ids = [self.ids["bos"], self.ids[language]] + [self.ids["streaming_pad"]] * (prefill - 2)
        audio_state, text_state = None, None
        start, end = 0, prefill * samples_per_token
        while end + 40 <= len(stream):
            count = (end - start) // samples_per_token
            audio, audio_state = self.encode_window(stream[max(start - 840, 0):end + 40], count, audio_state)
            logits, text_state = self.decoder_step(np.array([ids], np.int64), audio, text_state)
            scores = logits[0, -1].copy()
            scores[self.ids["eos"]] = -np.inf
            token = int(scores.argmax())
            generated.append(token)
            ids = [token]
            start, end = end, end + samples_per_token
        hidden = {self.ids[k] for k in ("bos", "eos", "pad", "streaming_pad", "streaming_word")}
        text = self.tokenizer.decode([t for t in generated if t not in hidden], skip_special_tokens=True)
        return {"final_text": text, "generated_token_ids": generated}

    def transcribe_chunk(self, waveform, language="en"):
        return self.decode_waveform(waveform, language)["final_text"]

    def decode(self, raw: bytes, language: str) -> str:
        """Existing server runtime interface: little-endian float32 PCM bytes."""
        wave = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(wave).all():
            raise ValueError("Audio8 samples must be finite")
        return self.transcribe_chunk(np.clip(wave, -1.0, 1.0), language)

    def set_threads(self, threads: int):
        """ORT thread pools are immutable; changing this reloads graphs serially.

        The server should pass the configured thread count at construction.
        Repeated requests with the same setting are cheap. Never hold old and
        new full-weight sessions simultaneously while changing thread count.
        """
        if type(threads) is not int or threads < 0:
            raise ValueError("threads must be a non-negative integer")
        if threads == 0 or threads == self.threads:
            return
        session_options(threads)  # validate before releasing any active session
        for name in ("embedder", "encoder", "decoder"):
            graph = self.descriptor["graphs"][name]
            graph_bytes = (self.root / graph["file"]).read_bytes()
            if hashlib.sha256(graph_bytes).hexdigest() != graph["sha256"]:
                raise ValueError(f"ONNX graph SHA256 mismatch: {name}")
            del self.sessions[name]
            self.sessions[name] = self._mapped_initializers.session(graph_bytes, threads)
        self.threads = threads

    decode_chunk = transcribe_chunk


class NumpyStepDecoder:
    """Per-live-session KV state; model/ORT sessions are shared, never reloaded."""
    def __init__(self, runtime, window_tokens):
        if type(window_tokens) is not int or not 14 <= window_tokens <= 256:
            raise ValueError("window_tokens must be 14..256")
        self.runtime, self.window_tokens = runtime, window_tokens
        sliding = runtime.audio_config.get("sliding_window")
        self.audio_window_frames = min(window_tokens * 8, sliding or window_tokens * 8)
        self.reset()

    def reset(self, *, frame_len=8, num_delay_tokens=3):
        if frame_len != 8 or num_delay_tokens != 3:
            raise ValueError("Audio8 ONNX clock is pinned to frame_len=8, delay=3")
        self.audio_state = self.text_state = None

    @property
    def absolute_audio_position(self):
        return self.audio_state.seen if self.audio_state else 0

    @property
    def absolute_text_position(self):
        return self.text_state.seen if self.text_state else 0

    def step(self, audio_window, ids):
        ids = list(ids)
        vocabulary = self.runtime.text_config["vocab_size"]
        if not ids or any(type(v) is not int or not 0 <= v < vocabulary for v in ids):
            raise ValueError("Audio8 ONNX step requires valid token IDs")
        audio, self.audio_state = self.runtime.encode_window(
            audio_window, len(ids), self.audio_state, cache_limit=self.audio_window_frames)
        logits, self.text_state = self.runtime.decoder_step(
            np.array([ids], np.int64), audio, self.text_state, cache_limit=self.window_tokens, prefix=13)
        scores = logits[0, -1].copy()
        scores[self.runtime.special_ids["eos_token_id"]] = -np.inf
        return int(scores.argmax())

    def cache_stats(self):
        def stats(state):
            if state is None:
                return 0, 0
            return state.values["past_key_0"].shape[2], sum(v.nbytes for v in state.values.values())
        audio_frames, audio_bytes = stats(self.audio_state)
        text_tokens, text_bytes = stats(self.text_state)
        return {"audio_position": self.absolute_audio_position, "text_position": self.absolute_text_position,
                "absolute_audio_position": self.absolute_audio_position, "absolute_text_position": self.absolute_text_position,
                "audio_kv_frames": audio_frames, "text_kv_tokens": text_tokens,
                "audio_cache_frames": audio_frames, "text_cache_tokens": text_tokens,
                "audio_kv_storage_frames": audio_frames, "cache_bytes": audio_bytes + text_bytes,
                "audio_cache_bytes": audio_bytes, "text_cache_bytes": text_bytes}


class OnnxRuntime(Audio8OnnxBackend):
    """Fixed shared-runtime constructor; ONNX's portable provider is CPU-only."""
    def __init__(self, descriptor_path, no_gpu=False, gpu_device=0, threads=4):
        # no_gpu/gpu_device preserve the existing runtime signature. They do not
        # claim GPU support; provider/device in ready events explicitly say CPU.
        if type(threads) is not int or threads < 0:
            raise ValueError("threads must be a non-negative integer")
        super().__init__(descriptor_path, threads=threads if threads > 0 else 4)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True, help="mono float32 16kHz .npy file")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    model = Audio8OnnxBackend(args.model, args.threads)
    print(model.transcribe_chunk(np.load(args.audio, allow_pickle=False), args.language))


if __name__ == "__main__":
    main()
