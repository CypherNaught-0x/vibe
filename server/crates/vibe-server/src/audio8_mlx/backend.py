"""Torch-free step decoder for the parent's rolling Audio8 audio clock.

Contract (one request, batch=1):
    decoder = MlxStepDecoder(converted_directory)
    decoder.reset(frame_len=8, num_delay_tokens=3)
    token = decoder.step(audio_window, input_token_ids)

Each audio_window is the canonical centered-STFT window: prefill covers
13 tokens; later windows cover one 160ms token plus 52.5ms left / 2.5ms right
context. Caller owns window scheduling, prompt queue, tail drain and events.
step() trims conv overlap, carries encoder/text caches and suppresses EOS.
For rolling LM eviction, pass text_window_tokens and keep_prompt_tokens
explicitly; no context is silently dropped by the canonical default.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .common import CHECKPOINT, REVISION, PACKAGE_REVISION, read_descriptor, verify_dependencies


class TokenizerAdapter:
    """The small HF tokenizer surface needed by the shared byte-level streamer."""
    def __init__(self, tokenizer):
        self.backend_tokenizer = tokenizer
        self.all_special_ids = [item["id"] for item in json.loads(tokenizer.to_str()).get("added_tokens", []) if item.get("special")]

    def token_to_id(self, token):
        return self.backend_tokenizer.token_to_id(token)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.backend_tokenizer.id_to_token(ids)
        return [self.backend_tokenizer.id_to_token(int(token)) for token in ids]

    def decode(self, ids, skip_special_tokens=True, **_kwargs):
        return self.backend_tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


QUANT_BITS = 8
QUANT_GROUP = 64
# A 160ms tick is memory-bandwidth bound: it streams ~8GB of FP16 weights. 8-bit groupwise
# weights halve that (117ms -> 76ms per tick on an M3 Pro) and gave transcripts identical
# to FP16 on every checked clip. 4-bit is faster but garbled non-English text, so it is
# not offered. VIBE_AUDIO8_MLX_BITS=0 selects FP16.


def quantization_bits():
    raw = os.environ.get("VIBE_AUDIO8_MLX_BITS", str(QUANT_BITS))
    if raw not in ("0", "8"):
        raise ValueError("VIBE_AUDIO8_MLX_BITS must be 0 (FP16) or 8")
    return int(raw)


def quantize_weights(model, mx, bits):
    """Weight-only groupwise quantization of the big projections and tied embedding.

    Norm-conditioning (AdaRMSNorm), the frame-length table and convolutions stay FP16.
    Layers whose width is not a multiple of the group size are left untouched.
    """
    import mlx.nn as nn

    def selected(path, module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if "ada_rms_norm" in path or "frame_len_embedding" in path:
            return False
        return module.weight.shape[-1] % QUANT_GROUP == 0
    nn.quantize(model, group_size=QUANT_GROUP, bits=bits, class_predicate=selected)
    mx.eval(model.parameters())


class MlxStepDecoder:
    def __init__(self, directory, *, text_window_tokens=None, keep_prompt_tokens=0, audio_window_frames=None, allow_unvalidated=False):
        verify_dependencies()  # no MLX import before platform gate
        descriptor = read_descriptor(directory)
        validation = descriptor.get("validation", {})
        if not allow_unvalidated and (validation.get("status") != "fp16-parity-passed" or
                                     validation.get("token_parity") is not True or validation.get("text_parity") is not True):
            raise RuntimeError("Audio8 MLX checkpoint has not passed FP16 fixture token/text parity validation")
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        from tokenizers import Tokenizer
        from mlx_audio.stt.models.voxtral_realtime.audio import compute_mel_filters, compute_mel_spectrogram
        from .model import Model
        if not mx.metal.is_available():
            raise RuntimeError("Audio8 MLX requires an available Apple Metal GPU")
        self.mx, self.np = mx, np
        self.directory = Path(directory).resolve()
        self.descriptor = descriptor
        self.model = Model(descriptor["model_config"])
        expected = {name: tuple(value.shape) for name, value in tree_flatten(self.model.parameters())}
        actual = {}
        for name in descriptor["weight_files"]:
            weights = mx.load(str(self.directory / name))
            for key, value in weights.items():
                if key in actual or key not in expected or tuple(value.shape) != expected[key] or value.dtype != mx.float16:
                    raise ValueError(f"Incompatible Audio8 MLX weight: {key}")
                actual[key] = tuple(value.shape)
            self.model.load_weights(list(weights.items()), strict=False)
        if actual != expected:
            raise ValueError("Missing Audio8 MLX model parameters")
        self.model.eval()
        mx.eval(self.model.parameters())
        bits = quantization_bits()
        if bits:
            quantize_weights(self.model, mx, bits)
        self.quantization_bits = bits or 16
        self.tokenizer = TokenizerAdapter(Tokenizer.from_file(str(self.directory / "tokenizer.json")))
        self.special_ids = {}
        for key, token in {"bos_token_id": "<|im_start|>", "eos_token_id": "<|im_end|>",
                           "pad_token_id": "<|endoftext|>", "streaming_pad_token_id": "[STREAMING_PAD]",
                           "streaming_word_token_id": "[STREAMING_WORD]"}.items():
            self.special_ids[key] = self._token_id(token)
        self.language_ids = {language: self._token_id(token) for language, token in
                             (("en", "[LANGUAGE_EN]"), ("zh", "[LANGUAGE_ZH]"))}
        processor = json.loads((self.directory / "preprocessor_config.json").read_text())
        if (processor.get("sampling_rate"), processor.get("feature_size"), processor.get("hop_length"),
            processor.get("n_fft"), processor.get("global_log_mel_max")) != (16000, 128, 160, 400, 1.5):
            raise ValueError("Unsupported Audio8 MLX frontend configuration")
        self.mel_filters = mx.array(compute_mel_filters(), dtype=mx.float32)
        self.compute_mel = compute_mel_spectrogram
        self.text_window_tokens = text_window_tokens
        self.keep_prompt_tokens = keep_prompt_tokens
        self.audio_window_frames = audio_window_frames
        self.device, self.dtype = "mlx:gpu", "float16"
        self.reset()

    @classmethod
    def from_loaded(cls, loaded, *, text_window_tokens=None, keep_prompt_tokens=0, audio_window_frames=None):
        """Fresh request caches, shared immutable weights; never call mx.load here."""
        decoder = cls.__new__(cls)
        for name in ("mx", "np", "directory", "descriptor", "model", "tokenizer", "special_ids", "language_ids",
                     "mel_filters", "compute_mel", "device", "dtype"):
            setattr(decoder, name, getattr(loaded, name))
        decoder.text_window_tokens = text_window_tokens
        decoder.keep_prompt_tokens = keep_prompt_tokens
        decoder.audio_window_frames = audio_window_frames
        decoder.reset()
        return decoder

    def _token_id(self, token):
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"Audio8 tokenizer is missing {token}")
        return value

    def reset(self, *, frame_len=8, num_delay_tokens=3):
        if frame_len != 8 or num_delay_tokens != 3:
            raise ValueError("Production Audio8 MLX clock is pinned to frame_len=8, delay=3 (160ms/480ms)")
        self.frame_len, self.num_delay_tokens = frame_len, num_delay_tokens
        self.audio_cache = self.model.audio_tower.make_cache(self.audio_window_frames)
        self.text_cache = self.model.make_text_cache(self.text_window_tokens, self.keep_prompt_tokens)
        if self.text_window_tokens is not None:
            from mlx_lm.models.cache import RotatingKVCache
            mx = self.mx

            class PrefixCache(RotatingKVCache):
                # Audio8's rolling LM is full-attention over retained prefix +
                # recent KV, not a sliding-mask LM. Synthetic multi-token fills
                # must retain the prompt too; the stock cache masks it out.
                def make_mask(self, length, **_kwargs):
                    if length == 1:
                        return None
                    past = min(self.offset, self.max_size - 1)
                    return mx.arange(past + length)[None, :] <= (mx.arange(length)[:, None] + past)

            self.text_cache = [PrefixCache(max_size=self.text_window_tokens, keep=self.keep_prompt_tokens)
                               for _ in self.model.language_model.model.layers]
        self.scales = self.model.delay_scales(num_delay_tokens, frame_len, self.mx.float16)
        self.mx.eval(self.scales)

    def encode_window(self, audio_window, num_tokens):
        """Return MLX [1,num_tokens,2048] projected audio; retains tower cache."""
        samples = self.np.asarray(audio_window, dtype=self.np.float32)
        if samples.ndim != 1 or samples.size < 400 or not self.np.isfinite(samples).all():
            raise ValueError("Audio8 MLX requires a finite mono 16kHz definition window")
        if type(num_tokens) is not int or num_tokens < 1:
            raise ValueError("Audio8 MLX window must contain a positive integer token count")
        mel = self.compute_mel(self.mx.array(samples), self.mel_filters).astype(self.mx.float16)
        conv = self.model.audio_tower.embedder(mel[None])
        frames = num_tokens * self.frame_len
        if conv.shape[1] < frames:
            raise ValueError("Audio8 MLX audio window is shorter than its text clock")
        conv = conv[:, -frames:, :]
        hidden = self.model.audio_tower(conv, self.audio_cache)
        return self.model.project(hidden, self.frame_len)

    def text_step(self, input_token_ids, audio_embeds):
        """Fuse embeddings and return greedy token; retains absolute text RoPE."""
        ids = list(input_token_ids)
        vocabulary = self.model.config["text_config"]["vocab_size"]
        if not ids or any(type(token) is not int or token < 0 or token >= vocabulary for token in ids):
            raise ValueError("Audio8 MLX step requires valid input token IDs")
        if tuple(audio_embeds.shape[:2]) != (1, len(ids)):
            raise ValueError("Audio8 MLX audio/text clocks do not align")
        if self.text_window_tokens is None and self.text_cache[0].offset + len(ids) > 32768:
            raise ValueError("Audio8 MLX full-attention context exhausted; configure an explicit rolling KV window")
        embeddings = self.model.language_model.model.embed_tokens(self.mx.array([ids])) + audio_embeds
        logits = self.model.language_model(embeddings, self.scales, self.text_cache)[0, -1]
        logits = self.mx.where(self.mx.arange(vocabulary) == self.special_ids["eos_token_id"], float("-inf"), logits)
        token = self.mx.argmax(logits)
        # mlx-lm's concat path retains capacity + query_length - 1 physical
        # slots until the next tick. Compact AFTER the forward, not before its
        # causal attention; keep absolute RoPE offset and the LM sink prefix.
        # The trims stay lazy so the token and caches resolve in one evaluation.
        pending = [token]
        for cache in self.audio_cache + self.text_cache:
            maximum = getattr(cache, "max_size", None)
            if maximum is not None and cache.keys.shape[-2] > maximum:
                excess = cache.keys.shape[-2] - maximum
                cache.keys = cache._trim(excess, cache.keys)
                cache.values = cache._trim(excess, cache.values)
                cache._idx = maximum
                pending.extend([cache.keys, cache.values])
        self.mx.eval(pending)
        return int(token.item())

    def step(self, audio_window, input_token_ids):
        ids = list(input_token_ids)
        return self.text_step(ids, self.encode_window(audio_window, len(ids)))

    def decode_visible_text(self, token_ids):
        excluded = set(self.special_ids.values())
        visible = []
        for token in token_ids:
            if token == self.special_ids["eos_token_id"]:
                break
            if token not in excluded:
                visible.append(int(token))
        return self.tokenizer.decode(visible, skip_special_tokens=True)

    @property
    def absolute_audio_position(self):
        return self.audio_cache[0].offset

    @property
    def absolute_text_position(self):
        return self.text_cache[0].offset

    def synchronize(self):
        arrays = self.scales + [value for cache in self.audio_cache + self.text_cache
                                for value in (cache.keys, cache.values) if value is not None]
        self.mx.eval(arrays)
        self.mx.synchronize()

    def cache_stats(self):
        audio_bytes = sum(value.nbytes for cache in self.audio_cache for value in (cache.keys, cache.values) if value is not None)
        text_bytes = sum(value.nbytes for cache in self.text_cache for value in (cache.keys, cache.values) if value is not None)
        frames = int(self.audio_cache[0].keys.shape[-2]) if self.audio_cache[0].keys is not None else 0
        tokens = self.text_cache[0].size()
        return {"audio_position": self.absolute_audio_position, "text_position": self.absolute_text_position,
                "absolute_audio_position": self.absolute_audio_position, "absolute_text_position": self.absolute_text_position,
                "audio_kv_frames": self.audio_cache[0].size(), "text_kv_tokens": tokens,
                "audio_kv_storage_frames": frames, "audio_cache_frames": frames, "text_cache_tokens": tokens,
                "cache_bytes": audio_bytes + text_bytes, "audio_cache_bytes": audio_bytes, "text_cache_bytes": text_bytes}

    def decode_waveform(self, waveform, language="en"):
        """Bounded-fixture convenience wrapper, not the parent streaming protocol.

        Matches the canonical batch=1 simulation's ceil-padded sample timeline.
        Long file/request lifecycle and incremental PCM buffering belong to parent.
        """
        if language not in self.language_ids:
            raise ValueError("Audio8 requires explicit language en or zh")
        samples = self.np.asarray(waveform, dtype=self.np.float32)
        if samples.ndim != 1 or not self.np.isfinite(samples).all():
            raise ValueError("Audio8 samples must be finite mono PCM")
        self.reset()
        spt, left, delay, tail = 2560, 9, 3, 10
        real_tokens = max(1, (len(samples) + spt - 1) // spt)
        stream = self.np.zeros((left + real_tokens + delay + 1 + tail) * spt, dtype=self.np.float32)
        stream[left * spt:left * spt + len(samples)] = self.np.clip(samples, -1, 1)
        prompt = [self.special_ids["bos_token_id"], self.language_ids[language]]
        prompt += [self.special_ids["streaming_pad_token_id"]] * (left + delay + 1 - len(prompt))
        generated = []
        start, end = 0, len(prompt) * spt
        while end + 40 <= len(stream):
            window = stream[max(start - 840, 0):end + 40]
            token = self.step(window, prompt)
            generated.append(token)
            prompt = [token]
            start, end = end, end + spt
        return {"final_text": self.decode_visible_text(generated), "generated_token_ids": generated}


class MlxRuntime:
    """One loaded model per worker; calibration/live sessions share its weights.

    raw decode input is mono little-endian float32 PCM at 16kHz, matching the
    shared worker. Thread controls are validated no-ops on Metal, not CPU tuning.
    """
    def __init__(self, directory, no_gpu=False, gpu_device=0, threads=4):
        if type(no_gpu) is not bool:
            raise ValueError("Audio8 MLX no_gpu must be a boolean")
        if no_gpu:
            raise RuntimeError("Audio8 MLX requires Metal; no_gpu is unsupported")
        if type(gpu_device) is not int or gpu_device != 0:
            raise RuntimeError("Audio8 MLX supports only the default Apple Metal device (gpu_device=0)")
        self.set_threads(threads)
        self._loaded = MlxStepDecoder(directory)
        for name in ("np", "mx", "tokenizer", "special_ids", "language_ids", "device", "dtype", "model", "descriptor"):
            setattr(self, name, getattr(self._loaded, name))
        self.backend, self.checkpoint, self.revision, self.package_revision = "mlx", CHECKPOINT, REVISION, PACKAGE_REVISION
        validation = self.descriptor.get("validation", {})
        self.source_git_verified = (self.descriptor["package_revision"] == PACKAGE_REVISION and
            validation.get("status") == "fp16-parity-passed" and
            validation.get("token_parity") is True and validation.get("text_parity") is True)

    def set_threads(self, threads):
        if type(threads) is not int or threads < 0:
            raise ValueError("Audio8 MLX threads must be a non-negative integer; Metal thread tuning is not supported")
        self.threads = threads  # 0 means unspecified; every value is a Metal no-op.

    def new_step_decoder(self, window_tokens):
        if type(window_tokens) is not int or not 14 <= window_tokens <= 512:
            raise ValueError("Audio8 MLX window_tokens must be 14..512")
        trained_window = int(self.model.config["audio_config"]["sliding_window"])
        return MlxStepDecoder.from_loaded(self._loaded, text_window_tokens=window_tokens,
                                         keep_prompt_tokens=13, audio_window_frames=min(window_tokens * 8, trained_window))

    def decode(self, raw, language):
        if language not in self.language_ids:
            raise ValueError("Audio8 requires explicit language en or zh")
        if not isinstance(raw, (bytes, bytearray, memoryview)) or memoryview(raw).nbytes % 4:
            raise ValueError("Audio8 MLX raw PCM must be little-endian float32 bytes")
        samples = self.np.frombuffer(raw, dtype="<f4")
        if not self.np.isfinite(samples).all():
            raise ValueError("Audio8 MLX samples must be finite")
        if samples.size == 0:
            return ""
        decoder = MlxStepDecoder.from_loaded(self._loaded)
        return decoder.decode_waveform(samples, language)["final_text"]

    def synchronize(self):
        self._loaded.synchronize()


# Backend remains the step-level API; host workers import MlxRuntime explicitly.
Backend = MlxStepDecoder
