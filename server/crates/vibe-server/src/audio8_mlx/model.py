"""Audio8's Voxtral encoder + delay-conditioned Qwen2, not vanilla Voxtral.

Reviewed implementations reused from mlx-audio 0.4.4 / mlx-lm 0.31.3:
https://github.com/Blaizzy/mlx-audio/tree/v0.4.4/mlx_audio/stt/models/voxtral_realtime
https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen2.py
Audio8 contract:
https://github.com/Edge0-AI/Audio8-ASR-Infinite/blob/c8ba8eea829be0339e8d7757f8ca52dac06e1e32/audio8_asr_infinite/modeling/modeling_audio8_asr_infinite.py

Voxtral's public loader cannot load Audio8: its decoder is Mistral, projector
is 4-frame/3072-wide and tokenizer is Tekken. Reuse only verified encoder
layers/causal conv and Qwen2 attention/MLP; implement Audio8's conditioning.
"""
from __future__ import annotations

import math

from .common import require_apple_silicon

require_apple_silicon()
import mlx.core as mx
import mlx.nn as nn
from mlx_audio.stt.models.voxtral_realtime.config import EncoderConfig
from mlx_audio.stt.models.voxtral_realtime.encoder import CausalConv1d, EncoderLayer
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.qwen2 import ModelArgs, TransformerBlock


def validate_config(config):
    if config.get("model_type") != "audio8_asr_infinite" or config.get("weight_format_version") != 2:
        raise ValueError("Expected Audio8 Infinite weight format 2")
    audio, text = config["audio_config"], config["text_config"]
    if audio["model_type"] != "voxtral_realtime_encoder" or text["model_type"] != "qwen2":
        raise ValueError("Only the pinned Voxtral Realtime/Qwen2 architecture is supported")
    for part in (audio, text):
        if part.get("rope_parameters", {}).get("rope_type", "default") != "default":
            raise ValueError("Unsupported Audio8 RoPE scaling")
        if part.get("hidden_act") != "silu":
            raise ValueError("Audio8 MLP must use SwiGLU")
    if audio.get("activation_function") != "gelu" or config.get("projector_hidden_act") != "gelu":
        raise ValueError("Audio8 convolution/projector must use exact GELU")
    if not text.get("tie_word_embeddings") or text.get("use_sliding_window"):
        raise ValueError("Expected tied, full-attention Qwen2 weights")
    if config.get("projection_size") != audio["hidden_size"] * config["max_frame_len"]:
        raise ValueError("Audio8 projector width does not match max frame length")
    if not config.get("use_frame_len_embedding"):
        raise ValueError("Pinned Audio8 requires frame-length conditioning")


def interleave_qk(value, heads, head_dim):
    """HF rotate_half rows -> mlx-audio's traditional/interleaved RoPE rows.

    Applies to q/k weights AND q bias. v/o must not be permuted.
    """
    if value.shape[0] != heads * head_dim:
        raise ValueError("Invalid Voxtral q/k projection shape")
    return value.reshape(heads, 2, head_dim // 2, *value.shape[1:]).swapaxes(1, 2).reshape(value.shape)


def map_weight(name, value, config):
    """Strict PyTorch -> MLX mapping; never silently drop an ASR parameter."""
    if name.startswith("audio_tower.embedder.conv"):
        if name.endswith(".weight"):
            if value.ndim != 3:
                raise ValueError("Expected PyTorch [out,in,k] convolution weight")
            value = value.transpose(0, 2, 1)  # MLX [out,k,in]
        stem, parameter = name.rsplit(".", 1)
        return stem + ".conv." + parameter, value
    if name.startswith("audio_tower.layers."):
        mappings = {
            ".self_attn_layer_norm.": ".attention_norm.",
            ".final_layer_norm.": ".ffn_norm.",
            ".self_attn.q_proj.": ".attention.wq.",
            ".self_attn.k_proj.": ".attention.wk.",
            ".self_attn.v_proj.": ".attention.wv.",
            ".self_attn.o_proj.": ".attention.wo.",
            ".mlp.gate_proj.": ".feed_forward_w1.",
            ".mlp.up_proj.": ".feed_forward_w3.",
            ".mlp.down_proj.": ".feed_forward_w2.",
        }
        if ".self_attn.q_proj." in name or ".self_attn.k_proj." in name:
            audio = config["audio_config"]
            value = interleave_qk(value, audio["num_attention_heads"], audio["head_dim"])
        for old, new in mappings.items():
            if old in name:
                return name.replace(old, new), value
        raise ValueError(f"Unknown audio-tower parameter: {name}")
    if name.startswith(("audio_tower.norm.", "language_model.model.", "multi_modal_projector.", "frame_len_embedding.")):
        return name, value
    # The public file omits the tied LM head and non-persistent sinusoidal
    # buffers. VAD is a separate checkpoint and is not consumed by ASR.
    raise ValueError(f"Unexpected Audio8 ASR parameter: {name}")


class Embedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = CausalConv1d(config["num_mel_bins"], config["hidden_size"], 3)
        self.conv2 = CausalConv1d(config["hidden_size"], config["hidden_size"], 3, stride=2)

    def __call__(self, features):
        # [batch, mel, frames] -> NLC. No cache: each definition window has
        # its own centered STFT and conv overlap, trimmed by the caller.
        x = features.transpose(0, 2, 1)
        return nn.gelu(self.conv2(nn.gelu(self.conv1(x))))


class AudioTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = Embedder(config)
        args = EncoderConfig(dim=config["hidden_size"], n_layers=config["num_hidden_layers"],
                             n_heads=config["num_attention_heads"], head_dim=config["head_dim"],
                             hidden_dim=config["intermediate_size"], norm_eps=config["rms_norm_eps"],
                             rope_theta=config["rope_parameters"]["rope_theta"],
                             sliding_window=config["sliding_window"])
        self.layers = [EncoderLayer(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.window = args.sliding_window

    def make_cache(self, max_frames=None):
        capacity = self.window if max_frames is None else max_frames
        if type(capacity) is not int or capacity < 1 or capacity > self.window or (max_frames is not None and capacity < 104):
            raise ValueError("Audio KV capacity must cover prefill and not exceed the trained sliding window")
        return [RotatingKVCache(max_size=capacity, keep=0) for _ in self.layers]

    def __call__(self, conv_embeds, cache=None):
        if conv_embeds.shape[0] != 1:
            raise ValueError("Audio8 MLX streaming backend supports batch size 1")
        x = conv_embeds[0]
        length = x.shape[0]
        if cache is None:
            cache = [None] * len(self.layers)
            q = mx.arange(length)[:, None]
            k = mx.arange(length)[None, :]
            mask = (k <= q) & (k > q - self.window)
            offset = 0
        else:
            offset = cache[0].offset  # absolute RoPE offset, NOT stored KV length
            mask = cache[0].make_mask(length, window_size=self.window)
        for layer, item in zip(self.layers, cache, strict=True):
            x = layer(x, offset, mask, cache=item)
        return self.norm(x)[None]


class AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, 32, bias=False)
        self.linear2 = nn.Linear(32, hidden_size, bias=False)

    def __call__(self, t_cond):
        return self.linear2(nn.gelu(self.linear1(t_cond)))


class DelayBlock(TransformerBlock):
    def __init__(self, args):
        super().__init__(args)
        self.ada_rms_norm = AdaRMSNorm(args.hidden_size)

    def __call__(self, x, scale, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        conditioned = self.post_attention_layernorm(h) * (1 + scale.astype(h.dtype))
        return h + self.mlp(conditioned)


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        params = dict(config)
        params["rope_theta"] = config["rope_parameters"]["rope_theta"]
        args = ModelArgs.from_dict(params)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DelayBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs_embeds, scales, cache=None):
        cache = cache if cache is not None else [None] * len(self.layers)
        mask = create_attention_mask(inputs_embeds, cache[0])
        h = inputs_embeds
        for layer, scale, item in zip(self.layers, scales, cache, strict=True):
            h = layer(h, scale, mask, item)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TextModel(config)

    def __call__(self, inputs_embeds, scales, cache=None):
        h = self.model(inputs_embeds, scales, cache)
        return self.model.embed_tokens.as_linear(h[:, -1:, :])


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config["text_config"]["hidden_size"]
        self.linear_1 = nn.Linear(config["projection_size"], hidden, bias=False)
        self.linear_2 = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x):
        return self.linear_2(nn.gelu(self.linear_1(x)))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_config(config)
        self.config = config
        self.audio_tower = AudioTower(config["audio_config"])
        self.language_model = LanguageModel(config["text_config"])
        self.multi_modal_projector = Projector(config)
        self.frame_len_embedding = nn.Embedding(len(config["supported_frame_lens"]), config["text_config"]["hidden_size"])

    def build_t_cond(self, num_delay_tokens, frame_len, dtype):
        hidden = self.config["text_config"]["hidden_size"]
        # Match upstream: generate inv_freq in fp32, cast BEFORE multiplying
        # delay and applying trig. Casting only the final result is different.
        inv_freq = mx.exp(-math.log(10000.0) * mx.arange(hidden // 2, dtype=mx.float32) / (hidden // 2)).astype(dtype)
        phase = mx.array(float(num_delay_tokens), dtype=dtype) * inv_freq
        delay = mx.concatenate([mx.cos(phase), mx.sin(phase)])
        index = self.config["supported_frame_lens"].index(frame_len)
        return (delay + self.frame_len_embedding(mx.array([index]))[0]).reshape(1, 1, hidden)

    def delay_scales(self, delay, frame_len, dtype):
        t_cond = self.build_t_cond(delay, frame_len, dtype)
        return [layer.ada_rms_norm(t_cond) for layer in self.language_model.model.layers]

    def project(self, hidden, frame_len):
        if frame_len not in self.config["supported_frame_lens"] or hidden.shape[1] % frame_len:
            raise ValueError("Audio8 streaming frames must form complete supported groups")
        batch, frames, width = hidden.shape
        grouped = hidden.reshape(batch, frames // frame_len, frame_len, width)
        maximum = self.config["max_frame_len"]
        if frame_len < maximum:
            grouped = mx.pad(grouped, [(0, 0), (0, 0), (0, maximum - frame_len), (0, 0)])
        return self.multi_modal_projector(grouped.reshape(batch, frames // frame_len, maximum * width))

    def make_text_cache(self, max_tokens=None, keep=0):
        # None retains canonical full attention. Rolling context is an explicit
        # caller policy; absolute RoPE offsets survive eviction.
        if max_tokens is not None and (max_tokens <= keep or keep < 0):
            raise ValueError("Text KV window must exceed retained prompt tokens")
        return [KVCache() if max_tokens is None else RotatingKVCache(max_size=max_tokens, keep=keep)
                for _ in self.language_model.model.layers]
