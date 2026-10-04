"""Portable ONNX implementation of the reviewed Audio8 ASR weight format.

Only standard-domain opset 18 operators are emitted. We deliberately construct
small graphs directly instead of tracing a 16GB Torch model (which duplicates
weights and specialises the mutable Transformers cache). All KV state is an
explicit graph input/output; zero-length past supports prefill in the same graph.
Reference: upstream c8ba8eea829be0339e8d7757f8ca52dac06e1e32 and
transformers 5.13.0 VoxtralRealtimeEncoder/Qwen2Attention implementations.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as H, numpy_helper


class Graph:
    def __init__(self, name, weights):
        self.name, self.weights = name, weights
        self.nodes, self.initializers, self.inputs, self.outputs = [], [], [], []
        self.used = set()
        self.weight_shapes = {}
        self.counter = 0

    def op(self, kind, *args, output=None, **attrs):
        self.counter += 1
        output = output or f"n{self.counter}"
        self.nodes.append(H.make_node(kind, list(args), [output], **attrs))
        return output

    def const(self, value, dtype=None):
        self.counter += 1
        name = f"c{self.counter}"
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def weight(self, name):
        if name not in self.used:
            tensor = self.weights.initializer(name)
            self.initializers.append(tensor)
            self.weight_shapes[name] = tuple(tensor.dims)
            self.used.add(name)
        return name

    def input(self, name, shape, dtype=T.FLOAT):
        self.inputs.append(H.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, name, shape, dtype=T.FLOAT):
        self.outputs.append(H.make_tensor_value_info(name, dtype, shape))

    def reshape(self, x, shape):
        return self.op("Reshape", x, self.const(shape, np.int64))

    def unsqueeze(self, x, axes):
        return self.op("Unsqueeze", x, self.const(axes, np.int64))

    def slice(self, x, start, end, axis):
        return self.op("Slice", x, self.const([start], np.int64),
                       self.const([end], np.int64), self.const([axis], np.int64))

    def linear(self, x, prefix):
        """Torch Linear for arbitrary leading dims, without materialising W.T."""
        w = self.weight(prefix + ".weight")
        dims = self.weight_shapes[w]
        if len(dims) != 2:
            raise ValueError(f"Linear weight must be a matrix: {w}")
        out_features, in_features = dims
        args = [self.reshape(x, [-1, in_features]), w]
        if prefix + ".bias" in self.weights.names:
            bias = self.weight(prefix + ".bias")
            if self.weight_shapes[bias] != (out_features,):
                raise ValueError(f"Linear bias shape mismatch: {bias}")
            args.append(bias)
        y = self.op("Gemm", *args, transB=1)
        leading = self.slice(self.op("Shape", x), 0, -1, 0)
        shape = self.op("Concat", leading, self.const([out_features], np.int64), axis=0)
        return self.op("Reshape", y, shape)

    def gelu(self, x):
        erf = self.op("Erf", self.op("Div", x, self.const(math.sqrt(2), np.float32)))
        return self.op("Mul", self.op("Mul", x, self.const(0.5, np.float32)),
                       self.op("Add", erf, self.const(1, np.float32)))

    def norm(self, x, prefix, eps):
        square = self.op("Mul", x, x)
        mean = self.op("ReduceMean", square, self.const([-1], np.int64), keepdims=1)
        denom = self.op("Sqrt", self.op("Add", mean, self.const(eps, np.float32)))
        return self.op("Mul", self.op("Div", x, denom), self.weight(prefix + ".weight"))

    def mlp(self, x, prefix):
        gate = self.linear(x, prefix + ".gate_proj")
        gate = self.op("Mul", gate, self.op("Sigmoid", gate))
        return self.linear(self.op("Mul", gate, self.linear(x, prefix + ".up_proj")),
                           prefix + ".down_proj")

    def rope(self, positions, dim, theta):
        inv = 1.0 / (np.float32(theta) ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        phase = self.op("Mul", self.unsqueeze(self.op("Cast", positions, to=T.FLOAT), [-1]),
                        self.const(inv.astype(np.float32)))
        phase = self.op("Concat", phase, phase, axis=-1)
        return self.unsqueeze(self.op("Cos", phase), [1]), self.unsqueeze(self.op("Sin", phase), [1])

    def rotate(self, x, cos, sin, dim):
        half = self.op("Concat", self.op("Neg", self.slice(x, dim // 2, dim, -1)),
                       self.slice(x, 0, dim // 2, -1), axis=-1)
        return self.op("Add", self.op("Mul", x, cos), self.op("Mul", half, sin))

    def attention(self, x, prefix, cfg, index, cos, sin, mask):
        heads = cfg["num_attention_heads"]
        kv_heads = cfg.get("num_key_value_heads", heads)
        dim = cfg.get("head_dim", cfg["hidden_size"] // heads)
        q, k, v = [self.op("Transpose", self.reshape(self.linear(x, prefix + "." + n + "_proj"),
                                                        [0, 0, -1, dim]), perm=[0, 2, 1, 3])
                   for n in ("q", "k", "v")]
        q, k = self.rotate(q, cos, sin, dim), self.rotate(k, cos, sin, dim)
        for kind, current in (("key", k), ("value", v)):
            past = self.input(f"past_{kind}_{index}", ["batch", kv_heads, "past", dim])
            out = f"present_{kind}_{index}"
            merged = self.op("Concat", past, current, axis=2, output=out)
            self.output(out, ["batch", kv_heads, "present", dim])
            if kind == "key":
                k = merged
            else:
                v = merged
        if heads != kv_heads:
            # Gather repeats each KV head consecutively, matching repeat_kv.
            repeats = self.const(np.repeat(np.arange(kv_heads), heads // kv_heads), np.int64)
            k, v = self.op("Gather", k, repeats, axis=1), self.op("Gather", v, repeats, axis=1)
        scores = self.op("Mul", self.op("MatMul", q, self.op("Transpose", k, perm=[0, 1, 3, 2])),
                         self.const(dim ** -0.5, np.float32))
        scores = self.op("Add", scores, mask)
        context = self.op("MatMul", self.op("Softmax", scores, axis=-1), v)
        context = self.reshape(self.op("Transpose", context, perm=[0, 2, 1, 3]),
                               [0, 0, heads * dim])
        return self.linear(context, prefix + ".o_proj")

    def save(self, folder):
        graph = H.make_graph(self.nodes, self.name, self.inputs, self.outputs, self.initializers)
        model = H.make_model(graph, opset_imports=[H.make_opsetid("", 18)], producer_name="vibe-audio8")
        # ORT 1.23 supports IR 10, not ONNX 1.19's default IR 12.
        model.ir_version = 10
        path = Path(folder) / (self.name + ".onnx")
        onnx.save_model(model, path)
        onnx.checker.check_model(str(path))
        def describe(values):
            return [{"name": v.name, "dtype": T.DataType.Name(v.type.tensor_type.elem_type).lower(),
                     "shape": [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]}
                    for v in values]
        return {"file": path.name, "inputs": describe(self.inputs), "outputs": describe(self.outputs)}


def build_embedder(cfg, weights, folder):
    g = Graph("embedder", weights)
    x = g.input("input_features", ["batch", cfg["num_mel_bins"], "mel_frames"])
    for i, stride in ((1, 1), (2, 2)):
        p = f"audio_tower.embedder.conv{i}"
        x = g.op("Conv", x, g.weight(p + ".weight"), g.weight(p + ".bias"),
                 pads=[3 - stride, 0], strides=[stride], kernel_shape=[3])
        x = g.gelu(x)
    g.op("Transpose", x, perm=[0, 2, 1], output="conv_embeds")
    g.output("conv_embeds", ["batch", "conv_frames", cfg["hidden_size"]])
    return g.save(folder), g.used


def build_encoder(config, weights, folder):
    cfg = config["audio_config"]
    g = Graph("encoder", weights)
    x = g.input("conv_embeds", ["batch", "frames", cfg["hidden_size"]])
    pos = g.input("position_ids", ["batch", "frames"], T.INT64)
    mask = g.input("attention_mask", ["batch", 1, "frames", "present"])
    cos, sin = g.rope(pos, cfg["head_dim"], cfg["rope_parameters"]["rope_theta"])
    for i in range(cfg["num_hidden_layers"]):
        p = f"audio_tower.layers.{i}"
        h = g.norm(x, p + ".self_attn_layer_norm", cfg["rms_norm_eps"])
        x = g.op("Add", x, g.attention(h, p + ".self_attn", cfg, i, cos, sin, mask))
        h = g.norm(x, p + ".final_layer_norm", cfg["rms_norm_eps"])
        x = g.op("Add", x, g.mlp(h, p + ".mlp"))
    x = g.norm(x, "audio_tower.norm", cfg["rms_norm_eps"])
    # Export contract fixes frame_len=8, the server's 160ms gear. Inputs must
    # contain a whole number of groups; host trims convolution overlap first.
    x = g.reshape(x, [0, -1, config["max_frame_len"] * cfg["hidden_size"]])
    x = g.gelu(g.linear(x, "multi_modal_projector.linear_1"))
    x = g.linear(x, "multi_modal_projector.linear_2")
    g.op("Identity", x, output="audio_embeds")
    g.output("audio_embeds", ["batch", "tokens", config["text_config"]["hidden_size"]])
    return g.save(folder), g.used


def build_decoder(config, weights, folder):
    cfg = config["text_config"]
    g = Graph("decoder", weights)
    ids = g.input("input_ids", ["batch", "tokens"], T.INT64)
    audio = g.input("audio_embeds", ["batch", "tokens", cfg["hidden_size"]])
    pos = g.input("position_ids", ["batch", "tokens"], T.INT64)
    mask = g.input("attention_mask", ["batch", 1, "tokens", "present"])
    delay = g.input("num_delay_tokens", ["batch"], T.FLOAT)
    x = g.op("Add", g.op("Gather", g.weight("language_model.model.embed_tokens.weight"), ids), audio)
    # Same fp32 sinusoidal time embedding as upstream build_t_cond.
    half = cfg["hidden_size"] // 2
    inv = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half).astype(np.float32)
    phase = g.op("Mul", g.unsqueeze(delay, [-1]), g.const(inv))
    cond = g.op("Concat", g.op("Cos", phase), g.op("Sin", phase), axis=-1)
    frame_index = config["supported_frame_lens"].index(8)
    frame = g.op("Gather", g.weight("frame_len_embedding.weight"), g.const(frame_index, np.int64))
    cond = g.unsqueeze(g.op("Add", cond, frame), [1])
    dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    cos, sin = g.rope(pos, dim, cfg["rope_parameters"]["rope_theta"])
    for i in range(cfg["num_hidden_layers"]):
        p = f"language_model.model.layers.{i}"
        h = g.norm(x, p + ".input_layernorm", cfg["rms_norm_eps"])
        x = g.op("Add", x, g.attention(h, p + ".self_attn", cfg, i, cos, sin, mask))
        h = g.norm(x, p + ".post_attention_layernorm", cfg["rms_norm_eps"])
        scale = g.linear(g.gelu(g.linear(cond, p + ".ada_rms_norm.linear1")),
                         p + ".ada_rms_norm.linear2")
        h = g.op("Mul", h, g.op("Add", scale, g.const(1, np.float32)))
        x = g.op("Add", x, g.mlp(h, p + ".mlp"))
    x = g.norm(x, "language_model.model.norm", cfg["rms_norm_eps"])
    x = g.slice(x, -1, np.iinfo(np.int64).max, 1)
    # Pinned checkpoint ties lm_head to embed_tokens; keep one external weight.
    g.op("Identity", g.linear(x, "language_model.model.embed_tokens"), output="logits")
    g.output("logits", ["batch", 1, cfg["vocab_size"]])
    return g.save(folder), g.used


def build_all(config, weights, folder):
    graphs, used = {}, set()
    for name, builder in (("embedder", build_embedder), ("encoder", build_encoder), ("decoder", build_decoder)):
        print(f"Exporting {name}", flush=True)
        cfg = config["audio_config"] if name == "embedder" else config
        graphs[name], keys = builder(cfg, weights, folder)
        used.update(keys)
    expected = set(weights.names) - {"language_model.lm_head.weight"}
    if used != expected:
        raise ValueError(f"Graph weight coverage mismatch: unused={sorted(expected - used)}, extra={sorted(used - expected)}")
    return graphs
