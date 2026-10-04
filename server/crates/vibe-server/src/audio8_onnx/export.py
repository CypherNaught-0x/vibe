# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "audio8-asr-infinite @ git+https://github.com/Edge0-AI/Audio8-ASR-Infinite@c8ba8eea829be0339e8d7757f8ca52dac06e1e32",
#   "torch==2.13.0", "transformers==5.13.0", "onnx==1.22.0",
#   "onnxruntime==1.23.2", "numpy==2.4.4", "safetensors==0.8.0", "tokenizers==0.22.2",
# ]
# ///
"""Export the pinned local Audio8 checkpoint to portable fp32 ONNX graphs.

uv run export.py --checkpoint <local HF snapshot> --output <new directory>

The output directory must not exist. No network/model remote code is used by
this script. The reviewed upstream package is only used for strict meta-device
weight-shape validation; no full Torch weight model is allocated. BF16 weights
are streamed to shared external fp32 weights in bounded blocks. Conversion
requires ~16.35GB disk; CPU inference still needs ~17GB RAM, not a small model.
The unvalidated descriptor is audio8_onnx.json. A small selectable marker is
written only by the separate pretrained parity harness after every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import TensorProto as T

from graphs import build_all

CHECKPOINT = "Edge0/Audio8-ASR-Infinite"
REVISION = "7476824bc222e4ad509d286e8cae8b8d3f371129"
UPSTREAM = "c8ba8eea829be0339e8d7757f8ca52dac06e1e32"
# HF revision API (blobs=true), inspected 2026-10-04. LFS SHA is content SHA256.
WEIGHT_SHA = "2cf97d69e9f5853855b783b359dffb661c035281a02053782c21fe009e31c7ba"
ASSET_SHA = {
    "config.json": "744baed356a8c86a9791fc76cecb73219ed4aff795147c6cdc082337a13f019c",
    "preprocessor_config.json": "ee0dac67c684faa527db5d26a75109e168723fdae38b49e0c1bb471a953f6c45",
    "tokenizer.json": "72f8b7e878ee6606c309881f68201e277f9dae50175a387074a5181d771b9599",
    "tokenizer_config.json": "ceca11f549d6a8550430c7272e9aa40582bffd633f8998335883a428f4d77fb9",
    "generation_config.json": "d3f96050ed55e02136d3cea2d1c03c0163d8e93631cdfb87bf0e82945ed2db64",
    "chat_template.jinja": "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_package():
    distribution = importlib.metadata.distribution("audio8-asr-infinite")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (direct.get("url", "").removesuffix(".git") != "https://github.com/Edge0-AI/Audio8-ASR-Infinite"
            or direct.get("vcs_info", {}).get("commit_id") != UPSTREAM):
        raise ValueError("Audio8 upstream must be installed from the pinned reviewed Git commit")


class ExternalWeights:
    """Shared external-data writer; one tensor is streamed at a time, not loaded."""
    def __init__(self, path, header, data_start, output):
        self.path, self.header, self.data_start = Path(path), header, data_start
        self.names = set(header)
        self.output = Path(output)
        self.locations = {}
        self.used = set()
        self.digest = hashlib.sha256()

    def initializer(self, name):
        metadata = self.header[name]
        if name not in self.locations:
            start, end = metadata["data_offsets"]
            count = math.prod(metadata["shape"])
            offset = self.output.stat().st_size if self.output.exists() else 0
            source = np.memmap(self.path, dtype="<u2", mode="r", offset=self.data_start + start, shape=(count,))
            with self.output.open("ab") as stream:
                for first in range(0, count, 4 * 1024 * 1024):
                    values = (source[first:first + 4 * 1024 * 1024].astype(np.uint32) << 16).view(np.float32)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite checkpoint tensor: {name}")
                    encoded = values.astype("<f4", copy=False).tobytes()
                    stream.write(encoded)
                    self.digest.update(encoded)
            del source
            self.locations[name] = (offset, count * 4)
        offset, length = self.locations[name]
        tensor = T(name=name, data_type=T.FLOAT, dims=metadata["shape"], data_location=T.EXTERNAL)
        for key, value in (("location", self.output.name), ("offset", str(offset)), ("length", str(length))):
            entry = tensor.external_data.add()
            entry.key, entry.value = key, value
        self.used.add(name)
        return tensor


def weight_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 4 * 1024 * 1024:
            raise ValueError("Oversized safetensors header")
        header = json.loads(stream.read(size))
    if header.pop("__metadata__", {}) != {"format": "pt"}:
        raise ValueError("Unsupported safetensors format metadata")
    intervals = []
    for name, meta in header.items():
        if set(meta) != {"shape", "dtype", "data_offsets"} or meta["dtype"] != "BF16":
            raise ValueError(f"Unsupported tensor metadata: {name}")
        shape = meta["shape"]
        if not shape or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        start, end = meta["data_offsets"]
        if type(start) is not int or type(end) is not int or start < 0 or end - start != math.prod(shape) * 2:
            raise ValueError(f"Invalid tensor extent: {name}")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Safetensors offsets overlap or contain a gap")
        cursor = end
    if cursor + size + 8 != path.stat().st_size:
        raise ValueError("Safetensors file size disagrees with tensor metadata")
    return header, size + 8


def validate_shapes(config_path, header):
    import torch
    from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
    from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import Audio8ASRInfiniteForConditionalGeneration
    cfg = Audio8ASRInfiniteConfig.from_pretrained(config_path, local_files_only=True)
    # Public ASR weights omit the separately distributed semantic VAD heads.
    cfg.semantic_vad_horizons_seconds = None
    with torch.device("meta"):
        model = Audio8ASRInfiniteForConditionalGeneration(cfg)
    expected = {name: list(tensor.shape) for name, tensor in model.state_dict().items()
                if name != "language_model.lm_head.weight"}
    actual = {name: meta["shape"] for name, meta in header.items()}
    if actual != expected:
        raise ValueError(f"Strict ASR weight mismatch: missing={sorted(expected.keys() - actual.keys())}, "
                         f"extra={sorted(actual.keys() - expected.keys())}, "
                         f"shapes={[k for k in actual.keys() & expected.keys() if actual[k] != expected[k]]}")
    return cfg.to_dict()


class ExistingWeights:
    """TensorProto metadata only; never opens/maps/rewrites the weight blob."""
    def __init__(self, source, descriptor):
        from backend import external_initializers
        self.tensors, self.header = {}, {}
        self.names = set()
        for graph in descriptor["graphs"].values():
            data = (source / graph["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != graph["sha256"]:
                raise ValueError(f"Source graph SHA256 mismatch: {graph['file']}")
            metadata = external_initializers(data)
            model = onnx.load_model_from_string(data)  # external data stays unloaded
            for tensor in model.graph.initializer:
                if tensor.name not in metadata:
                    continue
                shape, offset, length = metadata[tensor.name]
                if length != math.prod(shape) * 4 or offset % 4 or offset + length > descriptor["weights"]["size"]:
                    raise ValueError(f"Invalid source tensor extent: {tensor.name}")
                if tensor.name in self.tensors and self.tensors[tensor.name] != tensor:
                    raise ValueError(f"Conflicting source initializer: {tensor.name}")
                self.tensors[tensor.name] = tensor
                self.header[tensor.name] = {"shape": list(shape)}
        self.names = set(self.tensors)

    def initializer(self, name):
        tensor = T()
        tensor.CopyFrom(self.tensors[name])
        return tensor


def clone_blob(source, target):
    """Filesystem copy-on-write clone; ONNX 1.22 deliberately rejects hardlinks.

    No userspace data read/write, and no precision/layout conversion. Fail
    explicitly if the filesystem cannot clone rather than falling back to a
    silent 16GB copy during a metadata-only operation.
    """
    import sys
    if sys.platform == "darwin":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        clone = libc.clonefile
        clone.argtypes, clone.restype = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int], ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(target), 0) != 0:
            raise OSError(ctypes.get_errno(), "Copy-on-write weight clone failed")
    elif sys.platform.startswith("linux"):
        import fcntl
        with source.open("rb") as original, target.open("xb") as output:
            fcntl.ioctl(output.fileno(), 0x40049409, original.fileno())  # FICLONE
    else:
        raise RuntimeError("Metadata-only regeneration needs a clonefile/FICLONE filesystem")


def regenerate(source, output):
    """Graph-only Gemm regeneration, with unchanged cloned external bytes."""
    descriptor_path = source / "audio8_onnx.json"
    d = json.loads(descriptor_path.read_text())
    if (d.get("checkpoint") != CHECKPOINT or d.get("revision") != REVISION
            or d.get("upstream_revision") != UPSTREAM or d.get("dtype") != "float32"
            or d.get("weights", {}).get("file") != "weights.bin"
            or d.get("checkpoint_sha256") != WEIGHT_SHA
            or d.get("validation", {}).get("strict_weight_metadata") is not True):
        raise ValueError("Not a strict pinned FP32 Audio8 source export")
    if output.exists():
        raise ValueError("Output must be a new directory")
    original = source / "weights.bin"
    if original.stat().st_size != d["weights"]["size"]:
        raise ValueError("Source weight blob size mismatch")
    for name, digest in d["assets_sha256"].items():
        if Path(name).name != name or sha256(source / name) != digest:
            raise ValueError(f"Source asset SHA256 mismatch: {name}")
    weights = ExistingWeights(source, d)
    cfg = validate_shapes(source, weights.header)
    output.mkdir(parents=True)
    clone_blob(original, output / "weights.bin")
    for name in d["assets_sha256"]:
        (output / name).write_bytes((source / name).read_bytes())
    graphs = build_all(cfg, weights, output)
    counts = {}
    for name, graph in graphs.items():
        path = output / graph["file"]
        graph["sha256"] = sha256(path)
        model = onnx.load(path, load_external_data=False)
        metadata = {v.name: v for v in model.graph.initializer if v.data_location == T.EXTERNAL}
        if any(v != weights.tensors[k] for k, v in metadata.items()):
            raise ValueError("Gemm regeneration changed external tensor metadata")
        weight_transposes = [n for n in model.graph.node if n.op_type == "Transpose" and n.input[0] in metadata]
        if weight_transposes:
            raise ValueError("Gemm regeneration retained a weight Transpose")
        counts[name] = {"gemm": sum(n.op_type == "Gemm" for n in model.graph.node),
                        "activation_transposes": sum(n.op_type == "Transpose" for n in model.graph.node)}
        if graph["inputs"] != d["graphs"][name]["inputs"] or graph["outputs"] != d["graphs"][name]["outputs"]:
            raise ValueError("Gemm regeneration changed public graph interface")
    target = (output / "weights.bin").stat()
    old = original.stat()
    if target.st_size != old.st_size or target.st_nlink != 1:
        raise ValueError("Gemm output must be a complete, independently linked clone")
    d["optimization"] = {"linear": "reshape-gemm-transB1", "source_descriptor_sha256": sha256(descriptor_path),
                         "source_graph_sha256": {k: v["sha256"] for k, v in d["graphs"].items()},
                         "weights_reused_without_read_or_rewrite": True, "weight_storage": "copy-on-write clone", "nodes": counts}
    d["graphs"] = graphs
    d["validation"] = {"strict_weight_metadata": True, "checkpoint_sha256": True, "onnx_checker": True,
                       "pretrained_numerical_parity": False, "pretrained_token_parity": False, "pretrained_text_parity": False}
    (output / "audio8_onnx.json").write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps({"descriptor": str(output / "audio8_onnx.json"), "weights_unchanged_clone": True,
                      "strict_tensors": len(weights.names), "nodes": counts, "pretrained_parity": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--reuse-export", type=Path, help="metadata-only graph regeneration, reusing unchanged weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true", help="Validate small assets and weight header, without reading tensor data")
    args = parser.parse_args()
    verify_package()
    if args.reuse_export:
        if args.inspect_only:
            parser.error("--inspect-only is only for checkpoint metadata")
        regenerate(args.reuse_export, args.output)
        return
    assets = {}
    for name, digest in ASSET_SHA.items():
        source = args.checkpoint / name
        if sha256(source) != digest:
            raise ValueError(f"Pinned checkpoint asset SHA256 mismatch: {name}")
        assets[name] = source
    header, data_start = weight_header(args.checkpoint / "model.safetensors")
    cfg = validate_shapes(args.checkpoint, header)
    print(json.dumps({"checkpoint": CHECKPOINT, "revision": REVISION, "tensors": len(header),
                      "strict_shapes": True, "weight_hash_verified": False}), flush=True)
    if args.inspect_only:
        return
    if args.output.exists():
        raise ValueError("Output must be a new directory; existing conversions are never overwritten")
    print("Verifying pinned checkpoint SHA256", flush=True)
    if sha256(args.checkpoint / "model.safetensors") != WEIGHT_SHA:
        raise ValueError("Pinned checkpoint model.safetensors SHA256 mismatch")
    args.output.mkdir(parents=True)
    for name, source in assets.items():
        (args.output / name).write_bytes(source.read_bytes())
    # Capture the canonical Slaney filterbank once; deployment needs only NumPy.
    from transformers.audio_utils import mel_filter_bank
    mel = mel_filter_bank(201, 128, 0.0, 8000.0, 16000, norm="slaney", mel_scale="slaney")
    np.save(args.output / "mel_filters.npy", mel.astype(np.float32), allow_pickle=False)
    weights = ExternalWeights(args.checkpoint / "model.safetensors", header, data_start, args.output / "weights.bin")
    graphs = build_all(cfg, weights, args.output)
    for graph in graphs.values():
        graph["sha256"] = sha256(args.output / graph["file"])
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.output / "tokenizer.json"))
    special_ids = {name: tokenizer.token_to_id(token) for name, token in
                   (("bos_token_id", "<|im_start|>"), ("eos_token_id", "<|im_end|>"), ("pad_token_id", "<|endoftext|>"),
                    ("streaming_pad_token_id", "[STREAMING_PAD]"), ("streaming_word_token_id", "[STREAMING_WORD]"))}
    language_ids = {name: tokenizer.token_to_id(token) for name, token in
                    (("en", "[LANGUAGE_EN]"), ("zh", "[LANGUAGE_ZH]"))}
    if any(value is None for value in (*special_ids.values(), *language_ids.values())):
        raise ValueError("Pinned tokenizer lacks streaming IDs")
    descriptor = {
        "format_version": 1, "engine": "audio8", "backend": "onnx", "checkpoint": CHECKPOINT,
        "revision": REVISION, "upstream_revision": UPSTREAM, "checkpoint_sha256": WEIGHT_SHA,
        "package_revision": UPSTREAM, "source_git_verified": True,
        "special_ids": special_ids, "language_ids": language_ids,
        "opset": 18, "dtype": "float32", "graphs": graphs,
        "weights": {"file": "weights.bin", "size": (args.output / "weights.bin").stat().st_size,
                    "sha256": weights.digest.hexdigest()},
        "host": {"implementation": "numpy", "torch_required": False, "sample_rate": 16000,
                 "frame_len": 8, "num_delay_tokens": 3, "left_pad_tokens": 9, "right_pad_text_tokens": 10,
                 "look_back_samples": 840, "look_ahead_samples": 40, "max_chunk_samples": 384000,
                 "frontend": "centered reflect-pad periodic-Hann STFT; drop last frame; Slaney power mel; log10 floor -6.5; (x+4)/4"},
        "audio_config": cfg["audio_config"], "text_config": cfg["text_config"],
        "assets_sha256": {**ASSET_SHA, "mel_filters.npy": sha256(args.output / "mel_filters.npy")},
        "validation": {"strict_weight_metadata": True, "checkpoint_sha256": True,
                       "onnx_checker": True, "pretrained_numerical_parity": False},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "onnx", "onnxruntime", "numpy", "safetensors")},
    }
    path = args.output / "audio8_onnx.json"
    path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
