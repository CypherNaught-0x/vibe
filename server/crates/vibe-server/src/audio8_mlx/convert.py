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
"""Pinned, Torch-free Audio8 FP16 converter. Full conversion needs ~18GiB.

uv run convert.py --output-dir plans/audio8_mlx/artifacts/fp16
--inspect checks metadata/mapping without loading any checkpoint tensors.
No quantization option: FP16 checkpoint token/text parity must pass first.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audio8_mlx.common import CHECKPOINT, REVISION, PACKAGE_REVISION, FORMAT_VERSION, VERSIONS, WORKSPACE, verify_dependencies, explicit_output_directory, default_hf_cache
else:
    from .common import CHECKPOINT, REVISION, PACKAGE_REVISION, FORMAT_VERSION, VERSIONS, WORKSPACE, verify_dependencies, explicit_output_directory, default_hf_cache


def source_files(cache_dir, allow_download=False):
    from huggingface_hub import hf_hub_download
    names = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")
    return {name: Path(hf_hub_download(CHECKPOINT, name, revision=REVISION,
                                     cache_dir=str(cache_dir) if cache_dir is not None else None, local_files_only=not allow_download)) for name in names}


def expected_shapes(config):
    from mlx.utils import tree_flatten
    from audio8_mlx.model import Model
    # MLX's random initialization is lazy: inspect shapes and drop the graph,
    # never evaluate full random weights before replacing them.
    model = Model(config)
    return {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}


def inspect_checkpoint(files):
    from safetensors import safe_open
    from audio8_mlx.model import map_weight, validate_config
    import mlx.core as mx
    config = json.loads(files["config.json"].read_text())
    validate_config(config)
    if config["audio_config"]["hidden_size"] != 1280 or config["text_config"]["hidden_size"] != 2048:
        raise ValueError("Cached config is not the supported pinned checkpoint")
    expected = expected_shapes(config)
    actual = {}
    dtypes = set()
    with safe_open(files["model.safetensors"], framework="np") as handle:
        for name in handle.keys():
            view = handle.get_slice(name)
            shape = view.get_shape()
            dtypes.add(view.get_dtype())
            # A lazy placeholder maps shapes only; no source tensor is read.
            target, placeholder = map_weight(name, mx.zeros(shape), config)
            if target in actual:
                raise ValueError(f"Duplicate mapped checkpoint parameter: {target}")
            actual[target] = tuple(placeholder.shape)
    if actual != expected:
        missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
        mismatched = [key for key in expected.keys() & actual.keys() if expected[key] != actual[key]]
        raise ValueError(f"Checkpoint does not match implemented MLX model: missing={missing}, extra={extra}, shapes={mismatched}")
    if not dtypes <= {"BF16", "F16", "F32"}:
        raise ValueError(f"Unsupported checkpoint dtypes: {dtypes}")
    return config, {"parameters": len(actual), "source_dtypes": sorted(dtypes), "mapping": "exact"}


def convert(files, output, shard_bytes=512 * 1024 * 1024):
    import mlx.core as mx
    from audio8_mlx.model import map_weight
    config, inspection = inspect_checkpoint(files)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Conversion output must be empty; existing models are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    source = mx.load(str(files["model.safetensors"]))
    shards, current, size = [], {}, 0

    def flush():
        nonlocal current, size
        if not current:
            return
        name = f"model-{len(shards) + 1:05d}.safetensors"
        mx.save_safetensors(str(output / name), current, metadata={"format": "mlx"})
        shards.append(name)
        current, size = {}, 0
        mx.clear_cache()

    for source_name in sorted(list(source)):
        target, value = map_weight(source_name, source.pop(source_name), config)
        value = value.astype(mx.float16)
        mx.eval(value)
        if size and size + value.nbytes > shard_bytes:
            flush()
        current[target] = value
        size += value.nbytes
    flush()
    del source
    # Copy only static tokenizer/frontend data, never HF auto_map Python code.
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        (output / name).write_bytes(files[name].read_bytes())
    config["semantic_vad_horizons_seconds"] = None  # separate unused heads are not ASR weights
    config["text_config"]["use_cache"] = True
    config["use_cache"] = True
    descriptor = {"engine": "audio8", "backend": "mlx", "checkpoint": CHECKPOINT,
                  "revision": REVISION, "package_revision": PACKAGE_REVISION,
                  "format_version": FORMAT_VERSION, "dtype": "float16", "quantization": None,
                  "versions": VERSIONS, "model_config": config, "weight_files": shards,
                  "inspection": inspection, "validation": {"status": "unvalidated"}}
    # Descriptor last: an interrupted conversion is not a loadable model.
    (output / "config.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    return descriptor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache", type=Path, default=default_hf_cache(), help="Explicit HF cache, or normal HF cache outside this checkout")
    parser.add_argument("--download", action="store_true", help="Allow downloading ONLY the pinned checkpoint/static tokenizer files")
    parser.add_argument("--inspect", action="store_true", help="Metadata-only preflight; does not load full weights or write output")
    args = parser.parse_args()
    try:
        verify_dependencies()
        output = explicit_output_directory(args.output_dir)
        files = source_files(args.hf_cache, args.download)
        if args.inspect:
            _, inspection = inspect_checkpoint(files)
            print(json.dumps(inspection))
        else:
            print(json.dumps({"stage": "mlx-fp16-conversion", "pid": os.getpid(), "output_dir": str(output)}), flush=True)
            descriptor = convert(files, output)
            print(json.dumps({"output_dir": str(output), "weight_files": descriptor["weight_files"],
                              "validation": descriptor["validation"]}))
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Audio8 MLX conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
