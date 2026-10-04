"""Platform/descriptor contract, deliberately importable without MLX or Torch."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import sys

CHECKPOINT = "Edge0/Audio8-ASR-Infinite"
REVISION = "7476824bc222e4ad509d286e8cae8b8d3f371129"
PACKAGE_REVISION = "c8ba8eea829be0339e8d7757f8ca52dac06e1e32"
FORMAT_VERSION = 1
VERSIONS = {"mlx": "0.32.3", "mlx-lm": "0.31.3", "mlx-audio": "0.4.4", "transformers": "5.5.3"}
# Only validation defaults use this checkout-relative location. Installed
# runtimes may live at a shallower path; importing the backend must stay safe.
_SOURCE_PARENTS = Path(__file__).resolve().parents
WORKSPACE = _SOURCE_PARENTS[min(5, len(_SOURCE_PARENTS) - 1)]
MARKER_NAME = "Audio8-ASR-Infinite.audio8.json"


def require_apple_silicon():
    # Gate BEFORE importing MLX: Linux/Intel errors must be understandable.
    if sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Audio8 MLX requires macOS on Apple Silicon; use the Torch backend on this platform")


def verify_dependencies():
    require_apple_silicon()
    for name, expected in VERSIONS.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Audio8 MLX requires {name}=={expected}, found {actual}")


def workspace_output(path):
    result = Path(path).expanduser().resolve()
    if not result.is_relative_to(WORKSPACE) or result == WORKSPACE:
        raise ValueError(f"Conversion output must be an explicit directory inside {WORKSPACE}")
    return result


def explicit_output_directory(path):
    """User-selected portable output; unlike validation artifacts, not repo-bound."""
    if not str(path).strip():
        raise ValueError("Conversion requires an explicit non-empty output directory")
    result = Path(path).expanduser().resolve()
    if result == result.parent or (result.exists() and not result.is_dir()):
        raise ValueError("Conversion output must be a non-root directory")
    return result


def default_hf_cache():
    cached = WORKSPACE / "plans/audio8/artifacts/hf/hub"
    return cached if cached.is_dir() else None  # fall back to normal HF cache when installed elsewhere


def write_validated_marker(directory):
    """Publish the strict Rust-facing stub only AFTER pretrained FP16 parity."""
    directory = Path(directory).resolve()
    descriptor = read_descriptor(directory)
    validation = descriptor.get("validation", {})
    if (validation.get("status") != "fp16-parity-passed" or
            validation.get("token_parity") is not True or validation.get("text_parity") is not True):
        raise ValueError("Cannot write Audio8 MLX marker before full FP16 token/text parity passes")
    marker = {"engine": "audio8", "checkpoint": CHECKPOINT, "revision": REVISION,
              "backend": "mlx", "artifacts": "."}
    path = directory / MARKER_NAME
    if path.exists() and json.loads(path.read_text()) != marker:
        raise ValueError("Refusing to overwrite a different Audio8 model marker")
    path.write_text(json.dumps(marker, indent=2) + "\n")
    return path


def read_descriptor(directory):
    directory = Path(directory).resolve()
    path = directory / "config.json"
    if path.stat().st_size > 65536:
        raise ValueError("Audio8 MLX descriptor is too large")
    descriptor = json.loads(path.read_text())
    required = {"engine": "audio8", "backend": "mlx", "checkpoint": CHECKPOINT,
                "revision": REVISION, "package_revision": PACKAGE_REVISION,
                "format_version": FORMAT_VERSION, "dtype": "float16", "quantization": None}
    for key, value in required.items():
        if descriptor.get(key) != value:
            raise ValueError(f"Unsupported Audio8 MLX descriptor field: {key}")
    if descriptor.get("versions") != VERSIONS:
        raise ValueError("Unsupported Audio8 MLX package versions")
    names = descriptor.get("weight_files")
    if not isinstance(names, list) or not names:
        raise ValueError("Audio8 MLX descriptor has no weight files")
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid Audio8 MLX weight filename")
        file = directory / name
        if not file.is_file() or not file.resolve().is_relative_to(directory):
            raise ValueError("Missing or external Audio8 MLX weight file")
    return descriptor
