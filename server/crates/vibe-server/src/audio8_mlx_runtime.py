# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy>=2.1,<3", "mlx==0.32.3", "mlx-lm==0.31.3", "mlx-audio==0.4.4",
#   "transformers==5.5.3", "safetensors>=0.8,<1", "soundfile>=0.13,<1",
# ]
# ///
"""Pinned Apple Silicon worker; Torch is not a runtime dependency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio8_portable_runtime import main
if __name__ == "__main__":
    raise SystemExit(main("mlx"))
