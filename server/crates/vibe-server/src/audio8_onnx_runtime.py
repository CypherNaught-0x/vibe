# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.4.4", "onnxruntime==1.23.2", "tokenizers==0.22.2",
# ]
# ///
"""Pinned ONNX worker; Torch is not a runtime dependency."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio8_portable_runtime import main
if __name__ == "__main__":
    raise SystemExit(main("onnx"))
