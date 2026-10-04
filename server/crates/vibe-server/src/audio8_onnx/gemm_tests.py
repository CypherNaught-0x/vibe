# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.4.4", "onnxruntime==1.23.2", "onnx==1.22.0"]
# ///
"""Light Gemm linear oracle for bias/no-bias, rank2/rank3, dynamic dimensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, numpy_helper

from backend import session_options
from graphs import Graph


class FixtureWeights:
    def __init__(self, values):
        self.values, self.names = values, set(values)

    def initializer(self, name):
        return numpy_helper.from_array(self.values[name], name)


def check(root):
    rng = np.random.default_rng(750)
    report = {}
    for rank in (2, 3):
        for bias in (False, True):
            name = f"linear_rank{rank}_bias{int(bias)}"
            values = {"linear.weight": rng.normal(0, .2, (5, 9)).astype(np.float32)}
            if bias:
                values["linear.bias"] = rng.normal(0, .2, 5).astype(np.float32)
            graph = Graph(name, FixtureWeights(values))
            shape = ["batch", 9] if rank == 2 else ["batch", "tokens", 9]
            x = graph.input("x", shape)
            graph.op("Identity", graph.linear(x, "linear"), output="y")
            graph.output("y", shape[:-1] + [5])
            metadata = graph.save(root)
            model = onnx.load(root / metadata["file"])
            assert sum(n.op_type == "Gemm" for n in model.graph.node) == 1
            assert all(n.op_type != "Transpose" for n in model.graph.node)
            session = ort.InferenceSession(str(root / metadata["file"]), session_options(1), providers=["CPUExecutionProvider"])
            errors = []
            for batch, count in ((1, 1), (3, 2), (2, 7), (5, 3)):
                x_shape = (batch, 9) if rank == 2 else (batch, count, 9)
                data = rng.normal(0, .2, x_shape).astype(np.float32)
                expected = data @ values["linear.weight"].T
                if bias:
                    expected += values["linear.bias"]
                actual = session.run(["y"], {"x": data})[0]
                assert actual.shape == x_shape[:-1] + (5,)
                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
                errors.append(float(np.max(np.abs(actual - expected))))
            report[name] = {"dynamic_cases": 4, "maximum_absolute_error": max(errors), "passed": True}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": 16, "report": str(root / "report.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    check(args.output)


if __name__ == "__main__":
    main()
