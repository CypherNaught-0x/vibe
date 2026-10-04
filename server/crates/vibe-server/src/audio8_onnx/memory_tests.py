# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.4.4", "onnxruntime==1.23.2", "onnx==1.22.0"]
# ///
"""Lightweight ORT initializer ownership tests; never reads checkpoint weights."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as H
import onnxruntime as ort

from backend import MappedInitializers, external_initializers, session_options


def test_api(root):
    weights = np.arange(16, dtype=np.float32).reshape(4, 4)
    binary = root / "weights.bin"
    binary.write_bytes(weights.tobytes())
    tensor = T(name="W", data_type=T.FLOAT, dims=[4, 4], data_location=T.EXTERNAL)
    for key, value in (("location", "weights.bin"), ("offset", "0"), ("length", "64")):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    graph = H.make_graph([H.make_node("MatMul", ["X", "W"], ["Y"])], "ownership",
                         [H.make_tensor_value_info("X", T.FLOAT, ["batch", 4])],
                         [H.make_tensor_value_info("Y", T.FLOAT, ["batch", 4]),
                          H.make_tensor_value_info("W", T.FLOAT, [4, 4])], [tensor])
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 18)])
    model.ir_version = 10
    graph_path = root / "ownership.onnx"
    onnx.save_model(model, graph_path)
    onnx.checker.check_model(str(graph_path))
    x = np.arange(4, dtype=np.float32)[None]
    report = {"onnxruntime": ort.__version__}
    for mode in ("normal", "external", "initializer"):
        mapped = np.memmap(binary, dtype="<f4", mode="r+", shape=(4, 4))
        mapped[:] = weights
        mapped.flush()
        value = ort.OrtValue.ortvalue_from_numpy(mapped, "cpu")
        assert value.data_ptr() == mapped.ctypes.data, "CPU OrtValue unexpectedly copied data"
        opts = session_options(1)
        if mode == "external":
            opts.add_external_initializers(["W"], [value])
        elif mode == "initializer":
            opts.add_initializer("W", value)
        try:
            session = ort.InferenceSession(str(graph_path), opts, providers=["CPUExecutionProvider"])
            before = session.run(None, {"X": x})
            np.testing.assert_array_equal(before[0], x @ weights)
            mapped[:] = weights + 100
            mapped.flush()
            gc.collect()
            after = session.run(None, {"X": x})
            zero_copy = bool(np.array_equal(after[0], x @ (weights + 100)))
            report[mode] = {"ortvalue_pointer_equal": True, "live_mutation_visible": zero_copy,
                            "initializer_output_changed": bool(np.array_equal(after[1], weights + 100))}
            del session
        except Exception as error:
            report[mode] = {"error": type(error).__name__ + ": " + str(error)}
        del value, mapped, opts
    (root / "api_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def test_mapped_loader(root):
    import weakref
    binary, graph_path = root / "weights.bin", root / "ownership.onnx"
    weights = np.arange(16, dtype=np.float32).reshape(4, 4)
    binary.write_bytes(weights.tobytes())
    graph_bytes = graph_path.read_bytes()
    assert external_initializers(graph_bytes) == {"W": ((4, 4), 0, 64)}
    owner = MappedInitializers(binary)
    session = owner.session(graph_bytes, 1)
    assert owner.values["W"].data_ptr() == owner.views["W"].ctypes.data
    assert not owner.views["W"].flags.writeable
    assert np.shares_memory(owner.views["W"], owner.mapping)
    x = np.arange(4, dtype=np.float32)[None]
    np.testing.assert_array_equal(session.run(["Y"], {"X": x})[0], x @ weights)
    # Runtime's mapping is read-only; separate fixture-only writer proves aliasing.
    writer = np.memmap(binary, dtype="<f4", mode="r+", shape=(4, 4))
    writer[:] = weights + 100
    writer.flush()
    np.testing.assert_array_equal(session.run(["Y"], {"X": x})[0], x @ (weights + 100))
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is session._audio8_initializer_owner
    np.testing.assert_array_equal(session.run(["Y"], {"X": x})[0], x @ (weights + 100))
    del session
    gc.collect()
    assert owner_ref() is None, "Initializer owner leaked after last session released"
    del writer
    negative = []
    for change in ("dtype", "location", "offset", "length", "shape", "duplicate"):
        model = onnx.load_model_from_string(graph_bytes)
        tensor = model.graph.initializer[0]
        if change == "dtype":
            tensor.data_type = T.DOUBLE
        elif change == "shape":
            tensor.dims[0] = 9
        elif change == "duplicate":
            model.graph.initializer.append(tensor)
        else:
            value = {"location": "../outside.bin", "offset": "2", "length": "65"}[change]
            next(v for v in tensor.external_data if v.key == change).value = value
        try:
            MappedInitializers(binary).add(session_options(1), model.SerializeToString())
        except ValueError:
            negative.append(change)
        else:
            raise AssertionError(f"Unsafe initializer metadata accepted: {change}")
    for data in (graph_bytes[:-1], b"\x3a\xff", b"\x00"):
        try:
            external_initializers(data)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed ONNX metadata accepted")
    report = {"read_only_zero_copy": True, "lifetime_retained_until_session_release": True,
              "malformed_metadata_rejected": True, "negative_metadata_cases": negative,
              "graph_bytes_loading_without_external_file_resolution": True}
    (root / "loader_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def test_transpose_allocations(root, descriptor=None):
    # 2MB synthetic weight, not checkpoint data. Profile explicit tensor outputs.
    weight = np.arange(512 * 1024, dtype=np.float32).reshape(512, 1024) / 1000
    folder = root / "transpose"
    folder.mkdir()
    (folder / "weights.bin").write_bytes(weight.tobytes())
    tensor = T(name="W", data_type=T.FLOAT, dims=weight.shape, data_location=T.EXTERNAL)
    for key, value in (("location", "weights.bin"), ("offset", "0"), ("length", str(weight.nbytes))):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    x = np.arange(1024, dtype=np.float32)[None] / 1024
    outputs, report = {}, {}
    for mode in ("transpose_matmul", "gemm"):
        nodes = ([H.make_node("Transpose", ["W"], ["WT"], perm=[1, 0]), H.make_node("MatMul", ["X", "WT"], ["Y"])]
                 if mode == "transpose_matmul" else [H.make_node("Gemm", ["X", "W"], ["Y"], transB=1)])
        graph = H.make_graph(nodes, mode, [H.make_tensor_value_info("X", T.FLOAT, ["batch", 1024])],
                             [H.make_tensor_value_info("Y", T.FLOAT, ["batch", 512])], [tensor])
        model = H.make_model(graph, opset_imports=[H.make_opsetid("", 18)])
        model.ir_version = 10
        owner = MappedInitializers(folder / "weights.bin")
        options = session_options(1)
        options.enable_profiling = True
        options.profile_file_prefix = str(folder / mode)
        owner.add(options, model.SerializeToString())
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        session._audio8_initializer_owner = owner
        outputs[mode] = session.run(None, {"X": x})[0]
        profile = json.loads(Path(session.end_profiling()).read_text())
        report[mode] = [{"op": v["args"]["op_name"], "output_size": v["args"].get("output_size")}
                        for v in profile if v.get("cat") == "Node" and "op_name" in v.get("args", {})]
    np.testing.assert_allclose(outputs["transpose_matmul"], outputs["gemm"], atol=0.002, rtol=2e-6)
    assert any(v["op"] == "Transpose" and int(v["output_size"]) == weight.nbytes
               for v in report["transpose_matmul"])
    assert all(v["op"] != "Transpose" for v in report["gemm"])
    if descriptor:
        d = json.loads(descriptor.read_text())
        byte_count, nodes = 0, 0
        for g in d["graphs"].values():
            # Explicitly metadata-only: ONNX must not read external tensor data.
            model = onnx.load(descriptor.parent / g["file"], load_external_data=False)
            initializers = {v.name: v for v in model.graph.initializer}
            for n in model.graph.node:
                if n.op_type == "Transpose" and n.input[0] in initializers:
                    dims = initializers[n.input[0]].dims
                    byte_count += int(np.prod(dims)) * 4
                    nodes += 1
        report["existing_export_metadata_only"] = {"constant_weight_transpose_nodes": nodes,
                                                   "cumulative_output_bytes_per_step": byte_count,
                                                   "not_a_peak_memory_estimate": True}
    (root / "transpose_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-descriptor", type=Path, help="read graph metadata only, never mapped full weights")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    test_api(args.output)
    test_mapped_loader(args.output)
    test_transpose_allocations(args.output, args.inspect_descriptor)


if __name__ == "__main__":
    main()
