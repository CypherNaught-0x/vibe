# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Strict converted marker tests; never import inference backends."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import sys
import audio8_portable_runtime as runtime

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root=Path(self.temp.name)
        self.marker=self.root/"model.audio8.json"
        self.data={"engine":"audio8","checkpoint":runtime.protocol.CHECKPOINT,
            "revision":runtime.protocol.REVISION,"backend":"onnx","artifacts":"."}
    def tearDown(self): self.temp.cleanup()
    def check(self, data=None, backend="onnx"):
        self.marker.write_text(json.dumps(self.data if data is None else data))
        return runtime.converted_directory(self.marker,backend)
    def test_accepts_only_known_pinned_backend_fields(self):
        self.assertEqual(self.check(),self.root.resolve())
        for changes in ({"backend":"other"},{"revision":"main"},{"engine":"whisper"},
                        {"checkpoint":"private/unknown"},{"script":"bad.py"},{"trust_remote_code":True}):
            with self.assertRaises(ValueError): self.check({**self.data,**changes})
    def test_requires_artifacts_without_traversal(self):
        for relative in ("..","../other",str(self.root.resolve()),"missing"):
            with self.assertRaises(ValueError): self.check({**self.data,"artifacts":relative})
    def test_symlink_cannot_escape_marker_directory(self):
        outside=self.root.parent
        (self.root/"outside").symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError): self.check({**self.data,"artifacts":"outside"})
    def test_backend_cannot_cross_select_another_worker(self):
        with self.assertRaises(ValueError): self.check(backend="mlx")
        self.assertEqual(self.check({**self.data,"backend":"mlx"},backend="mlx"),self.root.resolve())
    def test_marker_byte_limit(self):
        self.marker.write_text(" "*4097)
        with self.assertRaises(ValueError): runtime.converted_directory(self.marker,"onnx")
    def test_both_builtin_workers_reach_constructor_and_serve_without_torch(self):
        import io
        for backend, name in (("onnx","OnnxRuntime"),("mlx","MlxRuntime")):
            with self.subTest(backend=backend):
                self.check({**self.data,"backend":backend},backend=backend)
                output=io.StringIO()
                fake=SimpleNamespace(device="fake",dtype="float16")
                constructor=Mock(return_value=fake)
                module=SimpleNamespace(**{name:constructor})
                with patch("sys.argv",["worker","--manifest",str(self.marker),"--gpu-device","-1"]), \
                        patch.dict(sys.modules,{f"audio8_{backend}.backend":module}), \
                        patch.object(runtime.protocol,"reserve_protocol_stdout",return_value=output), \
                        patch.object(runtime.protocol,"serve",return_value=0) as serve:
                    self.assertEqual(runtime.main(backend),0)
                    constructor.assert_called_once()
                    self.assertEqual(constructor.call_args.kwargs["gpu_device"],0)
                    serve.assert_called_once()
                ready=json.loads(output.getvalue())
                self.assertEqual(ready["type"],"ready")
                self.assertEqual(ready["backend"],backend)
    def test_mlx_cpu_request_has_clear_error_before_constructor(self):
        import io
        self.check({**self.data,"backend":"mlx"},backend="mlx")
        output=io.StringIO()
        with patch("sys.argv",["worker","--manifest",str(self.marker),"--no-gpu"]), \
                patch.object(runtime.protocol,"reserve_protocol_stdout",return_value=output):
            self.assertEqual(runtime.main("mlx"),1)
        error=json.loads(output.getvalue())
        self.assertEqual(error["type"],"error")
        self.assertIn("Disable Transcribe on CPU",error["message"])
    def test_startup_failure_does_not_expose_paths_or_private_values(self):
        import io
        output=io.StringIO()
        with patch("sys.argv",["worker","--manifest",str(self.marker),"--cache-dir",str(self.root)]), \
                patch.object(runtime.protocol,"reserve_protocol_stdout",return_value=output), patch.object(runtime,"converted_directory",side_effect=RuntimeError("secret_private_text")):
            self.assertEqual(runtime.main("onnx"),1)
        self.assertNotIn("secret_private_text",output.getvalue())
        self.assertNotIn(str(self.root),output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["type"],"error")

if __name__=="__main__": unittest.main(verbosity=2)
