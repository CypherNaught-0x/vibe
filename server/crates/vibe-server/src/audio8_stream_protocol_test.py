# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy>=2.1,<3"]
# ///
"""Protocol tests never load/download a model."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

worker = load("audio8_runtime")
protocol = load("audio8_stream_protocol")

class Decoder:
    def __init__(self, runtime, language, **options):
        self.options = options
        self.raw = bytearray()
        runtime.last_decoder = self
    def push_pcm(self, raw):
        self.raw.extend(raw)
        return [{"type":"partial", "text":"hello", "segment_index":0}]
    def finish(self):
        return [{"type":"segment", "start":0, "end":1, "text":"hello"},
            {"type":"result", "segments":1}]

class Tests(unittest.TestCase):
    def setUp(self):
        self.events, self.calls = [], []
        self.runtime = SimpleNamespace(np=np, set_threads=lambda n: self.calls.append(n))
        self.module = SimpleNamespace(StreamingDecoder=Decoder,
            calibrate=lambda runtime: {"window_tokens":32,"pcm_capacity_samples":15360,
                "step_seconds":0.02,"realtime":True,"device":"fake"})
        self.live = protocol.LiveProtocol(self.runtime, self.module, self.events.append)
    def start(self, threads=4):
        return self.live.dispatch({"type":"stream_start","language":"en","threads":threads}, b"")
    def test_readiness_is_gated_and_ring_parameters_are_forwarded(self):
        self.assertTrue(self.start())
        self.assertEqual([e["type"] for e in self.events], ["calibration","stream_ready"])
        self.assertEqual(self.runtime.last_decoder.options, {"window_tokens":32,"pcm_capacity_samples":15360})
    def test_slow_devices_never_construct_a_capture_decoder(self):
        self.module.calibrate = lambda _: {"realtime":False}
        self.assertFalse(self.start())
        self.assertIsNone(self.live.decoder)
        self.assertEqual(self.events[-1]["type"], "error")
    def test_frames_and_finish_reset_session(self):
        self.start()
        self.live.dispatch({"type":"stream_chunk"}, b"\0"*4)
        self.live.dispatch({"type":"stream_end"}, b"")
        self.assertIsNone(self.live.decoder)
        self.assertEqual(bytes(self.runtime.last_decoder.raw), b"\0"*4)
        self.assertEqual(self.events[-1], {"type":"result","segments":1})
    def test_duplicate_start_or_unstarted_data_rejected(self):
        with self.assertRaises(ValueError): self.live.dispatch({"type":"stream_chunk"}, b"\0"*4)
        self.start()
        with self.assertRaises(ValueError): self.start()
    def test_calibration_reused_only_with_same_thread_count(self):
        measured=[]
        original=self.module.calibrate
        self.module.calibrate=lambda r: measured.append(1) or original(r)
        for threads in (4,4,2):
            self.start(threads)
            self.live.dispatch({"type":"stream_end"}, b"")
        self.assertEqual(len(measured), 2)
    def test_stream_frame_validation_and_exact_payload(self):
        for count in (1,2560):
            data = json.dumps({"type":"stream_chunk","sample_count":count}).encode()+b"\n"+b"\0"*(4*count)
            _,raw=worker.read_request(io.BytesIO(data))
            self.assertEqual(len(raw), count*4)
        for count in (-1,0,2561,True):
            data=json.dumps({"type":"stream_chunk","sample_count":count}).encode()+b"\n"
            with self.assertRaises(worker.ProtocolError): worker.read_request(io.BytesIO(data))
    def test_control_frames_forbid_pcm(self):
        for kind in ("stream_start","stream_end"):
            data=json.dumps({"type":kind,"sample_count":1,"language":"en"}).encode()+b"\n"
            with self.assertRaises(worker.ProtocolError): worker.read_request(io.BytesIO(data))
    def test_startup_error_does_not_echo_signed_urls_or_credentials(self):
        output=io.StringIO()
        with patch.object(worker,"reserve_protocol_stdout",return_value=output), \
                patch.object(worker,"validate_manifest"), \
                patch.object(worker,"Runtime",side_effect=RuntimeError("synthetic_signed_url_secret")), \
                patch.object(worker.sys,"argv",["worker","--manifest","synthetic.audio8.json"]):
            self.assertEqual(worker.main(),1)
        self.assertNotIn("synthetic_signed_url_secret",output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["type"],"error")
    def test_serve_dispatches_live_frames_without_torch(self):
        source=(json.dumps({"type":"stream_start","language":"en"})+"\n"+
            json.dumps({"type":"stream_chunk","sample_count":1})+"\n").encode()+b"\0"*4+b'{"type":"stream_end"}\n'
        old=worker.local_module
        worker.local_module=lambda name: protocol if name=="audio8_stream_protocol" else self.module
        try:
            self.assertEqual(worker.serve(self.runtime,io.BytesIO(source),self.events.append),0)
        finally: worker.local_module=old
        self.assertEqual(self.events[-1]["type"],"result")

if __name__=="__main__": unittest.main(verbosity=2)
