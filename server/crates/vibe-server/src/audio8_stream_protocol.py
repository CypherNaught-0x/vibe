"""Bounded PCM streaming protocol, independent of Torch model construction.

The host sends stream_start, then stream_chunk JSON + <=2560 f32 samples,
then stream_end. Startup emits calibration before stream_ready; capture must
not begin unless realtime=true. Model exceptions never disclose private input.
"""
from __future__ import annotations

MAX_CHUNK_SAMPLES = 2560


class LiveProtocol:
    def __init__(self, runtime, module, emit):
        self.runtime, self.module, self.emit = runtime, module, emit
        self.decoder = None
        self.calibration = None
        self.threads = None

    def dispatch(self, request, raw):
        kind = request["type"]
        if kind == "stream_start":
            if self.decoder is not None:
                raise ValueError("Audio8 live session already started")
            threads = request.get("threads", 0)
            self.runtime.set_threads(threads)
            if self.calibration is None or self.threads != threads:
                self.calibration = self.module.calibrate(self.runtime)
                self.threads = threads
            self.emit({**self.calibration, "type": "calibration"})
            if not self.calibration["realtime"]:
                self.emit({"type": "error", "message":
                    "Audio8 live capture is unavailable: measured inference cannot keep up with real-time audio. Use file transcription or a faster backend/device."})
                return False
            self.decoder = self.module.StreamingDecoder(self.runtime, request["language"],
                window_tokens=self.calibration["window_tokens"],
                pcm_capacity_samples=self.calibration["pcm_capacity_samples"])
            self.emit({"type": "stream_ready"})
        elif kind == "stream_chunk":
            if self.decoder is None:
                raise ValueError("Audio8 live session has not started")
            for event in self.decoder.push_pcm(raw):
                self.emit(event)
        elif kind == "stream_end":
            if self.decoder is None:
                raise ValueError("Audio8 live session has not started")
            for event in self.decoder.finish():
                self.emit(event)
            self.decoder = None
        else:
            raise ValueError("Unexpected Audio8 live request")
        return True
