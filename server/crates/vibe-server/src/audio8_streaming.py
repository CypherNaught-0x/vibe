"""Bounded, stateful Audio8 decoding; no waveform replay or chunk resets.

Torch is imported only for a Torch runtime. Portable runtimes supply
new_step_decoder(window_tokens), returning step(audio_window, ids) -> int and
cache_stats() -> {cache_bytes, audio_frames, text_tokens, audio_position,
text_position}. Physical lengths and monotonically advancing absolute positions
are checked. Adapters own frontend/inference, cache cropping and EOS suppression;
the shared controller owns prefix/feedback IDs, PCM, UTF-8 and the drain clock.
step must accept multi-ID synthetic calibration fills without reloading weights.
An optional audio_window_frames property declares a smaller physical audio cap;
otherwise the pinned min(8 * window_tokens, 750) cap is assumed.


The frontend/clock follows the pinned simulated_streaming_greedy_decode_batch:
https://github.com/Edge0-AI/Audio8-ASR-Infinite/blob/c8ba8eea829be0339e8d7757f8ca52dac06e1e32/audio8_asr_infinite/simulated_streaming_audio.py
Rolling prefix retention follows the pinned vLLM controller, but retained keys
keep their original RoPE positions (we do not rebase or rerotate them):
https://github.com/Edge0-AI/Audio8-ASR-Infinite/blob/c8ba8eea829be0339e8d7757f8ca52dac06e1e32/audio8_asr_infinite/vllm_serve/vllm0271/audio8_rolling_scheduler.py

Partial text is cumulative *within segment_index*, not over the entire session.
Consumers concatenate committed segments and display only the current partial.
Segment bounds are emission-clock bounds, NOT word alignment or semantic VAD.
Rolling context necessarily differs from unbounded full-attention decoding.
Calls are synchronous: the host must apply producer backpressure and split PCM
into <= pcm_capacity_samples - buffered_samples packets (2560 is recommended).
Overflow rejects the whole push, never
silently dropping PCM. An inference failure poisons the session; do not retry it.
"""
from __future__ import annotations

import codecs
import math
import threading
import time

SAMPLE_RATE = 16000
SAMPLES_PER_TOKEN = 2560
FRAME_LEN = 8
DELAY_TOKENS = 3
LEFT_PAD_TOKENS = 9
PREFILL_TOKENS = LEFT_PAD_TOKENS + DELAY_TOKENS + 1
RIGHT_PAD_TEXT_TOKENS = 10
LOOK_BACK_SAMPLES = 840  # 52.5ms, including the convolution/STFT overlap
LOOK_AHEAD_SAMPLES = 40  # 2.5ms
MIN_PCM_CAPACITY_SAMPLES = 5 * SAMPLES_PER_TOKEN  # first 4 tokens + STFT lookahead
MAX_PCM_CAPACITY_SAMPLES = 32 * SAMPLES_PER_TOKEN  # hard 5.12s admission bound
PCM_CAPACITY_SAMPLES = 16 * SAMPLES_PER_TOKEN  # uncalibrated/test default only
COMMIT_STEPS = 32
MAX_WINDOW_TOKENS = 256  # finite production/calibration ceiling, not infinite attention


def _byte_decoder():
    # GPT-2/Qwen ByteLevel's reversible byte-to-Unicode alphabet. Decoding bytes
    # incrementally, instead of decoding individual IDs with errors='replace',
    # preserves characters split across two or more BPE tokens/segment commits.
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(values, chars)}


BYTE_DECODER = _byte_decoder()


def crop_recent(cache, max_length: int, prefix: int = 0):
    """Keep a stable prefix + newest tail, freeing the removed tensor storage.

    HF DynamicCache.crop keeps the *first* entries, so it is unsuitable here.
    Only plain dynamic layers are accepted: sliding layers have separate
    cumulative lengths and offsets, which must not masquerade as cached length.
    """
    if cache is None:
        return
    if max_length < 0 or prefix < 0 or prefix > max_length:
        raise ValueError("Invalid rolling cache bound")
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("Audio8 requires Transformers dynamic cache layers")
    for layer in layers:
        if getattr(layer, "is_sliding", False) or getattr(layer, "is_compileable", False):
            raise TypeError("Audio8 rolling cache requires plain dynamic layers")
        keys, values = layer.keys, layer.values
        if keys is None:
            continue
        length = int(keys.shape[-2])
        if length <= max_length:
            continue
        tail = max_length - prefix
        if prefix:
            # clone/cat owns a fresh bounded backing allocation, not a view of
            # a previously much larger cache. Tail=0 must not slice [-0:].
            import torch
            layer.keys = torch.cat((keys[..., :prefix, :], keys[..., -tail:, :]), dim=-2) if tail else keys[..., :prefix, :].clone()
            layer.values = torch.cat((values[..., :prefix, :], values[..., -tail:, :]), dim=-2) if tail else values[..., :prefix, :].clone()
        else:
            layer.keys = keys[..., -tail:, :].clone() if tail else keys[..., :0, :].clone()
            layer.values = values[..., -tail:, :].clone() if tail else values[..., :0, :].clone()


def _cache_length(cache):
    return int(cache.get_seq_length())


def _cache_bytes(cache):
    return sum(t.numel() * t.element_size() for layer in cache.layers
               for t in (layer.keys, layer.values) if t is not None)


class _PcmRing:
    """Fixed allocation, absolute sample indexes; never overwrites live input."""
    def __init__(self, np, capacity):
        self.np, self.capacity = np, capacity
        self.samples = np.empty(capacity, dtype=np.float32)
        self.start = self.end = 0

    def append(self, samples):
        count = len(samples)
        if self.end - self.start + count > self.capacity:
            raise BufferError("Audio8 PCM capacity exceeded; split frames/apply backpressure")
        offset = self.end % self.capacity
        first = min(count, self.capacity - offset)
        self.samples[offset:offset + first] = samples[:first]
        self.samples[:count - first] = samples[first:]
        self.end += count

    def window(self, start, end):
        # Negative offsets are the virtual left pad; offsets beyond accepted
        # PCM are used only by finish's virtual right silence.
        result = self.np.zeros(end - start, dtype=self.np.float32)
        lo, hi = max(0, start), min(self.end, end)
        if lo < hi:
            if lo < self.start:
                raise RuntimeError("Audio8 frontend requested already-consumed PCM")
            offset = lo % self.capacity
            first = min(hi - lo, self.capacity - offset)
            dest = lo - start
            result[dest:dest + first] = self.samples[offset:offset + first]
            result[dest + first:hi - start] = self.samples[:hi - lo - first]
        return result

    def discard_before(self, index):
        self.start = min(self.end, max(self.start, index))


class StreamingDecoder:
    def __init__(self, runtime, language, window_tokens=128, pcm_capacity_samples=None):
        if pcm_capacity_samples is None:
            pcm_capacity_samples = getattr(runtime, "streaming_pcm_capacity_samples", PCM_CAPACITY_SAMPLES)
        if (type(pcm_capacity_samples) is not int or
                not MIN_PCM_CAPACITY_SAMPLES <= pcm_capacity_samples <= MAX_PCM_CAPACITY_SAMPLES or
                pcm_capacity_samples % SAMPLES_PER_TOKEN):
            raise ValueError("Audio8 PCM capacity must be a bounded 160ms frame multiple")
        if language not in ("en", "zh") or language not in runtime.language_ids:
            raise ValueError("Audio8 requires language en or zh")
        if type(window_tokens) is not int or not PREFILL_TOKENS + 1 <= window_tokens <= MAX_WINDOW_TOKENS:
            raise ValueError(f"window_tokens must be {PREFILL_TOKENS + 1}..{MAX_WINDOW_TOKENS}")
        self.runtime, self.language, self.window_tokens = runtime, language, window_tokens
        self._backend_step = None
        self.language_cache = self.audio_cache = None
        factory = getattr(runtime, "new_step_decoder", None)
        if factory is not None:
            if not callable(factory):
                raise TypeError("Audio8 new_step_decoder must be callable")
            self._backend_step = factory(window_tokens)
            if not callable(getattr(self._backend_step, "step", None)) or not callable(getattr(self._backend_step, "cache_stats", None)):
                raise TypeError("Audio8 portable adapter requires step and cache_stats")
            cap = getattr(self._backend_step, "audio_window_frames", min(window_tokens * FRAME_LEN, 750))
            self.audio_window_frames = int(cap)
            if not 1 <= self.audio_window_frames <= window_tokens * FRAME_LEN:
                raise ValueError("Audio8 adapter audio cache cap exceeds its bounded window")
        else:
            from transformers import DynamicCache
            from audio8_asr_infinite.simulated_streaming_audio import (
                _extract_streaming_features_batch, _group_streaming_audio_hidden_states,
                select_current_20ms_frames,
            )
            self._features = _extract_streaming_features_batch
            self._group = _group_streaming_audio_hidden_states
            self._select = select_current_20ms_frames
            # No automatic sliding layers: explicit physical masks + positions.
            self.language_cache, self.audio_cache = DynamicCache(), DynamicCache()
            sliding = getattr(runtime.model.config.audio_config, "sliding_window", None)
            self.audio_window_frames = min(window_tokens * FRAME_LEN, int(sliding or window_tokens * FRAME_LEN))
            self._audio_sliding = sliding
        self.pcm_capacity_samples = pcm_capacity_samples
        self._pcm = _PcmRing(runtime.np, self.pcm_capacity_samples)
        self.absolute_text_position = self.absolute_audio_position = 0
        self._clock_start = 0
        self._clock_end = PREFILL_TOKENS * SAMPLES_PER_TOKEN
        self._next_token = None
        self._condition = None
        self._utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self._skip_ids = set(runtime.special_ids.values()) | set(getattr(runtime.tokenizer, "all_special_ids", ()))
        self._text = ""
        self._segment_index = self._committed_chars = self._segment_start = 0
        self._steps = self._segment_steps = self.segment_count = 0
        self._finished = self._failed = False
        self._lock = threading.Lock()
        # Fail closed if a different tokenizer is supplied; this byte decoding
        # contract is exact for the pinned Qwen tokenizer, not arbitrary SP/BPE.
        backend = getattr(runtime.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            import json
            decoder = json.loads(backend.to_str()).get("decoder", {})
            if decoder.get("type") != "ByteLevel":
                raise ValueError("Audio8 streaming requires the pinned ByteLevel tokenizer")

    @property
    def buffered_samples(self):
        return self._pcm.end - self._pcm.start

    @property
    def accepted_samples(self):
        return self._pcm.end

    def _enter(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent Audio8 calls require host backpressure")
        if self._failed:
            self._lock.release()
            raise RuntimeError("Audio8 session failed; create a new decoder")

    def _mask(self, query, cache, absolute, prefix=0, sliding=None):
        torch, device = self.runtime.torch, self.runtime.device
        length = _cache_length(cache)
        head = min(prefix, length)
        tail = length - head
        # Physical cache indexes and RoPE positions are different after crop.
        # The prefix stays at its original positions; the tail is contiguous.
        keys = torch.cat((torch.arange(head, device=device),
                          torch.arange(absolute - tail, absolute + query, device=device)))
        queries = torch.arange(absolute, absolute + query, device=device)
        allowed = keys.unsqueeze(0) <= queries.unsqueeze(1)
        if sliding:
            allowed &= keys.unsqueeze(0) > queries.unsqueeze(1) - int(sliding)
        mask = torch.zeros((query, length + query), device=device, dtype=self.runtime.dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def cache_stats(self):
        """Physical cache sizes, independent of absolute RoPE positions."""
        if self._backend_step is not None:
            raw = self._backend_step.cache_stats()
            fields = {
                "cache_bytes": ("cache_bytes", "total_cache_bytes", "kv_bytes"),
                "audio_frames": ("audio_frames", "audio_cache_frames", "cached_audio_frames"),
                "text_tokens": ("text_tokens", "text_cache_tokens", "cached_text_tokens"),
                "audio_position": ("audio_position", "absolute_audio_position", "audio_pos"),
                "text_position": ("text_position", "absolute_text_position", "text_pos"),
            }
            stats = {}
            for name, aliases in fields.items():
                value = next((raw[key] for key in aliases if key in raw), None)
                if value is None or isinstance(value, bool) or int(value) != value or value < 0:
                    raise ValueError(f"Audio8 adapter requires nonnegative integer {name} statistics")
                stats[name] = int(value)
            return stats
        return {"cache_bytes": _cache_bytes(self.audio_cache) + _cache_bytes(self.language_cache) if self.audio_cache is not None else 0,
                "audio_frames": _cache_length(self.audio_cache) if self.audio_cache is not None else 0,
                "text_tokens": _cache_length(self.language_cache) if self.language_cache is not None else 0,
                "audio_position": self.absolute_audio_position, "text_position": self.absolute_text_position}

    def _input_ids(self, width):
        if type(width) is not int or width < 1:
            raise ValueError("Audio8 step width must be positive")
        if self._next_token is None:
            if width != PREFILL_TOKENS:
                raise RuntimeError("Audio8 requires its canonical initial prefill")
            r = self.runtime
            return [r.special_ids["bos_token_id"], r.language_ids[self.language]] + [r.special_ids["streaming_pad_token_id"]] * (width - 2)
        return [self._next_token] * width

    def _forward(self, audio_window, width):
        """One frontend/tower/feedback tick, also used by synthetic calibration."""
        ids = self._input_ids(width)
        if self._backend_step is not None:
            import operator
            token = operator.index(self._backend_step.step(audio_window, ids))
            if token < 0 or token == int(self.runtime.special_ids["eos_token_id"]):
                raise RuntimeError("Audio8 portable adapter must greedily suppress EOS")
            stats = self.cache_stats()
            if (stats["audio_frames"] > self.audio_window_frames or stats["text_tokens"] > self.window_tokens or
                    stats["audio_position"] != self.absolute_audio_position + width * FRAME_LEN or
                    stats["text_position"] != self.absolute_text_position + width):
                raise RuntimeError("Audio8 portable cache bounds/absolute positions violated")
            self.absolute_text_position = stats["text_position"]
            self.absolute_audio_position = stats["audio_position"]
            self._next_token = token
            return token
        r, torch, model = self.runtime, self.runtime.torch, self.runtime.model
        with torch.inference_mode():
            if self._condition is None:
                self._condition = model.build_t_cond(DELAY_TOKENS, batch_size=1,
                    device=r.device, dtype=r.dtype, frame_len=FRAME_LEN)
            features = self._features(feature_extractor=r.extractor, audio_windows=[audio_window],
                sampling_rate=SAMPLE_RATE, device=r.device).to(dtype=r.dtype)
            conv, _ = self._select(model.audio_tower.embedder(features), expected_20ms_frames=width * FRAME_LEN)
            audio_positions = torch.arange(self.absolute_audio_position,
                self.absolute_audio_position + width * FRAME_LEN, device=r.device, dtype=torch.long).unsqueeze(0)
            output = model.audio_tower(inputs_embeds=conv, position_ids=audio_positions,
                attention_mask=self._mask(width * FRAME_LEN, self.audio_cache,
                    self.absolute_audio_position, sliding=self._audio_sliding),
                past_key_values=self.audio_cache, use_cache=True, return_dict=True)
            self.audio_cache = output.past_key_values
            if int(output.last_hidden_state.shape[1]) != width * FRAME_LEN:
                raise RuntimeError("Audio8 tower frame count differs from the 160ms clock")
            grouped = self._group(model=model, audio_hidden_states=output.last_hidden_state,
                num_80ms_tokens=width, frame_len=FRAME_LEN)
            audio_embeds = model.multi_modal_projector(grouped).to(device=r.device, dtype=r.dtype)
            positions = torch.arange(self.absolute_text_position, self.absolute_text_position + width,
                device=r.device, dtype=torch.long)
            inputs = model.build_text_inputs_embeds(
                input_ids=torch.tensor([ids], device=r.device, dtype=torch.long), audio_embeds=audio_embeds).to(dtype=r.dtype)
            output = model.forward_language_model_with_delay(inputs_embeds=inputs,
                attention_mask=self._mask(width, self.language_cache, self.absolute_text_position, prefix=PREFILL_TOKENS),
                position_ids=positions.unsqueeze(0), cache_position=positions,
                past_key_values=self.language_cache, use_cache=True, logits_to_keep=1,
                num_delay_tokens=DELAY_TOKENS, frame_len=FRAME_LEN, t_cond=self._condition)
            self.language_cache = output.past_key_values
            logits = output.logits[0, -1].clone()
            logits[int(r.special_ids["eos_token_id"])] = float("-inf")
            self._next_token = int(logits.argmax(dim=-1).item())
            self.absolute_text_position += width
            self.absolute_audio_position += width * FRAME_LEN
            # At rest <= N; bounded forward temporaries add at most one text
            # token / eight audio frames (or the fixed 13-token prefill).
            crop_recent(self.language_cache, self.window_tokens, prefix=PREFILL_TOKENS)
            crop_recent(self.audio_cache, self.audio_window_frames)
        return self._next_token

    def _emit_token(self, token):
        self._steps += 1
        self._segment_steps += 1
        events = []
        if token not in self._skip_ids:
            piece = self.runtime.tokenizer.convert_ids_to_tokens(token)
            try:
                data = bytes(BYTE_DECODER[char] for char in piece)
            except (KeyError, TypeError) as exc:
                raise ValueError("Unexpected Audio8 tokenizer byte alphabet") from exc
            text = self._utf8.decode(data, final=False)
            if text:
                self._text += text
                events.append({"type": "partial", "text": self._text,
                    "segment_index": self._segment_index, "committed_characters": self._committed_chars})
        if self._segment_steps >= COMMIT_STEPS and self._text:
            events.extend(self._commit())
        return events

    def _commit(self):
        if not self._text:
            return []
        # Output step zero corresponds to audio token zero after the delay
        # prefill. Clamp right-drain timestamps to real, accepted PCM.
        end = min(self.accepted_samples, self._steps * SAMPLES_PER_TOKEN)
        event = {"type": "segment", "start": self._segment_start // 160,
                 "end": (end + 159) // 160, "text": self._text,
                 "segment_index": self._segment_index}
        self._committed_chars += len(self._text)
        self._text = ""
        self._segment_start = end
        self._segment_index += 1
        self._segment_steps = 0
        self.segment_count += 1
        return [event]

    def _drain(self, final=False):
        events = []
        left = LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN
        # Batch reference rounds only the final partial frame up, then adds
        # delay + 1 + right_pad_text_tokens silence. Its last window must still
        # have 40 look-ahead samples; thus no extra final clock token is emitted.
        real_tokens = max(1, (self.accepted_samples + SAMPLES_PER_TOKEN - 1) // SAMPLES_PER_TOKEN)
        available = (LEFT_PAD_TOKENS + real_tokens + DELAY_TOKENS + 1 + RIGHT_PAD_TEXT_TOKENS) * SAMPLES_PER_TOKEN if final else left + self.accepted_samples
        while self._clock_end + LOOK_AHEAD_SAMPLES <= available:
            width = (self._clock_end - self._clock_start) // SAMPLES_PER_TOKEN
            start = max(0, self._clock_start - LOOK_BACK_SAMPLES) - left
            end = self._clock_end + LOOK_AHEAD_SAMPLES - left
            window = self._pcm.window(start, end)
            token = self._forward(window, width)
            events.extend(self._emit_token(token))
            self._clock_start = self._clock_end
            self._clock_end += SAMPLES_PER_TOKEN
            self._pcm.discard_before(self._clock_start - left - LOOK_BACK_SAMPLES)
        return events

    def push_pcm(self, raw: bytes) -> list[dict]:
        self._enter()
        try:
            if self._finished:
                raise RuntimeError("Audio8 PCM supplied after finish")
            if len(raw) % 4:
                raise ValueError("Audio8 PCM must contain complete little-endian float32 samples")
            count = len(raw) // 4
            if count + self.buffered_samples > self.pcm_capacity_samples:
                raise BufferError("Audio8 PCM capacity exceeded; split frames/apply backpressure")
            samples = self.runtime.np.frombuffer(raw, dtype="<f4")
            if not self.runtime.np.isfinite(samples).all():
                raise ValueError("Audio8 PCM samples must be finite")
            self._pcm.append(self.runtime.np.clip(samples, -1.0, 1.0))
            try:
                return self._drain()
            except Exception:
                self._failed = True
                raise
        finally:
            self._lock.release()

    def finish(self) -> list[dict]:
        self._enter()
        try:
            if self._finished:
                return []
            self._finished = True
            try:
                # Empty input cannot contain speech. Do not hallucinate from
                # silence, and do not allocate transformer caches for it.
                events = self._drain(final=True) if self.accepted_samples else []
                self._utf8.decode(b"", final=True)  # fail visibly on incomplete UTF-8
                events.extend(self._commit())
                events.append({"type": "result", "segments": self.segment_count,
                    "samples": self.accepted_samples})
                self._pcm.discard_before(self.accepted_samples)
                self.language_cache = self.audio_cache = None
                self._backend_step = self._condition = None
                return events
            except Exception:
                self._failed = True
                raise
        finally:
            self._lock.release()


def _device_type(runtime):
    return getattr(runtime.device, "type", str(runtime.device).split(":", 1)[0])


def _synchronize(runtime):
    synchronize = getattr(runtime, "synchronize", None)
    if callable(synchronize):
        synchronize()
    elif hasattr(runtime, "torch"):
        if _device_type(runtime) == "cuda":
            runtime.torch.cuda.synchronize(runtime.device)
        elif _device_type(runtime) == "mps":
            runtime.torch.mps.synchronize()


def _pcm_capacity(prefill_seconds, step_seconds, mean_step_seconds):
    # Input admission must fit the delayed first frontend window, its retained
    # overlap, and a measured producer stall. Two observed jitter widths are a
    # conservative cushion, not permission to conceal inference slower than
    # capture. Capacity is bounded/rounded to complete 160ms PCM packets.
    jitter = max(0.0, step_seconds - mean_step_seconds)
    stall = max(prefill_seconds, step_seconds) + 2.0 * jitter
    required = ((DELAY_TOKENS + 1) * SAMPLES_PER_TOKEN + LOOK_AHEAD_SAMPLES +
                LOOK_BACK_SAMPLES + math.ceil(SAMPLE_RATE * stall))
    required = max(MIN_PCM_CAPACITY_SAMPLES,
                   math.ceil(required / SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN)
    return min(required, MAX_PCM_CAPACITY_SAMPLES), required <= MAX_PCM_CAPACITY_SAMPLES


def _hardware_key(runtime):
    import platform
    import subprocess
    supplied = getattr(runtime, "hardware_key", None)
    if supplied is not None:
        return str(supplied() if callable(supplied) else supplied)
    identity = platform.machine()
    torch = getattr(runtime, "torch", None)
    if _device_type(runtime) == "cuda" and torch is not None:
        identity = torch.cuda.get_device_name(runtime.device)
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True,
                                    text=True, check=True, timeout=2)
            identity = result.stdout.strip() or identity
        except (OSError, subprocess.SubprocessError):
            pass
    version = str(torch.__version__) if torch is not None else str(getattr(runtime, "backend_version", type(runtime).__name__))
    threads = torch.get_num_threads() if torch is not None else getattr(runtime, "threads", "unknown")
    return "|".join((identity, platform.release(), str(runtime.device), str(runtime.dtype),
                     version, f"threads={threads}"))


def calibrate(runtime, abort_check=None) -> dict:
    """Short synthetic, saturated-cache trials; does not load any model weights.

    Seed each trial with a batched synthetic context (not timed/not transcription)
    then warm up and measure five ordinary frontend + eight-frame + feedback
    ticks with device synchronisation. Choose the largest bounded context whose
    worst observed tick has 20% headroom, not a promise of future latency/WER.
    abort_check is called between windows/ticks; True aborts with InterruptedError
    (callbacks may also raise). The measured ring size becomes the runtime's
    constructor default; callers may pass pcm_capacity_samples explicitly.
    """
    def check_abort():
        if abort_check is not None and abort_check():
            raise InterruptedError("Audio8 calibration cancelled")
    trials = []
    budget = int(getattr(runtime, "streaming_memory_budget_bytes", 1024 ** 3))
    if budget <= 0:
        raise ValueError("Audio8 streaming memory budget must be positive")
    for window in (32, 64, 128, 256):
        check_abort()
        decoder = StreamingDecoder(runtime, "en", window)
        zeros = runtime.np.zeros(PREFILL_TOKENS * SAMPLES_PER_TOKEN + LOOK_AHEAD_SAMPLES, dtype=runtime.np.float32)
        started = time.perf_counter()
        decoder._forward(zeros, PREFILL_TOKENS)
        _synchronize(runtime)
        prefill_seconds = time.perf_counter() - started
        check_abort()
        # Populate the measured *physical* cache to this candidate's maximum
        # in one synthetic forward. No repeated/copied fake KV entries.
        fill = window - PREFILL_TOKENS
        zeros = runtime.np.zeros(fill * SAMPLES_PER_TOKEN + LOOK_BACK_SAMPLES + LOOK_AHEAD_SAMPLES, dtype=runtime.np.float32)
        decoder._forward(zeros, fill)
        stats = decoder.cache_stats()
        if (stats["text_tokens"] != window or stats["audio_frames"] != decoder.audio_window_frames or
                stats["cache_bytes"] <= 0):
            raise RuntimeError("Audio8 calibration cache is not genuinely saturated")
        tick = runtime.np.zeros(SAMPLES_PER_TOKEN + LOOK_BACK_SAMPLES + LOOK_AHEAD_SAMPLES, dtype=runtime.np.float32)
        for _ in range(3):
            check_abort()
            decoder._forward(tick, 1)
        _synchronize(runtime)
        seconds = []
        for _ in range(5):
            check_abort()
            _synchronize(runtime)
            started = time.perf_counter()
            decoder._forward(tick, 1)
            _synchronize(runtime)
            seconds.append(time.perf_counter() - started)
        cache_bytes = decoder.cache_stats()["cache_bytes"]
        mean_seconds = sum(seconds) / len(seconds)
        capacity, sufficient = _pcm_capacity(prefill_seconds, max(seconds), mean_seconds)
        trials.append({"window_tokens": window, "step_seconds": max(seconds),
            "mean_step_seconds": mean_seconds, "cache_bytes": cache_bytes,
            "prefill_seconds": prefill_seconds, "pcm_capacity_samples": capacity,
            "pcm_capacity_sufficient": sufficient,
            "realtime": max(seconds) <= 0.8 * 0.160 and cache_bytes * 2 + capacity * 4 <= budget and sufficient})
        del decoder
    check_abort()
    eligible = [trial for trial in trials if trial["realtime"]]
    selected = max(eligible, key=lambda trial: trial["window_tokens"]) if eligible else min(trials, key=lambda trial: trial["step_seconds"])
    runtime.streaming_pcm_capacity_samples = selected["pcm_capacity_samples"]
    return {**selected, "hardware_key": _hardware_key(runtime),
        "device": str(runtime.device), "dtype": str(runtime.dtype), "frame_seconds": 0.160,
        "delay_seconds": 0.480, "memory_budget_bytes": budget, "trials": trials,
        "measurement": "synthetic saturated caches; 3 warmup + 5 synchronised ticks",
        "timestamps": "emission-clock bounds, not alignment", "accuracy_validated": False}
