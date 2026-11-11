//! RAM-only recording. No filesystem or ffmpeg path is reachable from this module.
use crate::cmd::audio::{buffer_peak, get_output_device_and_config, AudioDevice, LevelMeter, StreamHandle};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use eyre::{bail, eyre, Context, ContextCompat, Result};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Listener};
use tauri_plugin_store::StoreExt;

const PREFIX: &str = "vibe-memory:";
const RATE: u32 = 16_000;
const HEADER_LEN: usize = 44;
// Two hours of mono PCM16, including the header, fit below 256 MiB. The process-wide
// quota also covers queued captures and ref-counted request bodies, not just recording.
const MAX_CAPTURE_BYTES: usize = HEADER_LEN + RATE as usize * 2 * 60 * 60 * 2;
const MAX_TOTAL_BYTES: usize = 240 * 1024 * 1024;
const GROW_BYTES: usize = 512 * 1024;
const MAX_HANDLES: usize = 16;
static RESERVED: AtomicUsize = AtomicUsize::new(0);
static RECORDING: AtomicBool = AtomicBool::new(false);
static AUDIO: OnceLock<Mutex<HashMap<String, Bytes>>> = OnceLock::new();

fn audio_store() -> &'static Mutex<HashMap<String, Bytes>> {
    AUDIO.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn is_memory_path(path: &str) -> bool {
    path.starts_with(PREFIX)
}

/// Read at recording start, so tray and shortcut invocations enforce the same policy.
/// An unreadable or malformed setting fails closed instead of silently recording to disk.
pub fn enabled(app: &AppHandle) -> Result<bool> {
    let store = app
        .store(crate::config::STORE_FILENAME)
        .context("Failed to read recording privacy setting")?;
    match store.get("privacy.noAudioStorage") {
        None => Ok(false),
        Some(value) => value.as_bool().context("Recording privacy setting must be a boolean"),
    }
}

/// Register incoming audio without changing its encoding or copying its allocation.
/// The server's RAM-only decoder handles compressed inputs through ffmpeg pipes.
pub fn store_bytes(bytes: Vec<u8>) -> Result<String> {
    let capacity = bytes.capacity();
    if bytes.is_empty() || capacity > MAX_CAPTURE_BYTES {
        bail!("Private audio is empty or exceeds the RAM capture limit");
    }
    RESERVED
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(capacity).filter(|total| *total <= MAX_TOTAL_BYTES)
        })
        .map_err(|_| eyre!("Private audio RAM quota is full; transcribe or discard pending recordings"))?;
    store(Bytes::from_owner(WavBuffer {
        data: bytes,
        reserved: capacity,
        limit: MAX_CAPTURE_BYTES,
        captured: true,
    }))
}

fn store(bytes: Bytes) -> Result<String> {
    let mut captures = audio_store().lock().map_err(|_| eyre!("Private audio store lock failed"))?;
    if captures.len() >= MAX_HANDLES {
        bail!("Too many pending private recordings; transcribe or discard them first");
    }
    let path = format!("{PREFIX}{}", uuid::Uuid::new_v4());
    captures.insert(path.clone(), bytes);
    Ok(path)
}

/// Bytes clones share one allocation, including across connection retries.
pub fn get(path: &str) -> Result<Bytes> {
    audio_store()
        .lock()
        .map_err(|_| eyre!("Private audio store lock failed"))?
        .get(path)
        .cloned()
        .context("Private recording is no longer available")
}

#[allow(dead_code)]
pub fn take(path: &str) -> Result<Bytes> {
    audio_store()
        .lock()
        .map_err(|_| eyre!("Private audio store lock failed"))?
        .remove(path)
        .context("Private recording is no longer available")
}

fn discard(path: &str) {
    if is_memory_path(path) {
        // Recover poisoning for cleanup: privacy cleanup must not leave a retained capture.
        audio_store().lock().unwrap_or_else(|e| e.into_inner()).remove(path);
    }
}

#[tauri::command]
pub fn discard_private_audio(path: String) {
    discard(&path);
}

/// Cleans up early failures, cancellation and dropped transcription futures as well as success.
pub struct Cleanup(Option<String>);
impl Cleanup {
    pub fn new(path: &str) -> Result<Self> {
        if is_memory_path(path) {
            let captures = audio_store().lock().map_err(|_| eyre!("Private audio store lock failed"))?;
            if !captures.contains_key(path) {
                bail!("Private recording is no longer available");
            }
            Ok(Self(Some(path.to_owned())))
        } else {
            Ok(Self(None))
        }
    }
}
impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            discard(path);
        }
    }
}

// The quota travels with the Bytes owner. Removing a handle during an upload cannot
// release its quota until the last request-body reference has actually been dropped.
struct WavBuffer {
    data: Vec<u8>,
    reserved: usize,
    limit: usize,
    captured: bool,
}
impl AsRef<[u8]> for WavBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}
impl Drop for WavBuffer {
    fn drop(&mut self) {
        RESERVED.fetch_sub(self.reserved, Ordering::AcqRel);
    }
}
impl WavBuffer {
    fn new(limit: usize) -> Result<Self> {
        let mut buffer = Self {
            data: Vec::new(),
            reserved: 0,
            limit,
            captured: false,
        };
        buffer.ensure_len(HEADER_LEN)?;
        Ok(buffer)
    }

    fn ensure_len(&mut self, len: usize) -> Result<()> {
        if len > self.limit {
            bail!("Private recording exceeded the two-hour RAM limit; no audio was written to disk");
        }
        if len > self.data.capacity() {
            let capacity = len.max(self.data.capacity().saturating_add(GROW_BYTES)).min(self.limit);
            let additional = capacity - self.data.capacity();
            RESERVED
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                    used.checked_add(additional).filter(|total| *total <= MAX_TOTAL_BYTES)
                })
                .map_err(|_| eyre!("Private audio RAM quota is full; transcribe or discard pending recordings"))?;
            if let Err(error) = self.data.try_reserve_exact(capacity - self.data.len()) {
                RESERVED.fetch_sub(additional, Ordering::AcqRel);
                return Err(eyre!("Could not allocate private recording buffer: {error}"));
            }
            self.reserved += additional;
        }
        if len > self.data.len() {
            self.data.resize(len, 0);
        }
        Ok(())
    }

    fn mix(&mut self, frame: usize, sample: f32) -> Result<()> {
        let offset = frame
            .checked_mul(2)
            .and_then(|n| n.checked_add(HEADER_LEN))
            .context("Private recording frame offset overflow")?;
        self.ensure_len(offset.checked_add(2).context("Private recording size overflow")?)?;
        let old = i16::from_le_bytes([self.data[offset], self.data[offset + 1]]) as i32;
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i32;
        let mixed = (old + value).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        self.data[offset..offset + 2].copy_from_slice(&mixed.to_le_bytes());
        self.captured = true;
        Ok(())
    }

    fn finish(mut self) -> Result<Bytes> {
        if !self.captured {
            bail!("Recording stopped without any captured audio");
        }
        let data_len = (self.data.len() - HEADER_LEN) as u32;
        let header = &mut self.data[..HEADER_LEN];
        header[0..4].copy_from_slice(b"RIFF");
        header[4..8].copy_from_slice(&(data_len + 36).to_le_bytes());
        header[8..16].copy_from_slice(b"WAVEfmt ");
        header[16..20].copy_from_slice(&16u32.to_le_bytes());
        header[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
        header[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
        header[24..28].copy_from_slice(&RATE.to_le_bytes());
        header[28..32].copy_from_slice(&(RATE * 2).to_le_bytes());
        header[32..34].copy_from_slice(&2u16.to_le_bytes());
        header[34..36].copy_from_slice(&16u16.to_le_bytes());
        header[36..40].copy_from_slice(b"data");
        header[40..44].copy_from_slice(&data_len.to_le_bytes());
        Ok(Bytes::from_owner(self))
    }
}

/// A timeline-indexed destination shared by private WAV and bounded live capture.
pub(crate) trait SampleSink {
    fn mix(&mut self, frame: usize, sample: f32) -> Result<()>;
}
impl SampleSink for WavBuffer {
    fn mix(&mut self, frame: usize, sample: f32) -> Result<()> {
        WavBuffer::mix(self, frame, sample)
    }
}

/// Streaming area-weighted resampling: downmix frames first, then integrate their
/// intervals into 16kHz output intervals. No source-rate recording or scratch Vec is kept.
pub(crate) struct Resampler {
    rate: u32,
    channels: usize,
    gain: f32,
    remaining: u32,
    weighted: f64,
    next_frame: usize,
    first_capture: Option<cpal::StreamInstant>,
    first_frame: usize,
}
impl Resampler {
    pub(crate) fn new(rate: u32, channels: u16, sources: usize) -> Result<Self> {
        if rate == 0 || channels == 0 || sources == 0 {
            bail!("Invalid private recording audio configuration");
        }
        Ok(Self {
            rate,
            channels: channels as usize,
            gain: 1.0 / sources as f32,
            remaining: rate,
            weighted: 0.0,
            next_frame: 0,
            first_capture: None,
            first_frame: 0,
        })
    }

    pub(crate) fn position(&self) -> usize {
        self.next_frame
    }

    pub(crate) fn align<S: SampleSink>(
        &mut self,
        info: &cpal::InputCallbackInfo,
        started: Instant,
        buffer: &mut S,
    ) -> Result<()> {
        let stamp = info.timestamp();
        let frame = match self.first_capture {
            Some(first) => {
                self.first_frame
                    + if stamp.capture >= first {
                        (stamp.capture.duration_since(first).as_secs_f64() * RATE as f64) as usize
                    } else {
                        0
                    }
            }
            None => {
                let latency = if stamp.callback >= stamp.capture {
                    stamp.callback.duration_since(stamp.capture)
                } else {
                    std::time::Duration::ZERO
                };
                let offset = (started.elapsed().saturating_sub(latency).as_secs_f64() * RATE as f64) as usize;
                self.first_capture = Some(stamp.capture);
                self.first_frame = offset;
                self.next_frame = offset;
                offset
            }
        };
        // Timestamp-based placement preserves device start offsets and real capture gaps.
        // Ignore sub-20ms timestamp jitter; never shift continuous buffers to callback time.
        if frame > self.next_frame.saturating_add(RATE as usize / 50) {
            self.flush(buffer)?;
            self.next_frame = frame;
        }
        Ok(())
    }

    pub(crate) fn push<T, S>(&mut self, data: &[T], buffer: &mut S) -> Result<()>
    where
        T: Sample,
        S: SampleSink,
        f32: FromSample<T>,
    {
        if data.len() % self.channels != 0 {
            bail!("Private recording received an incomplete audio frame");
        }
        for frame in data.chunks_exact(self.channels) {
            let mono = frame
                .iter()
                .map(|sample| {
                    let value = f32::from_sample(*sample);
                    if value.is_finite() {
                        value
                    } else {
                        0.0
                    }
                })
                .sum::<f32>()
                / self.channels as f32
                * self.gain;
            let mut duration = RATE;
            while duration > 0 {
                let weight = duration.min(self.remaining);
                self.weighted += mono as f64 * weight as f64;
                self.remaining -= weight;
                duration -= weight;
                if self.remaining == 0 {
                    buffer.mix(self.next_frame, (self.weighted / self.rate as f64) as f32)?;
                    self.next_frame += 1;
                    self.remaining = self.rate;
                    self.weighted = 0.0;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn flush<S: SampleSink>(&mut self, buffer: &mut S) -> Result<()> {
        let weight = self.rate - self.remaining;
        if weight > 0 {
            buffer.mix(self.next_frame, (self.weighted / weight as f64) as f32)?;
            self.next_frame += 1;
            self.remaining = self.rate;
            self.weighted = 0.0;
        }
        Ok(())
    }
}

struct Capture {
    buffer: Option<WavBuffer>,
    resamplers: Vec<Resampler>,
}
enum Stop {
    Requested,
    Failed(String),
}
struct RecordingLease;
impl Drop for RecordingLease {
    fn drop(&mut self) {
        RECORDING.store(false, Ordering::Release);
    }
}

fn fail(capture: &Arc<Mutex<Capture>>, sender: &mpsc::Sender<Stop>, message: String) {
    let mut capture = capture.lock().unwrap_or_else(|e| e.into_inner());
    if capture.buffer.take().is_some() {
        let _ = sender.send(Stop::Failed(message));
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: cpal::SupportedStreamConfig,
    index: usize,
    capture: Arc<Mutex<Capture>>,
    meter: Arc<LevelMeter>,
    started: Instant,
    sender: mpsc::Sender<Stop>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let errors = capture.clone();
    let error_sender = sender.clone();
    Ok(device.build_input_stream(
        config.into(),
        move |data: &[T], info| {
            meter.push(buffer_peak(data));
            let result = {
                let mut guard = capture.lock().unwrap_or_else(|e| e.into_inner());
                let Capture { buffer, resamplers } = &mut *guard;
                match buffer {
                    Some(buffer) => resamplers[index]
                        .align(info, started, buffer)
                        .and_then(|()| resamplers[index].push(data, buffer)),
                    None => return,
                }
            };
            if let Err(error) = result {
                fail(&capture, &sender, error.to_string());
            }
        },
        move |_| {
            fail(&errors, &error_sender, "Private recording audio stream failed".to_string());
        },
        None,
    )?)
}

pub fn start_record(app: AppHandle, devices: Vec<AudioDevice>, recording_name: Option<String>) -> Result<()> {
    if RECORDING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        bail!("A private recording is already in progress");
    }
    let lease = RecordingLease;
    let host = cpal::default_host();
    let started = Instant::now();
    let meter = Arc::new(LevelMeter::new(app.clone()));
    let capture = Arc::new(Mutex::new(Capture {
        buffer: Some(WavBuffer::new(MAX_CAPTURE_BYTES)?),
        resamplers: Vec::new(),
    }));
    let (sender, receiver) = mpsc::channel();
    let mut streams = Vec::new();
    let sources = devices.len();
    for audio_device in devices {
        let (device, config) = if audio_device.is_input {
            let id: usize = audio_device.id.parse().context("Failed to parse device ID")?;
            let device = host.devices()?.nth(id).context("Failed to get device by ID")?;
            let config = device.default_input_config().context("Failed to get default input config")?;
            (device, config)
        } else {
            get_output_device_and_config(&host, &audio_device)?
        };
        let index = streams.len();
        capture
            .lock()
            .map_err(|_| eyre!("Private capture lock failed"))?
            .resamplers
            .push(Resampler::new(config.sample_rate(), config.channels(), sources)?);
        macro_rules! build {
            ($ty:ty) => {
                build_stream::<$ty>(
                    &device,
                    config,
                    index,
                    capture.clone(),
                    meter.clone(),
                    started,
                    sender.clone(),
                )?
            };
        }
        let stream = match config.sample_format() {
            cpal::SampleFormat::I8 => build!(i8),
            cpal::SampleFormat::I16 => build!(i16),
            cpal::SampleFormat::I32 => build!(i32),
            cpal::SampleFormat::I64 => build!(i64),
            cpal::SampleFormat::U8 => build!(u8),
            cpal::SampleFormat::U16 => build!(u16),
            cpal::SampleFormat::U32 => build!(u32),
            cpal::SampleFormat::U64 => build!(u64),
            cpal::SampleFormat::F32 => build!(f32),
            cpal::SampleFormat::F64 => build!(f64),
            format => bail!("Unsupported private recording sample format '{format}'"),
        };
        streams.push(StreamHandle(stream));
    }
    // `listen`, not `once`: a repeated stop_record must be a no-op, not a tauri panic.
    let listener = app.listen("stop_record", move |_| {
        let _ = sender.send(Stop::Requested);
    });
    for stream in &streams {
        if let Err(error) = stream.0.play() {
            app.unlisten(listener);
            return Err(error.into());
        }
    }
    crate::meeting_prompt::recording_started(&app);
    // Finalize off the audio callback thread: pausing/dropping its own stream can deadlock.
    // Builder errors still drop streams, capture, listener and the recording lease.
    let thread_app = app.clone();
    if let Err(error) = std::thread::Builder::new().name("private-audio".into()).spawn(move || {
        let _lease = lease;
        let stop = receiver.recv();
        thread_app.unlisten(listener);
        let _ = thread_app.emit("record_stopping", ());
        for stream in &streams {
            let _ = stream.0.pause();
        }
        drop(streams);
        let result = (|| -> Result<String> {
            if let Ok(Stop::Failed(message)) = stop {
                bail!("{message}");
            }
            let mut guard = capture.lock().map_err(|_| eyre!("Private capture lock failed"))?;
            let Capture { buffer, resamplers } = &mut *guard;
            let mut buffer = buffer.take().context("Private recording capture failed")?;
            for resampler in resamplers {
                resampler.flush(&mut buffer)?;
            }
            store(buffer.finish()?)
        })();
        match result {
            Ok(path) => {
                let stem = recording_name
                    .as_deref()
                    .map(crate::cmd::files::sanitize_filename_stem)
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(crate::ffmpeg::get_local_time);
                if thread_app
                    .emit(
                        "record_finish",
                        json!({ "path": path, "name": format!("{stem}.wav"), "warning": null }),
                    )
                    .is_err()
                {
                    discard(&path);
                }
            }
            Err(error) => {
                let _ = thread_app.emit("record_error", json!({ "message": error.to_string() }));
            }
        }
        crate::meeting_prompt::recording_stopped(&thread_app);
    }) {
        app.unlisten(listener);
        crate::meeting_prompt::recording_stopped(&app);
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn samples(bytes: &Bytes) -> Vec<i16> {
        let mut wav = hound::WavReader::new(Cursor::new(bytes.as_ref())).unwrap();
        assert_eq!(wav.spec().sample_rate, RATE);
        assert_eq!(wav.spec().channels, 1);
        assert_eq!(wav.spec().bits_per_sample, 16);
        wav.samples::<i16>().map(Result::unwrap).collect()
    }

    #[test]
    fn resamples_48k_stereo_across_callback_boundaries() {
        let mut buffer = WavBuffer::new(1024).unwrap();
        let mut resampler = Resampler::new(48_000, 2, 1).unwrap();
        resampler.push(&[1.0f32, 0.0, 1.0, 0.0], &mut buffer).unwrap();
        resampler
            .push(&[1.0f32, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0], &mut buffer)
            .unwrap();
        assert_eq!(samples(&buffer.finish().unwrap()), vec![16384, -16384]);
    }

    #[test]
    fn fractional_rate_and_upsampling_preserve_duration() {
        let mut buffer = WavBuffer::new(40_000).unwrap();
        let mut resampler = Resampler::new(44_100, 1, 1).unwrap();
        for _ in 0..100 {
            resampler.push(&[0.25f32; 441], &mut buffer).unwrap();
        }
        let output = samples(&buffer.finish().unwrap());
        assert_eq!(output.len(), 16_000);
        assert!(output.iter().all(|s| *s == 8192));
        let mut buffer = WavBuffer::new(1024).unwrap();
        Resampler::new(8_000, 1, 1)
            .unwrap()
            .push(&[0.5f32, -0.5], &mut buffer)
            .unwrap();
        assert_eq!(samples(&buffer.finish().unwrap()), vec![16384, 16384, -16384, -16384]);
    }

    #[test]
    fn mixing_keeps_offsets_and_silence_instead_of_concatenating_sources() {
        let mut buffer = WavBuffer::new(1024).unwrap();
        let mut mic = Resampler::new(RATE, 1, 2).unwrap();
        let mut loopback = Resampler::new(RATE, 2, 2).unwrap();
        loopback.next_frame = 2;
        mic.push(&[0.5f32; 3], &mut buffer).unwrap();
        loopback.push(&[0.5f32; 4], &mut buffer).unwrap();
        buffer.mix(5, 0.25).unwrap();
        assert_eq!(samples(&buffer.finish().unwrap()), vec![8192, 8192, 16384, 8192, 0, 8192]);
    }

    #[test]
    fn empty_and_overflow_fail_without_a_disk_fallback() {
        assert!(WavBuffer::new(100).unwrap().finish().is_err());
        let mut buffer = WavBuffer::new(HEADER_LEN + 2).unwrap();
        buffer.mix(0, 0.1).unwrap();
        assert!(buffer.mix(1, 0.1).is_err());
        assert_eq!(buffer.data.len(), HEADER_LEN + 2);
        assert!(buffer.mix(usize::MAX, 0.1).is_err());
    }

    #[test]
    fn short_buffers_flush_the_partial_output_frame() {
        let mut buffer = WavBuffer::new(1024).unwrap();
        let mut resampler = Resampler::new(48_000, 2, 1).unwrap();
        resampler.push(&[0.5f32, 0.5], &mut buffer).unwrap();
        resampler.flush(&mut buffer).unwrap();
        assert_eq!(samples(&buffer.finish().unwrap()), vec![16384]);
    }

    #[test]
    fn failure_drops_capture_and_signals_the_finalizer() {
        let capture = Arc::new(Mutex::new(Capture {
            buffer: Some(WavBuffer::new(100).unwrap()),
            resamplers: Vec::new(),
        }));
        let (tx, rx) = mpsc::channel();
        fail(&capture, &tx, "limit".into());
        assert!(capture.lock().unwrap().buffer.is_none());
        assert!(matches!(rx.recv().unwrap(), Stop::Failed(message) if message == "limit"));
        fail(&capture, &tx, "again".into());
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn cancel_and_failure_scope_cleanup_and_retry_share_one_allocation() {
        let mut buffer = WavBuffer::new(100).unwrap();
        buffer.mix(0, 0.1).unwrap();
        let path = store(buffer.finish().unwrap()).unwrap();
        {
            let _guard = Cleanup::new(&path).unwrap();
            let first = get(&path).unwrap();
            let retry = get(&path).unwrap();
            assert_eq!(first.as_ptr(), retry.as_ptr());
        }
        assert!(get(&path).is_err());
        discard_private_audio(path); // idempotent
        let _guard = Cleanup::new("imported.wav").unwrap(); // imported files are never touched
    }
}
