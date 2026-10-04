//! Bounded, RAM-only live PCM capture. Callbacks only produce into the mixing ring.
use crate::cmd::audio::{buffer_peak, get_output_device_and_config, AudioDevice, LevelMeter, StreamHandle};
use crate::private_audio::{Resampler, SampleSink};
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use eyre::{bail, Context, ContextCompat, Result};
use futures::channel::mpsc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tauri::AppHandle;

const CHUNK_SAMPLES: usize = 1_600;
const WORKER_TICK: Duration = Duration::from_millis(20);
pub(crate) const RUNNING: u8 = 0;
pub(crate) const STOP: u8 = 1;
pub(crate) const CANCEL: u8 = 2;
pub(crate) const FAILED: u8 = 3;
pub(crate) type UploadSender = mpsc::Sender<std::result::Result<Bytes, std::io::Error>>;

#[derive(Default)]
pub(crate) struct Control {
    state: AtomicU8,
    error: Mutex<Option<String>>,
    upload_aborted: AtomicBool,
    upload_changed: tokio::sync::Notify,
    pub(crate) changed: tokio::sync::Notify,
}
impl Control {
    pub(crate) fn state(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }
    pub(crate) fn stop(&self) {
        let _ = self
            .state
            .compare_exchange(RUNNING, STOP, Ordering::AcqRel, Ordering::Acquire);
        self.changed.notify_one();
    }
    pub(crate) fn cancel(&self) {
        self.state.store(CANCEL, Ordering::Release);
        self.abort_upload();
        self.changed.notify_one();
    }
    pub(crate) fn abort_upload(&self) {
        self.upload_aborted.store(true, Ordering::Release);
        self.upload_changed.notify_one();
    }
    pub(crate) async fn upload_cancelled(&self) {
        loop {
            let changed = self.upload_changed.notified();
            if self.upload_aborted.load(Ordering::Acquire) {
                return;
            }
            changed.await;
        }
    }
    pub(crate) fn fail(&self, message: String) {
        let mut error = self.error.lock().unwrap_or_else(|e| e.into_inner());
        let mut state = self.state();
        if state >= CANCEL {
            return;
        }
        *error = Some(message);
        loop {
            match self
                .state
                .compare_exchange(state, FAILED, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    self.abort_upload();
                    self.changed.notify_one();
                    return;
                }
                Err(updated) if updated < CANCEL => state = updated,
                Err(_) => return, // an abort must never be downgraded to an error
            }
        }
    }
    pub(crate) fn error(&self) -> String {
        self.error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap_or_else(|| "Live capture failed".into())
    }
}

/// Absolute frame indices preserve device offsets, silence and capture gaps across wraparound.
/// A source's horizon is the first frame it might still modify; only the common prefix is sent.
struct MixRing {
    samples: Vec<f32>,
    base: usize,
    end: usize,
    horizons: Vec<usize>,
    finished: bool,
}
impl MixRing {
    fn new(capacity: usize, sources: usize) -> Result<Self> {
        if capacity == 0 || sources == 0 {
            bail!("Invalid live PCM ring capacity or source count");
        }
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(capacity)
            .context("Could not allocate bounded live PCM ring")?;
        samples.resize(capacity, 0.0);
        Ok(Self {
            samples,
            base: 0,
            end: 0,
            horizons: vec![0; sources],
            finished: false,
        })
    }
    fn available_end(&self) -> usize {
        if self.finished {
            self.end
        } else {
            self.horizons.iter().copied().min().unwrap_or(0).min(self.end)
        }
    }
    fn take_chunk(&mut self) -> Option<Bytes> {
        let count = self.available_end().saturating_sub(self.base).min(CHUNK_SAMPLES);
        if count == 0 {
            return None;
        }
        let mut bytes = Vec::with_capacity(count * 4);
        for frame in self.base..self.base + count {
            let slot = frame % self.samples.len();
            bytes.extend_from_slice(&self.samples[slot].clamp(-1.0, 1.0).to_le_bytes());
            self.samples[slot] = 0.0;
        }
        self.base += count;
        Some(Bytes::from(bytes))
    }
}
impl SampleSink for MixRing {
    fn mix(&mut self, frame: usize, sample: f32) -> Result<()> {
        if frame < self.base {
            bail!("Live capture fatal lag: a device arrived after its PCM was uploaded");
        }
        if frame - self.base >= self.samples.len() {
            bail!("Live capture PCM overflow: the backend or an audio device cannot keep up; no audio was silently dropped");
        }
        let slot = frame % self.samples.len();
        self.samples[slot] += sample;
        self.end = self.end.max(frame.checked_add(1).context("Live PCM timeline overflow")?);
        Ok(())
    }
}

struct Producer {
    ring: MixRing,
    resamplers: Vec<Resampler>,
    started: Option<Instant>,
    accepting: bool,
}
struct Shared {
    producer: Mutex<Producer>,
    wake: Condvar,
    peak: AtomicU32,
    abandoned: AtomicBool,
    control: Arc<Control>,
}

pub(crate) struct Capture {
    streams: Vec<StreamHandle>,
    shared: Arc<Shared>,
}
impl Capture {
    /// Called only after the server has sent a realtime calibration followed by ready.
    pub(crate) fn start(
        app: AppHandle,
        devices: Vec<AudioDevice>,
        capacity: usize,
        upload: UploadSender,
        control: Arc<Control>,
    ) -> Result<Self> {
        if control.state() != RUNNING {
            bail!("Live recording cancelled before capture started");
        }
        let sources = devices.len();
        let shared = Arc::new(Shared {
            producer: Mutex::new(Producer {
                ring: MixRing::new(capacity, sources)?,
                resamplers: Vec::with_capacity(sources),
                started: None,
                accepting: false,
            }),
            wake: Condvar::new(),
            peak: AtomicU32::new(0),
            abandoned: AtomicBool::new(false),
            control,
        });
        let mut capture = Self {
            streams: Vec::with_capacity(sources),
            shared,
        };
        let host = cpal::default_host();
        for audio_device in devices {
            if capture.shared.control.state() != RUNNING {
                bail!("Live recording cancelled before capture started");
            }
            let (device, config) = if audio_device.is_input {
                let id: usize = audio_device.id.parse().context("Failed to parse audio device ID")?;
                let device = host.devices()?.nth(id).context("Failed to get audio device by ID")?;
                let config = device
                    .default_input_config()
                    .context("Failed to get default audio input config")?;
                (device, config)
            } else {
                get_output_device_and_config(&host, &audio_device)?
            };
            let index = capture.streams.len();
            capture
                .shared
                .producer
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .resamplers
                .push(Resampler::new(config.sample_rate(), config.channels(), sources)?);
            macro_rules! build {
                ($ty:ty) => {
                    build_stream::<$ty>(&device, config, index, capture.shared.clone())?
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
                format => bail!("Unsupported live recording sample format '{format}'"),
            };
            capture.streams.push(StreamHandle(stream));
        }
        if capture.shared.control.state() != RUNNING {
            bail!("Live recording cancelled before capture started");
        }
        let worker_shared = capture.shared.clone();
        let meter = LevelMeter::new(app);
        std::thread::Builder::new()
            .name("live-pcm-upload".into())
            .spawn(move || {
                upload_worker(worker_shared, upload, Some(meter));
            })
            .context("Failed to start live PCM upload worker")?;
        {
            let mut producer = capture.shared.producer.lock().unwrap_or_else(|e| e.into_inner());
            producer.started = Some(Instant::now());
            producer.accepting = true;
        }
        for stream in &capture.streams {
            if capture.shared.control.state() != RUNNING {
                bail!("Live recording cancelled before capture started");
            }
            stream.0.play().context("Failed to start live audio capture")?;
        }
        Ok(capture)
    }

    /// Stop callbacks first, flush partial resampler frames, then let the worker drain to EOF.
    pub(crate) fn finish(&mut self) -> Result<()> {
        self.close_streams();
        let mut guard = self.shared.producer.lock().unwrap_or_else(|e| e.into_inner());
        let Producer { ring, resamplers, .. } = &mut *guard;
        for resampler in resamplers {
            resampler.flush(ring)?;
        }
        ring.finished = true;
        self.shared.wake.notify_all();
        Ok(())
    }
    fn close_streams(&mut self) {
        self.shared.producer.lock().unwrap_or_else(|e| e.into_inner()).accepting = false;
        for stream in &self.streams {
            let _ = stream.0.pause();
        }
        self.streams.clear();
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        self.close_streams();
        let finished = self.shared.producer.lock().unwrap_or_else(|e| e.into_inner()).ring.finished;
        if !finished {
            self.shared.abandoned.store(true, Ordering::Release);
        }
        self.shared.wake.notify_all();
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: cpal::SupportedStreamConfig,
    index: usize,
    shared: Arc<Shared>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let errors = shared.clone();
    Ok(device.build_input_stream(
        config.into(),
        move |data: &[T], info| {
            if shared.control.state() >= CANCEL {
                return;
            }
            shared.peak.fetch_max(buffer_peak(data).to_bits(), Ordering::Relaxed);
            let result = {
                let mut guard = shared.producer.lock().unwrap_or_else(|e| e.into_inner());
                let Producer {
                    ring,
                    resamplers,
                    started,
                    accepting,
                } = &mut *guard;
                if !*accepting {
                    return;
                }
                let resampler = &mut resamplers[index];
                resampler
                    .align(info, started.expect("capture start epoch"), ring)
                    .and_then(|()| {
                        resampler.push(data, ring)?;
                        ring.horizons[index] = resampler.position();
                        Ok(())
                    })
            };
            if let Err(error) = result {
                shared.control.fail(error.to_string());
            }
            shared.wake.notify_one();
        },
        move |error| {
            errors.control.fail(format!("Live audio stream failed: {error}"));
            errors.wake.notify_all();
        },
        None,
    )?)
}

/// Serialize outside the callback. Backpressure retains the current chunk and lets the
/// bounded ring fill; exhausting that ring is fatal, never an overwrite/drop policy.
fn upload_worker(shared: Arc<Shared>, mut upload: UploadSender, meter: Option<LevelMeter>) {
    loop {
        if shared.control.state() >= CANCEL || shared.abandoned.load(Ordering::Acquire) {
            return;
        }
        let chunk = {
            let mut guard = shared.producer.lock().unwrap_or_else(|e| e.into_inner());
            let chunk = guard.ring.take_chunk();
            if chunk.is_none() && guard.ring.finished {
                return; // dropping the only producer is the HTTP request-body EOF
            }
            if chunk.is_none() {
                let _ = shared.wake.wait_timeout(guard, WORKER_TICK);
            }
            chunk
        };
        if let Some(meter) = &meter {
            meter.push(f32::from_bits(shared.peak.swap(0, Ordering::Relaxed)));
        }
        let Some(chunk) = chunk else { continue };
        let mut pending = Ok(chunk);
        loop {
            if shared.control.state() >= CANCEL || shared.abandoned.load(Ordering::Acquire) {
                return;
            }
            match upload.try_send(pending) {
                Ok(()) => break,
                Err(error) if error.is_full() => {
                    pending = error.into_inner();
                    let guard = shared.producer.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = shared.wake.wait_timeout(guard, WORKER_TICK);
                }
                Err(_) => {
                    shared.control.fail("Live PCM upload closed before capture completed".into());
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "live_capture_tests.rs"]
mod tests;
