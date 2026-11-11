//! Stored-recording formats, and the live encoder that turns a disk recording into its final
//! file while it is still being captured.
//!
//! Encoding after stop scaled with the recording: two full ffmpeg passes over gigabytes of
//! float WAV (seconds per hour), and a lossy codec on top of that would add ~20 s per hour of
//! audio. Here the capture callbacks mix into a short timeline buffer that a pump thread feeds
//! to one ffmpeg process, so stopping only flushes the last fraction of a second. The raw WAV
//! captures are still written beside it: if the live encoder fails for any reason, the
//! post-stop transcode in `cmd::audio` rebuilds the recording from them.
use cpal::{FromSample, Sample};
use eyre::{bail, eyre, Result};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// File format a disk recording is stored in. Unknown or missing values mean the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecordingFormat {
    /// Ogg Opus, 48 kHz mono, 48 kbps VBR: ~22 MB per hour, transparent for speech.
    #[default]
    Opus,
    /// 16 kHz mono PCM, what Vibe stored before formats existed: ~115 MB per hour.
    Wav,
    /// 16 kHz mono lossless.
    Flac,
    /// 48 kHz mono 64 kbps.
    Mp3,
    /// AAC in MP4, 48 kHz mono 64 kbps.
    M4a,
}

impl RecordingFormat {
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("wav") => Self::Wav,
            Some("flac") => Self::Flac,
            Some("mp3") => Self::Mp3,
            Some("m4a") => Self::M4a,
            _ => Self::Opus,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::M4a => "m4a",
        }
    }

    /// ffmpeg output options: rate, channels, codec and muxer. The muxer is explicit so the
    /// output never depends on how a given ffmpeg build maps file extensions.
    pub fn output_args(self) -> &'static [&'static str] {
        match self {
            Self::Opus => &[
                "-ar", "48000", "-ac", "1", "-c:a", "libopus", "-b:a", "48k", "-vbr", "on", "-f", "ogg",
            ],
            Self::Wav => &["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav"],
            Self::Flac => &["-ar", "16000", "-ac", "1", "-c:a", "flac", "-f", "flac"],
            Self::Mp3 => &["-ar", "48000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
            Self::M4a => &["-ar", "48000", "-ac", "1", "-c:a", "aac", "-b:a", "64k", "-f", "mp4"],
        }
    }
}

/// Mixes N capture streams into one mono timeline, aligned by sample index like ffmpeg's
/// `amix` did on the finished WAVs, and releases the prefix every stream has passed.
///
/// A stream that falls more than `max_lag` frames behind the leading one (WASAPI loopback
/// delivers nothing while nothing plays; a device may start late) stops holding the output
/// back: its gap is released as silence, and its next buffer is placed at the leading edge.
pub(crate) struct TimelineMixer {
    /// Absolute frame index of `frames[0]`.
    base: u64,
    frames: VecDeque<f32>,
    /// Next absolute frame each source writes to.
    positions: Vec<u64>,
    /// Sources that were passed over; their next buffer re-anchors at the leading edge.
    stalled: Vec<bool>,
    max_lag: u64,
    max_buffered: usize,
}

impl TimelineMixer {
    pub(crate) fn new(sources: usize, max_lag: u64, max_buffered: usize) -> Self {
        Self {
            base: 0,
            frames: VecDeque::new(),
            positions: vec![0; sources],
            stalled: vec![false; sources],
            max_lag,
            max_buffered,
        }
    }

    fn lead(&self) -> u64 {
        self.base + self.frames.len() as u64
    }

    /// Add one interleaved buffer from `source`, downmixed to mono by averaging its channels.
    pub(crate) fn push<T>(&mut self, source: usize, channels: usize, data: &[T]) -> Result<()>
    where
        T: Sample,
        f32: FromSample<T>,
    {
        if channels == 0 || source >= self.positions.len() {
            bail!("invalid live mixer input");
        }
        let count = (data.len() / channels) as u64;
        if self.stalled[source] {
            self.positions[source] = self.lead().saturating_sub(count).max(self.base);
            self.stalled[source] = false;
        }
        for frame in data.chunks_exact(channels) {
            let sum: f32 = frame
                .iter()
                .map(|sample| {
                    let value = f32::from_sample(*sample);
                    if value.is_finite() {
                        value
                    } else {
                        0.0
                    }
                })
                .sum();
            let mono = sum / channels as f32;
            // Positions never trail `base` (take_ready lifts them) nor pass the lead.
            let index = (self.positions[source] - self.base) as usize;
            if index == self.frames.len() {
                self.frames.push_back(mono);
            } else {
                self.frames[index] += mono;
            }
            self.positions[source] += 1;
        }
        if self.frames.len() > self.max_buffered {
            bail!("live recording encoder fell behind");
        }
        Ok(())
    }

    /// Move every frame that all sources have passed (or that lagging ones gave up) to `out`.
    pub(crate) fn take_ready(&mut self, out: &mut Vec<f32>) {
        let lead = self.lead();
        let slowest = self.positions.iter().copied().min().unwrap_or(lead);
        let ready = slowest.max(lead.saturating_sub(self.max_lag)).min(lead);
        out.extend(self.frames.drain(..(ready - self.base) as usize));
        self.base = ready;
        for (position, stalled) in self.positions.iter_mut().zip(self.stalled.iter_mut()) {
            if *position < ready {
                *position = ready;
                *stalled = true;
            }
        }
    }

    /// Release everything; shorter sources end in silence (`amix=duration=longest`).
    pub(crate) fn take_all(&mut self, out: &mut Vec<f32>) {
        out.extend(self.frames.drain(..));
        self.base = self.positions.iter().copied().max().unwrap_or(self.base).max(self.base);
    }
}

/// How often the pump moves mixed audio into ffmpeg.
const PUMP_INTERVAL: Duration = Duration::from_millis(100);
/// A source this far behind the leading one no longer holds the output back.
const MAX_LAG_SECONDS: u64 = 2;
/// Buffered audio beyond this means ffmpeg stopped reading; give up and use the WAVs.
const MAX_BUFFERED_SECONDS: usize = 60;
/// Upper bound for ffmpeg to flush once its input closed. Normally well under a second.
const FINISH_TIMEOUT: Duration = Duration::from_secs(60);

/// What a capture callback holds: pushes its buffers into the shared mixer.
#[derive(Clone)]
pub(crate) struct LiveSink {
    mixer: Arc<Mutex<TimelineMixer>>,
    failed: Arc<AtomicBool>,
    source: usize,
    channels: usize,
}

impl LiveSink {
    pub(crate) fn push<T>(&self, data: &[T])
    where
        T: Sample,
        f32: FromSample<T>,
    {
        if self.failed.load(Ordering::Relaxed) {
            return;
        }
        let mut mixer = self.mixer.lock().unwrap_or_else(|e| e.into_inner());
        if mixer.push(self.source, self.channels, data).is_err() {
            self.failed.store(true, Ordering::Relaxed);
            mixer.frames.clear();
        }
    }
}

pub(crate) struct LiveEncoder {
    mixer: Arc<Mutex<TimelineMixer>>,
    failed: Arc<AtomicBool>,
    channels: Vec<usize>,
    child: Arc<Mutex<Child>>,
    stop: mpsc::Sender<()>,
    result: mpsc::Receiver<Result<()>>,
    output: PathBuf,
}

impl LiveEncoder {
    /// Spawn ffmpeg for `channels.len()` sources that all deliver `rate` Hz.
    pub(crate) fn start(rate: u32, channels: &[u16], format: RecordingFormat, output: PathBuf) -> Result<Self> {
        if rate == 0 || channels.is_empty() || channels.contains(&0) {
            bail!("invalid live encoder configuration");
        }
        let mut child = crate::ffmpeg::spawn_live_encoder(rate, channels.len() > 1, format.output_args(), &output)?;
        let stdin = child.stdin.take().ok_or_else(|| eyre!("ffmpeg stdin unavailable"))?;
        let stderr = child.stderr.take();
        let mixer = Arc::new(Mutex::new(TimelineMixer::new(
            channels.len(),
            rate as u64 * MAX_LAG_SECONDS,
            rate as usize * MAX_BUFFERED_SECONDS,
        )));
        let failed = Arc::new(AtomicBool::new(false));
        let child = Arc::new(Mutex::new(child));
        let (stop, stop_rx) = mpsc::channel();
        let (result_tx, result) = mpsc::channel();
        let pump = Pump {
            mixer: mixer.clone(),
            failed: failed.clone(),
            child: child.clone(),
            output: output.clone(),
        };
        std::thread::Builder::new().name("record-encode".into()).spawn(move || {
            let outcome = pump.run(stdin, stderr, stop_rx);
            if outcome.is_err() {
                pump.failed.store(true, Ordering::Relaxed);
            }
            let _ = result_tx.send(outcome);
        })?;
        Ok(Self {
            mixer,
            failed,
            channels: channels.iter().map(|c| *c as usize).collect(),
            child,
            stop,
            result,
            output,
        })
    }

    pub(crate) fn sink(&self, source: usize) -> LiveSink {
        LiveSink {
            mixer: self.mixer.clone(),
            failed: self.failed.clone(),
            source,
            channels: self.channels[source],
        }
    }

    /// Flush the remaining audio and wait for ffmpeg; the capture streams must be stopped.
    /// On failure the partial output is removed and the caller falls back to the WAVs.
    pub(crate) fn finish(self) -> Result<PathBuf> {
        let _ = self.stop.send(());
        let outcome = match self.result.recv_timeout(FINISH_TIMEOUT) {
            Ok(outcome) => outcome,
            Err(_) => {
                self.child.lock().unwrap_or_else(|e| e.into_inner()).kill().ok();
                Err(eyre!("live encoder did not finish in time"))
            }
        };
        match outcome {
            Ok(()) => Ok(self.output),
            Err(error) => {
                std::fs::remove_file(&self.output).ok();
                Err(error)
            }
        }
    }
}

struct Pump {
    mixer: Arc<Mutex<TimelineMixer>>,
    failed: Arc<AtomicBool>,
    child: Arc<Mutex<Child>>,
    output: PathBuf,
}

impl Pump {
    fn run(&self, stdin: ChildStdin, stderr: Option<std::process::ChildStderr>, stop: mpsc::Receiver<()>) -> Result<()> {
        // Drained concurrently: a full stderr pipe would stall ffmpeg and, through it, the pump.
        let stderr = stderr.map(|mut pipe| {
            std::thread::spawn(move || {
                let mut text = String::new();
                let _ = (&mut pipe).take(4000).read_to_string(&mut text);
                let _ = std::io::copy(&mut pipe, &mut std::io::sink());
                text
            })
        });
        let pumped = self.pump(stdin, stop);
        if pumped.is_err() {
            self.child.lock().unwrap_or_else(|e| e.into_inner()).kill().ok();
        }
        let status = self.wait()?;
        let stderr = stderr.and_then(|handle| handle.join().ok()).unwrap_or_default();
        pumped?;
        if !status.success() {
            bail!("ffmpeg live encoder failed ({status}): {}", stderr.trim());
        }
        if std::fs::metadata(&self.output).map(|m| m.len()).unwrap_or(0) == 0 {
            bail!("ffmpeg live encoder produced no output");
        }
        Ok(())
    }

    /// Stdin is dropped on return, which is ffmpeg's end of input.
    fn pump(&self, mut stdin: ChildStdin, stop: mpsc::Receiver<()>) -> Result<()> {
        let mut samples = Vec::new();
        let mut bytes = Vec::new();
        loop {
            let stopping = !matches!(stop.recv_timeout(PUMP_INTERVAL), Err(mpsc::RecvTimeoutError::Timeout));
            if self.failed.load(Ordering::Relaxed) {
                bail!("live recording encoder fell behind");
            }
            {
                let mut mixer = self.mixer.lock().unwrap_or_else(|e| e.into_inner());
                if stopping {
                    mixer.take_all(&mut samples);
                } else {
                    mixer.take_ready(&mut samples);
                }
            }
            if !samples.is_empty() {
                bytes.clear();
                bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
                samples.clear();
                stdin.write_all(&bytes)?;
            }
            if stopping {
                return Ok(());
            }
        }
    }

    fn wait(&self) -> Result<std::process::ExitStatus> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.lock().unwrap_or_else(|e| e.into_inner()).try_wait()? {
                return Ok(status);
            }
            if started.elapsed() > FINISH_TIMEOUT {
                self.child.lock().unwrap_or_else(|e| e.into_inner()).kill().ok();
                bail!("ffmpeg live encoder did not exit");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Removes a file unless it is `keep`; used for the capture intermediates.
pub(crate) fn remove_except(paths: impl IntoIterator<Item = PathBuf>, keep: &Path) {
    for path in paths {
        if path != keep && path.exists() {
            if let Err(error) = std::fs::remove_file(&path) {
                tracing::error!("failed to remove recording intermediate {}: {error}", path.display());
            }
        }
    }
}

#[cfg(test)]
#[path = "recording_encoder_tests.rs"]
mod tests;
