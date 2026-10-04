//! Duplex live protocol. This child module can reuse the private process pipes.
use super::{Event, Model, StreamCallbacks};
use anyhow::{bail, Context as _};
use std::{
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Duration,
};
use tokio::sync::mpsc::{error::TryRecvError, Receiver};

pub(crate) const LIVE_FRAME_BYTES: usize = 2560 * 4;

impl Model {
    /// The receiver and producer are bounded. Capture must wait for `ready`.
    pub(crate) fn transcribe_live(
        &mut self,
        language: &str,
        threads: usize,
        mut pcm: Receiver<Result<bytes::Bytes, String>>,
        mut emit: impl FnMut(serde_json::Value) -> anyhow::Result<()>,
        mut should_abort: impl FnMut() -> bool + Send,
    ) -> anyhow::Result<()> {
        if language != "en" && language != "zh" {
            bail!("Audio8 live requires language en or zh");
        }
        let mut callbacks = StreamCallbacks {
            should_abort: Some(Box::new(&mut should_abort)),
            ..StreamCallbacks::default()
        };
        self.ensure_started(&mut callbacks)?;
        drop(callbacks);
        if let Err(err) = emit(serde_json::json!({"type":"loading", "stage":"calibrating"})) {
            self.abort_runtime();
            return Err(err);
        }
        let result = self.begin_live(language, threads, &mut emit, &mut should_abort);
        if result.is_err() {
            self.abort_runtime();
        }
        result?;
        let mut input = self.input.take().context("Audio8 stdin is unavailable")?;
        let failed = Arc::new(AtomicBool::new(false));
        let writer_failed = Arc::clone(&failed);
        thread::scope(|scope| {
            let writer = scope.spawn(move || {
                let result = (|| loop {
                    if writer_failed.load(Ordering::SeqCst) {
                        bail!("Audio8 live upload aborted");
                    }
                    match pcm.try_recv() {
                        Ok(Ok(bytes)) => {
                            if bytes.is_empty() || bytes.len() > LIVE_FRAME_BYTES || bytes.len() % 4 != 0 {
                                bail!("Invalid Audio8 live PCM frame");
                            }
                            write_control(
                                &mut input,
                                serde_json::json!({
                                    "type":"stream_chunk", "sample_count":bytes.len() / 4
                                }),
                            )?;
                            input.write_all(&bytes)?;
                            input.flush()?;
                        }
                        Ok(Err(message)) => bail!("{message}"),
                        Err(TryRecvError::Empty) => thread::sleep(Duration::from_millis(10)),
                        Err(TryRecvError::Disconnected) => {
                            write_control(&mut input, serde_json::json!({"type":"stream_end"}))?;
                            return Ok(());
                        }
                    }
                })();
                if result.is_err() {
                    writer_failed.store(true, Ordering::SeqCst);
                }
                (input, result)
            });
            let result = self.receive_live(&mut emit, &mut || failed.load(Ordering::SeqCst) || should_abort());
            if result.is_err() {
                failed.store(true, Ordering::SeqCst);
                self.abort_runtime(); // also unblocks a pipe writer
            }
            let (input, written) = writer.join().map_err(|_| anyhow::anyhow!("Audio8 live writer failed"))?;
            // Prefer an explicit upload/overload error over its cancellation consequence.
            if let Err(err) = written {
                self.abort_runtime();
                return Err(err);
            }
            result?;
            self.input = Some(input);
            Ok(())
        })
    }

    fn begin_live(
        &mut self,
        language: &str,
        threads: usize,
        emit: &mut impl FnMut(serde_json::Value) -> anyhow::Result<()>,
        abort: &mut impl FnMut() -> bool,
    ) -> anyhow::Result<()> {
        write_control(
            self.input.as_mut().context("Audio8 stdin is unavailable")?,
            serde_json::json!({"type":"stream_start", "language":language, "threads":threads}),
        )?;
        let mut calibrated = false;
        loop {
            match self.live_event(abort)? {
                Event::Calibration { mut details } => {
                    if calibrated {
                        bail!("Repeated Audio8 calibration event");
                    }
                    calibrated = details.get("realtime").and_then(|v| v.as_bool()) == Some(true);
                    details.insert("type".into(), "calibration".into());
                    emit(serde_json::Value::Object(details))?;
                }
                Event::StreamReady {} if calibrated => {
                    emit(serde_json::json!({"type":"ready"}))?;
                    return Ok(());
                }
                Event::Error { message } => bail!("{message}"),
                _ => bail!("Invalid Audio8 live startup event"),
            }
        }
    }

    fn receive_live(
        &mut self,
        emit: &mut impl FnMut(serde_json::Value) -> anyhow::Result<()>,
        abort: &mut impl FnMut() -> bool,
    ) -> anyhow::Result<()> {
        let mut count = 0;
        let mut last_end = 0;
        loop {
            let event = match self.live_event(abort)? {
                Event::Partial { text, segment_index } if segment_index == count => {
                    serde_json::json!({"type":"partial", "text":text, "segment_index":segment_index})
                }
                Event::Segment { start, end, text } => {
                    if start < last_end || end < start {
                        bail!("Invalid Audio8 live segment bounds");
                    }
                    last_end = end;
                    let event = serde_json::json!({"type":"segment", "start":start,
                        "end":end, "text":text, "segment_index":count});
                    count += 1;
                    event
                }
                Event::Result { segments } if segments == count => {
                    emit(serde_json::json!({"type":"result", "segments":count}))?;
                    return Ok(());
                }
                Event::Error { message } => bail!("{message}"),
                _ => bail!("Invalid Audio8 live protocol event"),
            };
            emit(event)?;
        }
    }

    fn live_event(&self, abort: &mut impl FnMut() -> bool) -> anyhow::Result<Event> {
        loop {
            if abort() {
                bail!("Audio8 live transcription aborted");
            }
            match self
                .events
                .as_ref()
                .context("Audio8 event pipe unavailable")?
                .recv_timeout(Duration::from_millis(100))
            {
                Ok(event) => return event.map_err(anyhow::Error::msg),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("Audio8 live worker exited; check device memory"),
            }
        }
    }
}

#[cfg(all(test, unix))]
#[path = "audio8_live_tests.rs"]
mod tests;

fn write_control(input: &mut impl Write, value: serde_json::Value) -> anyhow::Result<()> {
    serde_json::to_writer(&mut *input, &value)?;
    input.write_all(b"\n")?;
    input.flush()?;
    Ok(())
}
