//! One managed, full-duplex Audio8 live session. Live audio never reaches a file.
use crate::cmd::audio::AudioDevice;
use crate::live_capture::{Capture, Control, UploadSender, CANCEL, FAILED, RUNNING, STOP};
use crate::setup::ServerState;
use crate::transcript::Segment;
use eyre::{bail, eyre, Context, ContextCompat, Result};
use futures::{channel::mpsc, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Listener, Manager};
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

static ACTIVE: OnceLock<Mutex<Option<Arc<Control>>>> = OnceLock::new();

fn active() -> &'static Mutex<Option<Arc<Control>>> {
    ACTIVE.get_or_init(|| Mutex::new(None))
}

fn matches_session(payload: &str, id: &str) -> bool {
    let value = serde_json::from_str::<Value>(payload).unwrap_or(Value::Null);
    value
        .get("sessionId")
        .and_then(Value::as_str)
        .is_none_or(|target| target == id)
}

/// The main process should call this on ExitRequested/Exit, including tray quits.
/// The abort event also uses this path; dropping HTTP/capture belongs to the session task.
pub fn cancel_live_recording() {
    if let Some(control) = active().lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        control.cancel();
    }
}

struct Lease(Arc<Control>);
impl Lease {
    fn acquire() -> Result<Self> {
        let mut active = active().lock().map_err(|_| eyre!("Live session lock failed"))?;
        if active.is_some() {
            bail!("A live recording is already in progress");
        }
        let control = Arc::new(Control::default());
        *active = Some(control.clone());
        Ok(Self(control))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut active = active().lock().unwrap_or_else(|e| e.into_inner());
        if active.as_ref().is_some_and(|control| Arc::ptr_eq(control, &self.0)) {
            *active = None;
        }
    }
}

struct Session {
    app: AppHandle,
    id: String,
    name: String,
    lease: Lease,
    listeners: Vec<tauri::EventId>,
    capture: Option<Capture>,
    prompted: bool,
}
impl Session {
    fn emit(&self, kind: &str, mut payload: Value) {
        let object = payload.as_object_mut().expect("live event object");
        object.insert("type".into(), json!(kind));
        object.insert("sessionId".into(), json!(self.id));
        object.insert("name".into(), json!(self.name));
        let _ = self.app.emit("live_transcription", payload);
    }
    fn stopped_prompt(&mut self) {
        if self.prompted {
            crate::meeting_prompt::recording_stopped(&self.app);
            self.prompted = false;
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        self.lease.0.abort_upload();
        self.capture.take();
        self.stopped_prompt();
        for listener in self.listeners.drain(..) {
            self.app.unlisten(listener);
        }
    }
}

#[derive(Debug, Deserialize)]
struct Profile {
    window_tokens: usize,
    pcm_capacity_samples: usize,
    step_seconds: f64,
    realtime: bool,
}
impl Profile {
    fn validate(&self) -> Result<()> {
        if !self.realtime {
            bail!("Audio8 backend is too slow for realtime transcription on this machine; live capture was not started");
        }
        if !(14..=256).contains(&self.window_tokens)
            || !(2560..=81_920).contains(&self.pcm_capacity_samples)
            || self.pcm_capacity_samples % 2560 != 0
            || !self.step_seconds.is_finite()
            || self.step_seconds <= 0.0
            || self.step_seconds > 0.128
        {
            bail!("Server returned an invalid live calibration profile");
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct LiveSegment {
    start: i64,
    end: i64,
    text: String,
    #[serde(default)]
    speaker: Option<i32>,
}
impl LiveSegment {
    fn into_segment(self) -> Result<Segment> {
        if self.start < 0 || self.end < self.start {
            bail!("Server returned invalid live segment centisecond bounds");
        }
        Ok(Segment {
            start: self.start,
            stop: self.end,
            text: self.text,
            speaker: self.speaker,
        })
    }
}
struct ModelLoad {
    path: String,
    gpu_device: Option<i32>,
    no_gpu: Option<bool>,
    unload_timeout_minutes: u32,
}

// load_model contains synchronous sidecar startup. Isolate that poll so its startup wait
// cannot block the live session's stop select. Aborting at the next await drops its HTTP.
struct LoadingTask(tauri::async_runtime::JoinHandle<Result<String>>);
impl Drop for LoadingTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Returns immediately after installing stop/abort listeners. Optional Audio8 model loading
/// runs in the managed task, so a stop during startup never misses its listener or opens CPAL.
#[tauri::command]
pub async fn start_live_recording(
    app: AppHandle,
    devices: Vec<AudioDevice>,
    recording_name: Option<String>,
    language: String,
    threads: Option<i32>,
    model_path: Option<String>,
    gpu_device: Option<i32>,
    no_gpu: Option<bool>,
    unload_timeout_minutes: Option<u32>,
) -> crate::error::CommandResult<String> {
    if devices.is_empty() {
        return Err(eyre!("At least one audio device is required for live recording").into());
    }
    if language.trim().is_empty() || threads.is_some_and(|n| n <= 0) {
        return Err(eyre!("Live language must not be empty and thread count must be positive").into());
    }
    let lease = Lease::acquire()?;
    let session_id = uuid::Uuid::new_v4().to_string();
    let control = lease.0.clone();
    let stop = control.clone();
    let stop_id = session_id.clone();
    // FIRST: stop/abort may target this session; legacy global events still work.
    let stop_listener = app.listen("stop_record", move |event| {
        if matches_session(event.payload(), &stop_id) {
            stop.stop();
        }
    });
    let abort = control.clone();
    let abort_id = session_id.clone();
    let abort_listener = app.listen("abort_transcribe", move |event| {
        if matches_session(event.payload(), &abort_id) {
            abort.cancel();
        }
    });
    let close = control.clone();
    let close_listener = app.listen("closeapp", move |_| close.cancel());
    let name = recording_name
        .as_deref()
        .map(crate::cmd::files::sanitize_filename_stem)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(crate::ffmpeg::get_local_time);
    let mut session = Session {
        app,
        id: session_id,
        name,
        lease,
        listeners: vec![stop_listener, abort_listener, close_listener],
        capture: None,
        prompted: false,
    };
    session.emit("loading", json!({}));
    let session_id = session.id.clone();
    tauri::async_runtime::spawn(async move {
        let model = model_path.map(|path| ModelLoad {
            path,
            gpu_device,
            no_gpu,
            unload_timeout_minutes: unload_timeout_minutes.unwrap_or(5),
        });
        let result = run(&mut session, devices, language, threads, model).await;
        if !matches!(&result, Ok(Some(_))) {
            session.lease.0.abort_upload();
        }
        // Tear down capture before reporting an error/cancellation to the frontend.
        session.capture.take();
        session.stopped_prompt();
        match result {
            Ok(Some((segments, processing_time))) => session.emit(
                "finished",
                json!({
                    "segments": segments, "processingTimeSec": processing_time,
                }),
            ),
            Ok(None) => session.emit("cancelled", json!({})),
            Err(_) if session.lease.0.state() == CANCEL => session.emit("cancelled", json!({})),
            Err(error) => session.emit("error", json!({ "message": error.to_string() })),
        }
        // No record_finish: the live server already performed transcription.
    });
    Ok(session_id)
}

async fn open_response(
    client: &reqwest::Client,
    base_url: &str,
    language: &str,
    threads: Option<i32>,
    upload: mpsc::Receiver<std::result::Result<bytes::Bytes, std::io::Error>>,
    control: Arc<Control>,
) -> Result<reqwest::Response> {
    let mut url = reqwest::Url::parse(&format!("{}/v1/audio/live", base_url.trim_end_matches('/')))?;
    url.query_pairs_mut().append_pair("language", language);
    if let Some(threads) = threads {
        url.query_pairs_mut().append_pair("n_threads", &threads.to_string());
    }
    // A body error cancels hyper's request even if send() was dropped while waiting for
    // response headers. EOF is reserved for a graceful stop and server tail flush.
    let body = futures::stream::unfold((upload, control, false), |(mut upload, control, done)| async move {
        if done {
            return None;
        }
        let next = tokio::select! {
            biased;
            _ = control.upload_cancelled() => Some(Err(std::io::Error::new(std::io::ErrorKind::Interrupted, "Live upload cancelled"))),
            next = upload.next() => next,
        };
        next.map(|item| {
            let done = item.is_err();
            (item, (upload, control, done))
        })
    });
    let response = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .header(reqwest::header::ACCEPT, "application/x-ndjson")
        .body(reqwest::Body::wrap_stream(body))
        .send()
        .await
        .context("Failed to start live Audio8 HTTP stream")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("Live Audio8 request failed ({status}): {body}");
    }
    Ok(response)
}

async fn run(
    session: &mut Session,
    devices: Vec<AudioDevice>,
    language: String,
    threads: Option<i32>,
    model: Option<ModelLoad>,
) -> Result<Option<(Vec<Segment>, u64)>> {
    let control = session.lease.0.clone();
    if control.state() != RUNNING {
        return Ok(None);
    }
    if let Some(model) = model {
        let app = session.app.clone();
        let mut loading = LoadingTask(tauri::async_runtime::spawn(async move {
            crate::cmd::server_cmd::load_model(app, model.path, model.gpu_device, model.unload_timeout_minutes, model.no_gpu)
                .await
                .map_err(crate::error::Error::into_report)
        }));
        tokio::select! {
            biased;
            _ = control.changed.notified() => return Ok(None),
            result = &mut loading.0 => { result.context("Live model loading task failed")??; }
        }
        if control.state() != RUNNING {
            return Ok(None);
        }
    }
    let endpoint = session.app.state::<tokio::sync::Mutex<ServerState>>();
    let (client, base_url) = tokio::select! {
        biased;
        _ = control.changed.notified() => return Ok(None),
        state = endpoint.lock() => {
            let process = state.process.as_ref().context("Please load the Audio8 model before starting live recording")?;
            (process.client(), process.base_url())
        }
    };
    let (upload, receiver) = mpsc::channel(2);
    // Keep the sole producer here until ready: a body EOF during calibration would end the job.
    let mut upload: Option<UploadSender> = Some(upload);
    let response = tokio::select! {
        biased;
        _ = control.changed.notified() => return Ok(None),
        response = open_response(&client, &base_url, &language, threads, receiver, control.clone()) => response?,
    };
    let bytes = response.bytes_stream().map(|result| result.map_err(std::io::Error::other));
    let reader = StreamReader::new(bytes);
    let mut lines = FramedRead::new(reader, LinesCodec::new_with_max_length(4 * 1024 * 1024));
    let mut profile: Option<Profile> = None;
    let mut devices = Some(devices);
    let mut ready = false;
    let mut stopped = false;
    let mut segments = Vec::new();
    let mut started = Instant::now();
    loop {
        match control.state() {
            CANCEL => return Ok(None),
            FAILED => bail!("{}", control.error()),
            STOP if !ready => return Ok(None), // stop during calibration never opens CPAL
            STOP if !stopped => {
                session.capture.as_mut().context("Live capture missing at stop")?.finish()?;
                session.capture.take();
                session.stopped_prompt();
                stopped = true;
            }
            _ => {}
        }
        let line = tokio::select! {
            biased;
            _ = control.changed.notified() => continue,
            line = lines.next() => line.context("Live Audio8 stream ended without a final result")??,
        };
        if line.trim().is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(&line).context("Invalid live Audio8 NDJSON event")?;
        match event.get("type").and_then(Value::as_str).context("Live event missing type")? {
            "loading" => session.emit("loading", event.clone()),
            "calibration" => {
                if profile.is_some() || ready {
                    bail!("Unexpected duplicate live calibration");
                }
                let calibrated: Profile = serde_json::from_value(event.get("profile").unwrap_or(&event).clone())
                    .context("Invalid live calibration profile")?;
                session.emit("calibration", event.clone());
                calibrated.validate()?;
                profile = Some(calibrated);
            }
            "ready" => {
                if ready {
                    bail!("Unexpected duplicate live ready event");
                }
                let calibrated = profile.as_ref().context("Live server sent ready before calibration")?;
                calibrated.validate()?;
                if control.state() != RUNNING {
                    return Ok(None);
                }
                started = Instant::now();
                session.capture = Some(Capture::start(
                    session.app.clone(),
                    devices.take().context("Live devices already consumed")?,
                    calibrated.pcm_capacity_samples,
                    upload.take().context("Live upload already consumed")?,
                    control.clone(),
                )?);
                crate::meeting_prompt::recording_started(&session.app);
                session.prompted = true;
                ready = true;
                session.emit("ready", json!({}));
            }
            "partial" => {
                if !ready {
                    bail!("Live server returned text before capture was ready");
                }
                event
                    .get("text")
                    .and_then(Value::as_str)
                    .context("Live partial missing text")?;
                // Text is cumulative only within segment_index, never the whole session.
                session.emit("partial", event.clone());
            }
            "segment" => {
                if !ready {
                    bail!("Live server returned text before capture was ready");
                }
                let segment = serde_json::from_value::<LiveSegment>(event.clone())?.into_segment()?;
                session.emit(
                    "segment",
                    json!({ "segment": segment, "segment_index": event.get("segment_index") }),
                );
                segments.push(segment);
            }
            "result" => {
                if !ready || !stopped {
                    bail!("Live server finished before the recording was stopped");
                }
                let count = event
                    .get("segments")
                    .and_then(Value::as_u64)
                    .context("Live result missing segment count")?;
                if count != segments.len() as u64 {
                    bail!("Live result segment count does not match the received segments");
                }
                let elapsed = event
                    .get("processing_time_sec")
                    .and_then(Value::as_u64)
                    .unwrap_or_else(|| started.elapsed().as_secs());
                return Ok(Some((segments, elapsed)));
            }
            "error" => bail!(
                "{}",
                event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Live Audio8 server failed")
            ),
            other => bail!("Unknown live Audio8 event type: {other}"),
        }
    }
}

#[cfg(test)]
#[path = "live_tests.rs"]
mod tests;
