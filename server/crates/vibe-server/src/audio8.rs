//! RAM-only Audio8 subprocess: 24s file chunks or bounded PCM/rolling-KV live inference.
//! Segment times are coarse emission-clock bounds, not aligned timestamps.
#[path = "audio8_backends.rs"]
mod backends;
#[path = "audio8_live.rs"]
mod live;
use anyhow::{bail, Context as _};
use serde::Deserialize;
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use whisper_rs::{ContextOptions, Segment, StreamCallbacks, TranscribeOptions, TranscribeResult};

pub const CHECKPOINT: &str = "Edge0/Audio8-ASR-Infinite";
pub const REVISION: &str = "7476824bc222e4ad509d286e8cae8b8d3f371129";
const SCRIPT: &str = include_str!("audio8_runtime.py");
const MAX_EVENT_BYTES: usize = 2 * 1024 * 1024;
const MAX_DIAGNOSTIC_BYTES: usize = 16 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    engine: String,
    checkpoint: String,
    revision: String,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    artifacts: Option<String>,
}

pub fn validate_manifest(path: &str) -> anyhow::Result<()> {
    if !path.ends_with(".audio8.json") {
        bail!("Audio8 model marker must end in .audio8.json");
    }
    let file = std::fs::File::open(path).context("Could not open Audio8 model marker")?;
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes)?;
    if bytes.len() > 65536 {
        bail!("Audio8 model marker is too large");
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).context("Invalid Audio8 model marker JSON")?;
    if manifest.engine != "audio8" || manifest.checkpoint != CHECKPOINT || manifest.revision != REVISION {
        bail!("Audio8 marker must select {CHECKPOINT} at pinned revision {REVISION}");
    }
    backends::validate_selection(
        manifest.backend.as_deref(),
        manifest.artifacts.as_deref(),
        std::path::Path::new(path),
    )
}

pub fn capabilities() -> crate::engine::EngineCapabilities {
    crate::engine::EngineCapabilities {
        engine: "audio8".to_owned(),
        requires_vad: false,
        languages: vec!["en".to_owned(), "zh".to_owned()],
        language_detection: false,
        streaming: true,
        translation: false,
        timestamps: false,
        text_prompts: false,
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    Ready {},
    Progress {
        value: i32,
    },
    Segment {
        start: i64,
        end: i64,
        text: String,
    },
    Result {
        segments: usize,
    },
    Error {
        message: String,
    },
    Calibration {
        #[serde(flatten)]
        details: serde_json::Map<String, serde_json::Value>,
    },
    StreamReady {},
    Partial {
        text: String,
        segment_index: usize,
    },
}

type Events = mpsc::Receiver<Result<Event, String>>;

pub struct Model {
    manifest: PathBuf,
    backend: String,
    options: ContextOptions,
    child: Option<Child>,
    input: Option<ChildStdin>,
    events: Option<Events>,
    diagnostics: Arc<Mutex<Vec<u8>>>,
    readers: Vec<thread::JoinHandle<()>>,
}

impl Model {
    pub fn load(path: &str, options: ContextOptions) -> anyhow::Result<Self> {
        validate_manifest(path)?;
        let marker: Manifest = serde_json::from_reader(std::fs::File::open(path)?)?;
        // Only transcription owns cancellable startup; never download under the model mutex.
        Ok(Self {
            manifest: std::fs::canonicalize(path).context("Could not resolve Audio8 marker path")?,
            backend: marker.backend.unwrap_or_else(|| "torch".into()),
            options,
            child: None,
            input: None,
            events: None,
            diagnostics: Arc::new(Mutex::new(Vec::new())),
            readers: Vec::new(),
        })
    }

    pub(crate) fn ensure_started(&mut self, callbacks: &mut StreamCallbacks<'_>) -> anyhow::Result<()> {
        if callbacks.should_abort.as_mut().is_some_and(|callback| callback()) {
            self.stop();
            bail!("Audio8 transcription aborted; runtime will reload for the next request");
        }
        if let Some(child) = self.child.as_mut() {
            if child.try_wait()?.is_none() {
                return Ok(());
            }
            self.stop();
        }
        self.start(callbacks)
    }

    fn start(&mut self, callbacks: &mut StreamCallbacks<'_>) -> anyhow::Result<()> {
        // Never use an executable, script path, or dependency URL from a marker.
        let directory = self
            .manifest
            .parent()
            .context("Audio8 marker has no parent directory")?
            .join(".audio8-runtime");
        std::fs::create_dir_all(&directory).context("Could not create local Audio8 runtime directory")?;
        for (name, source) in [
            ("audio8_runtime.py", SCRIPT),
            ("audio8_streaming.py", include_str!("audio8_streaming.py")),
            ("audio8_stream_protocol.py", include_str!("audio8_stream_protocol.py")),
        ] {
            let mut embedded = tempfile::NamedTempFile::new_in(&directory)?;
            embedded.write_all(source.as_bytes())?;
            embedded
                .persist(directory.join(name))
                .map_err(|err| err.error)
                .context("Could not install embedded Audio8 runtime")?;
        }
        let script = backends::install(&directory, &self.backend)?;
        let mut command = if let Some(python) = std::env::var_os("VIBE_AUDIO8_PYTHON") {
            let mut command = Command::new(python);
            command.args(["-I", "-u"]).arg(&script);
            command
        } else {
            let mut command = Command::new("uv");
            command
                .args([
                    "run",
                    "--no-project",
                    "--no-config",
                    "--no-env-file",
                    "--python",
                    "3.12",
                    "--script",
                ])
                .arg(&script);
            command
        };
        command
            .arg("--manifest")
            .arg(&self.manifest)
            .arg("--gpu-device")
            .arg(self.options.gpu_device.to_string())
            .current_dir(&directory)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONSAFEPATH", "1")
            .env_remove("PYTHONPATH")
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if std::env::var_os("HF_HOME").is_none() {
            // Keep the normal HF CLI login while storing large checkpoints beside the marker.
            if std::env::var_os("HF_TOKEN_PATH").is_none() {
                let cache = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| {
                    std::env::var_os("HOME")
                        .or_else(|| std::env::var_os("USERPROFILE"))
                        .map(|home| PathBuf::from(home).join(".cache"))
                });
                if let Some(cache) = cache {
                    command.env("HF_TOKEN_PATH", cache.join("huggingface").join("token"));
                }
            }
            command.env("HF_HOME", directory.join("hf"));
        }
        if self.options.no_gpu {
            command.arg("--no-gpu");
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0); // uv and its Python child share a killable group.
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let mut child = command
            .spawn()
            .context("Could not start Audio8: install uv and Python 3.12, or set VIBE_AUDIO8_PYTHON to a prepared interpreter")?;
        self.input = child.stdin.take();
        self.child = Some(child);
        let child = self.child.as_mut().expect("child initialized");
        let stdout = child.stdout.take().expect("piped Audio8 stdout");
        let stderr = child.stderr.take().expect("piped Audio8 stderr");
        self.diagnostics = Arc::new(Mutex::new(Vec::new()));
        let diagnostics = Arc::clone(&self.diagnostics);
        self.readers
            .push(thread::spawn(move || collect_diagnostics(stderr, diagnostics)));
        let (events, reader) = read_events(stdout);
        self.events = Some(events);
        self.readers.push(reader);
        // Download/startup timeout only, not a transcription-duration limit.
        let seconds = std::env::var("VIBE_AUDIO8_STARTUP_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1800);
        let started = Instant::now();
        let ready = loop {
            if callbacks.should_abort.as_mut().is_some_and(|callback| callback()) {
                break Err(anyhow::anyhow!(
                    "Audio8 transcription aborted during startup; runtime will reload for the next request"
                ));
            }
            if started.elapsed() >= Duration::from_secs(seconds) {
                break Err(anyhow::anyhow!(
                    "Audio8 startup timed out; check model downloads and dependencies"
                ));
            }
            match self
                .events
                .as_ref()
                .expect("events initialized")
                .recv_timeout(Duration::from_millis(100))
            {
                Ok(Ok(Event::Ready {})) => break Ok(()),
                Ok(Ok(Event::Error { message })) => {
                    break Err(anyhow::anyhow!(message.chars().take(2000).collect::<String>()));
                }
                Ok(Ok(_)) => break Err(anyhow::anyhow!("Unexpected Audio8 startup protocol event")),
                Ok(Err(message)) => break Err(anyhow::anyhow!(message)),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(anyhow::anyhow!("Audio8 runtime exited before becoming ready"))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
            }
        };
        if let Err(error) = ready {
            self.stop();
            let count = self.diagnostics.lock().map(|bytes| bytes.len()).unwrap_or_default();
            if count == 0 {
                return Err(error);
            }
            // uv stderr may contain authenticated URLs or HF credentials. Keep
            // bounded capture, but never forward raw diagnostics to the client.
            let summary =
                format!("{error}; Audio8 startup captured {count} diagnostic bytes (contents suppressed to protect credentials)");
            return Err(error.context(summary));
        }
        Ok(())
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
        options: TranscribeOptions,
        mut callbacks: StreamCallbacks<'_>,
    ) -> anyhow::Result<TranscribeResult> {
        let language = validate_options(&options)?;
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("Audio8 samples must be finite 16kHz mono f32 audio");
        }
        if callbacks.should_abort.as_mut().is_some_and(|callback| callback()) {
            self.stop();
            bail!("Audio8 transcription aborted");
        }
        if samples.is_empty() {
            if let Some(callback) = callbacks.on_progress.as_mut() {
                callback(100);
            }
            return Ok(TranscribeResult::default());
        }
        self.ensure_started(&mut callbacks)?;
        let header = serde_json::to_vec(&serde_json::json!({
            "type": "transcribe", "sample_count": samples.len(),
            "language": language, "threads": options.threads.max(0),
        }))?;
        let mut input = self.input.take().context("Audio8 stdin pipe is unavailable")?;
        thread::scope(|scope| {
            // Writing and consuming events separately keeps cancellation responsive
            // even during a large upload. No copy of the complete audio is made.
            let writer = scope.spawn(move || {
                let result = write_request(&mut input, &header, samples);
                (input, result)
            });
            let result = self.receive_transcription(&mut callbacks, samples.len());
            if result.is_err() {
                self.stop(); // Closing the process also unblocks a pipe writer.
            }
            let (input, written) = writer.join().map_err(|_| anyhow::anyhow!("Audio8 input writer failed"))?;
            // Cancellation closes stdin, so a broken pipe is expected; retain
            // the meaningful inference/abort error rather than that consequence.
            let result = result?;
            if let Err(error) = written {
                self.stop();
                return Err(error.context("Could not send Audio8 audio frame"));
            }
            self.input = Some(input);
            Ok(result)
        })
    }

    fn receive_transcription(
        &mut self,
        callbacks: &mut StreamCallbacks<'_>,
        sample_count: usize,
    ) -> anyhow::Result<TranscribeResult> {
        let mut segments = Vec::new();
        let end_limit = sample_count.div_ceil(160) as i64;
        loop {
            if callbacks.should_abort.as_mut().is_some_and(|callback| callback()) {
                bail!("Audio8 transcription aborted; runtime will reload for the next request");
            }
            let event = match self
                .events
                .as_ref()
                .context("Audio8 event pipe is unavailable")?
                .recv_timeout(Duration::from_millis(100))
            {
                Ok(event) => event.map_err(anyhow::Error::msg)?,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    bail!("Audio8 runtime exited during transcription; check device memory")
                }
            };
            match event {
                Event::Progress { value } => {
                    if !(0..=100).contains(&value) {
                        bail!("Invalid Audio8 progress event");
                    }
                    if let Some(callback) = callbacks.on_progress.as_mut() {
                        callback(value);
                    }
                }
                Event::Segment { start, end, text } => {
                    if start < 0
                        || end < start
                        || end > end_limit
                        || segments.last().is_some_and(|previous: &Segment| previous.end > start)
                    {
                        bail!("Invalid Audio8 chunk-boundary event");
                    }
                    let segment = Segment {
                        start,
                        end,
                        text,
                        no_speech_prob: 0.0,
                        // No word timings from this engine.
                        words: Vec::new(),
                    };
                    if let Some(callback) = callbacks.on_segment.as_mut() {
                        callback(segment.clone());
                    }
                    segments.push(segment);
                }
                Event::Result { segments: count } => {
                    if count != segments.len() {
                        bail!("Audio8 result does not match received segments");
                    }
                    return Ok(TranscribeResult { segments });
                }
                Event::Error { message } => bail!("{message}"),
                _ => bail!("Unexpected Audio8 live/startup event during file transcription"),
            }
        }
    }

    pub fn abort_runtime(&mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        self.input.take();
        self.events.take(); // Unblock an event reader waiting on the bounded queue.
        if let Some(mut child) = self.child.take() {
            #[cfg(unix)]
            unsafe {
                // The group was created by us; also terminate uv's Python child.
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                let _ = Command::new("taskkill")
                    .args(["/PID", &child.id().to_string(), "/T", "/F"])
                    .creation_flags(0x08000000)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.abort_runtime();
    }
}

fn validate_options(options: &TranscribeOptions) -> anyhow::Result<&str> {
    if options.translate {
        bail!("Audio8 does not support translation");
    }
    if options.prompt.is_some() {
        bail!("Audio8 does not support text prompts");
    }
    if options.detect_language {
        bail!("Audio8 does not support language detection; select en or zh");
    }
    if options.word_timestamps || options.stable_timestamps {
        bail!("Audio8 does not provide aligned timestamps; only coarse chunk bounds are available");
    }
    let language = options.language.as_deref().unwrap_or("en");
    if language != "en" && language != "zh" {
        bail!("Audio8 supports only en and zh; select a language explicitly");
    }
    Ok(language)
}

fn write_request(input: &mut impl Write, header: &[u8], samples: &[f32]) -> anyhow::Result<()> {
    input.write_all(header)?;
    input.write_all(b"\n")?;
    let mut buffer = Vec::with_capacity(65536);
    for chunk in samples.chunks(16384) {
        buffer.clear();
        for sample in chunk {
            buffer.extend_from_slice(&sample.to_le_bytes());
        }
        input.write_all(&buffer)?;
    }
    input.flush()?;
    Ok(())
}

fn read_events(stdout: impl Read + Send + 'static) -> (Events, thread::JoinHandle<()>) {
    let (sender, receiver) = mpsc::sync_channel(32);
    let reader = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = Vec::new();
            match reader
                .by_ref()
                .take((MAX_EVENT_BYTES + 1) as u64)
                .read_until(b'\n', &mut line)
            {
                Ok(0) => break,
                Ok(_) if line.len() > MAX_EVENT_BYTES || !line.ends_with(b"\n") => {
                    let _ = sender.send(Err("Invalid Audio8 event frame".to_owned()));
                    break;
                }
                Ok(_) => {
                    let event = serde_json::from_slice(&line).map_err(|_| "Invalid Audio8 event JSON".to_owned());
                    let invalid = event.is_err();
                    if sender.send(event).is_err() || invalid {
                        break;
                    }
                }
                Err(_) => {
                    let _ = sender.send(Err("Could not read Audio8 runtime events".to_owned()));
                    break;
                }
            }
        }
    });
    (receiver, reader)
}

fn collect_diagnostics(mut stderr: impl Read, diagnostics: Arc<Mutex<Vec<u8>>>) {
    let mut buffer = [0u8; 4096];
    while let Ok(count) = stderr.read(&mut buffer) {
        if count == 0 {
            break;
        }
        if let Ok(mut bytes) = diagnostics.lock() {
            bytes.extend_from_slice(&buffer[..count]);
            let excess = bytes.len().saturating_sub(MAX_DIAGNOSTIC_BYTES);
            bytes.drain(..excess);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_pinned_marker_is_accepted() {
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let path = directory.path().join("test.audio8.json");
        std::fs::write(
            &path,
            serde_json::json!({"engine":"audio8", "checkpoint":CHECKPOINT, "revision":REVISION}).to_string(),
        )
        .unwrap();
        assert!(validate_manifest(path.to_str().unwrap()).is_ok());
        std::fs::write(
            &path,
            serde_json::json!({"engine":"audio8", "checkpoint":CHECKPOINT, "revision":REVISION, "executable":"untrusted"})
                .to_string(),
        )
        .unwrap();
        assert!(validate_manifest(path.to_str().unwrap()).is_err());
        std::fs::write(
            &path,
            serde_json::json!({"engine":"audio8", "checkpoint":CHECKPOINT, "revision":"main"}).to_string(),
        )
        .unwrap();
        assert!(validate_manifest(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn binary_audio_is_little_endian_and_not_json() {
        let header = br#"{"type":"transcribe","sample_count":2,"language":"en"}"#;
        let mut frame = Vec::new();
        write_request(&mut frame, header, &[0.5, -0.25]).unwrap();
        assert_eq!(&frame[..header.len()], header);
        assert_eq!(
            &frame[header.len() + 1..],
            &[0.5f32.to_le_bytes(), (-0.25f32).to_le_bytes()].concat()
        );
    }

    #[test]
    fn capabilities_and_options_are_honest() {
        let caps = capabilities();
        assert!(caps.streaming && !caps.timestamps && !caps.language_detection && !caps.requires_vad);
        assert_eq!(validate_options(&TranscribeOptions::default()).unwrap(), "en");
        for options in [
            TranscribeOptions {
                detect_language: true,
                ..Default::default()
            },
            TranscribeOptions {
                language: Some("auto".to_owned()),
                ..Default::default()
            },
            TranscribeOptions {
                translate: true,
                ..Default::default()
            },
            TranscribeOptions {
                word_timestamps: true,
                ..Default::default()
            },
            TranscribeOptions {
                prompt: Some("text".to_owned()),
                ..Default::default()
            },
        ] {
            assert!(validate_options(&options).is_err());
        }
    }

    #[test]
    fn malformed_events_do_not_expose_protocol_contents() {
        let (events, reader) = read_events(std::io::Cursor::new(b"private transcript\n"));
        assert_eq!(events.recv().unwrap().unwrap_err(), "Invalid Audio8 event JSON");
        reader.join().unwrap();
    }

    fn mock_worker(code: &str) -> Model {
        #[cfg(unix)]
        let _environment = super::cancellation_tests::environment_lock();
        let mut command = Command::new("uv");
        command
            .args(["run", "--no-project", "--python", "3.12", "python", "-u", "-c", code])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command.spawn().expect("uv/Python 3.12 is required for Audio8 protocol tests");
        let input = child.stdin.take();
        let (events, reader) = read_events(child.stdout.take().unwrap());
        Model {
            manifest: PathBuf::from("unused.audio8.json"),
            backend: "torch".into(),
            options: ContextOptions::default(),
            child: Some(child),
            input,
            events: Some(events),
            diagnostics: Arc::new(Mutex::new(Vec::new())),
            readers: vec![reader],
        }
    }

    #[cfg(unix)]
    #[test]
    fn persistent_subprocess_delivers_callbacks_and_results() {
        let code = r#"
import json,sys
print(json.dumps({'type':'ready'}),flush=True)
while True:
    line=sys.stdin.buffer.readline()
    if not line: break
    header=json.loads(line)
    size=header['sample_count']*4
    raw=sys.stdin.buffer.read(size)
    assert len(raw)==size
    for event in [{'type':'progress','value':0},
                  {'type':'segment','start':0,'end':1,'text':'decoded'},
                  {'type':'progress','value':100}, {'type':'result','segments':1}]:
        print(json.dumps(event),flush=True)
"#;
        let fixture = super::cancellation_tests::Fixture::new(code);
        let mut model = fixture.load();
        assert!(model.child.is_none());
        let mut pid = None;
        for _ in 0..2 {
            let progress = Arc::new(Mutex::new(Vec::new()));
            let observed = Arc::clone(&progress);
            let segments = Arc::new(Mutex::new(Vec::new()));
            let observed_segments = Arc::clone(&segments);
            let callbacks = StreamCallbacks {
                on_progress: Some(Box::new(move |value| observed.lock().unwrap().push(value))),
                on_segment: Some(Box::new(move |segment| observed_segments.lock().unwrap().push(segment))),
                should_abort: None,
            };
            let result = model
                .transcribe(&[0.5, -0.25], TranscribeOptions::default(), callbacks)
                .unwrap();
            assert_eq!(result.text(), "decoded");
            assert_eq!(*progress.lock().unwrap(), vec![0, 100]);
            assert_eq!(result.segments, *segments.lock().unwrap());
            let current_pid = model.child.as_ref().unwrap().id();
            assert_eq!(current_pid, *pid.get_or_insert(current_pid));
        }
        model.stop();
        assert!(model.child.is_none() && model.events.is_none());
    }

    #[test]
    fn cancellation_unblocks_large_audio_upload_and_reaps_process() {
        let mut model = mock_worker("import time; time.sleep(600)");
        let mut calls = 0;
        let callbacks = StreamCallbacks {
            should_abort: Some(Box::new(move || {
                calls += 1;
                calls >= 5
            })),
            ..Default::default()
        };
        let started = Instant::now();
        let error = model
            .transcribe(&vec![0.0; 1_000_000], TranscribeOptions::default(), callbacks)
            .unwrap_err();
        assert!(error.to_string().contains("aborted"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(model.child.is_none() && model.events.is_none());
    }
}

#[cfg(all(test, unix))]
#[path = "audio8_cancellation_tests.rs"]
mod cancellation_tests;
