//! Fake installed interpreters exercise lazy startup, cancellation and process ownership.
use super::*;
use std::{ffi::OsString, os::unix::fs::PermissionsExt, sync::OnceLock};

static ENVIRONMENT: Mutex<()> = Mutex::new(());
static PYTHON: OnceLock<PathBuf> = OnceLock::new();

pub(super) fn environment_lock() -> std::sync::MutexGuard<'static, ()> {
    ENVIRONMENT.lock().unwrap_or_else(|error| error.into_inner())
}

pub(super) struct Fixture {
    directory: tempfile::TempDir,
    previous: Vec<(&'static str, Option<OsString>)>,
    _environment: std::sync::MutexGuard<'static, ()>,
}

impl Fixture {
    pub(super) fn new(code: &str) -> Self {
        let environment = environment_lock();
        let python = PYTHON.get_or_init(|| {
            let output = Command::new("uv")
                .args([
                    "run",
                    "--no-project",
                    "--no-config",
                    "--no-env-file",
                    "--python",
                    "3.12",
                    "python",
                    "-c",
                    "import sys; print(sys.executable)",
                ])
                .output()
                .expect("uv/Python 3.12 is required for Audio8 fake-worker tests");
            assert!(output.status.success(), "could not locate test Python");
            PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
        });
        let directory = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        let interpreter = directory.path().join("fake-python");
        let quoted_python = python.to_str().unwrap().replace('\'', "'\\''");
        std::fs::write(
            &interpreter,
            format!("#!/bin/sh\nexec '{quoted_python}' -I -u \"$0.py\" \"$@\"\n"),
        )
        .unwrap();
        std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(interpreter.with_extension("py"), code).unwrap();
        std::fs::write(
            directory.path().join("test.audio8.json"),
            serde_json::json!({"engine":"audio8", "checkpoint":CHECKPOINT, "revision":REVISION}).to_string(),
        )
        .unwrap();
        let mut fixture = Self {
            directory,
            previous: Vec::new(),
            _environment: environment,
        };
        fixture.set_env("VIBE_AUDIO8_PYTHON", Some(interpreter.into_os_string()));
        fixture.set_env("VIBE_AUDIO8_STARTUP_TIMEOUT_SECONDS", Some("1800".into()));
        fixture
    }

    fn set_env(&mut self, name: &'static str, value: Option<OsString>) {
        if !self.previous.iter().any(|(previous, _)| *previous == name) {
            self.previous.push((name, std::env::var_os(name)));
        }
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }

    fn marker(&self) -> PathBuf {
        self.directory.path().join("test.audio8.json")
    }
    fn runtime(&self) -> PathBuf {
        self.directory.path().join(".audio8-runtime")
    }
    pub(super) fn load(&self) -> Model {
        Model::load(self.marker().to_str().unwrap(), ContextOptions::default()).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..).rev() {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

const DECODE: &str = r#"
import json,sys
print(json.dumps({'type':'ready'}),flush=True)
while True:
    line=sys.stdin.buffer.readline()
    if not line: break
    header=json.loads(line)
    raw=sys.stdin.buffer.read(header['sample_count']*4)
    assert len(raw)==header['sample_count']*4
    print(json.dumps({'type':'segment','start':0,'end':1,'text':'restarted'}),flush=True)
    print(json.dumps({'type':'result','segments':1}),flush=True)
"#;

fn assert_stopped(model: &Model) {
    assert!(model.child.is_none() && model.input.is_none() && model.events.is_none());
    assert!(model.readers.is_empty(), "reader threads must be joined, not detached");
}

fn assert_gone(pid: u32) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if unsafe { libc::kill(pid as i32, 0) } == -1 {
            assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
            break;
        }
        assert!(Instant::now() < deadline, "worker/descendant {pid} survived cancellation");
        thread::sleep(Duration::from_millis(20));
    }
}

fn pids(path: &std::path::Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|pid| pid.parse().unwrap())
        .collect()
}

#[test]
fn load_and_empty_or_preaborted_request_never_start_a_worker() {
    let fixture = Fixture::new("raise AssertionError('load must not execute the interpreter')");
    let started = Instant::now();
    let mut model = fixture.load();
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_stopped(&model);
    let result = model
        .transcribe(&[], TranscribeOptions::default(), StreamCallbacks::default())
        .unwrap();
    assert!(result.segments.is_empty());
    let callbacks = StreamCallbacks {
        should_abort: Some(Box::new(|| true)),
        ..Default::default()
    };
    assert!(model
        .transcribe(&[0.0], TranscribeOptions::default(), callbacks)
        .unwrap_err()
        .to_string()
        .contains("aborted"));
    assert_stopped(&model);
    assert!(!fixture.runtime().exists(), "load must not even install the runtime");
}

#[test]
fn never_ready_startup_is_cancelled_and_entire_group_is_reaped() {
    let fixture = Fixture::new(
        r#"
import os,pathlib,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'])
sys.stderr.write('hf_startup_secret_token\n'*10000)
sys.stderr.flush()
pathlib.Path('pids').write_text(f'{os.getpid()} {child.pid}')
time.sleep(600)
"#,
    );
    let mut model = fixture.load();
    let pid_path = fixture.runtime().join("pids");
    let started = Instant::now();
    let mut polls = Vec::new();
    let mut progress = Vec::new();
    let callbacks = StreamCallbacks {
        should_abort: Some(Box::new(|| {
            polls.push(Instant::now());
            (pid_path.exists() && started.elapsed() > Duration::from_millis(350)) || started.elapsed() > Duration::from_secs(3)
        })),
        on_progress: Some(Box::new(|value| progress.push(value))),
        ..Default::default()
    };
    let error = model
        .transcribe(&[0.0; 160], TranscribeOptions::default(), callbacks)
        .unwrap_err();
    assert!(error.to_string().contains("aborted"), "{error}");
    assert!(!format!("{error:#}").contains("hf_startup_secret_token"));
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(polls
        .windows(2)
        .all(|pair| pair[1].duration_since(pair[0]) < Duration::from_millis(300)));
    assert!(progress.is_empty(), "startup must not fabricate quantified download progress");
    assert_stopped(&model);
    assert!(model.diagnostics.lock().unwrap().len() <= MAX_DIAGNOSTIC_BYTES);
    let worker_pids = pids(&pid_path);
    assert_eq!(worker_pids.len(), 2);
    for pid in &worker_pids {
        assert_gone(*pid);
    }
    assert_eq!(
        unsafe { libc::waitpid(worker_pids[0] as i32, std::ptr::null_mut(), libc::WNOHANG) },
        -1
    );
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::ECHILD));
}

#[test]
fn default_uv_launcher_cancellation_kills_uv_and_python_children() {
    let mut fixture = Fixture::new(
        r#"
import json,os,pathlib,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'])
pathlib.Path('args.json').write_text(json.dumps(sys.argv[1:]))
pathlib.Path('pids').write_text(f'{os.getppid()} {os.getpid()} {child.pid}')
time.sleep(600)
"#,
    );
    let interpreter = fixture.directory.path().join("fake-python");
    let launcher = fixture.directory.path().join("uv");
    // Keep the fake uv parent alive, just as uv waits for its Python worker.
    let wrapper = std::fs::read_to_string(&interpreter)
        .unwrap()
        .replace("exec ", "")
        .replace(" \"$@\"\n", " \"$@\" &\nwait\n");
    std::fs::write(&launcher, wrapper).unwrap();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::copy(interpreter.with_extension("py"), launcher.with_extension("py")).unwrap();
    let mut paths = vec![fixture.directory.path().to_owned()];
    paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()));
    fixture.set_env("PATH", Some(std::env::join_paths(paths).unwrap()));
    fixture.set_env("VIBE_AUDIO8_PYTHON", None);
    let mut model = fixture.load();
    let pid_path = fixture.runtime().join("pids");
    let started = Instant::now();
    let mut callbacks = StreamCallbacks {
        should_abort: Some(Box::new(|| pid_path.exists() || started.elapsed() > Duration::from_secs(3))),
        ..Default::default()
    };
    assert!(model
        .ensure_started(&mut callbacks)
        .unwrap_err()
        .to_string()
        .contains("aborted"));
    assert_stopped(&model);
    let worker_pids = pids(&pid_path);
    assert_eq!(worker_pids.len(), 3);
    for pid in &worker_pids {
        assert_gone(*pid);
    }
    assert_eq!(
        unsafe { libc::waitpid(worker_pids[0] as i32, std::ptr::null_mut(), libc::WNOHANG) },
        -1
    );
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::ECHILD));
    let args: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture.runtime().join("args.json")).unwrap()).unwrap();
    assert_eq!(
        &args.as_array().unwrap()[..8],
        &serde_json::json!([
            "run",
            "--no-project",
            "--no-config",
            "--no-env-file",
            "--python",
            "3.12",
            "--script",
            fixture.runtime().join("audio8_runtime.py")
        ])
        .as_array()
        .unwrap()[..]
    );
}

#[test]
fn aborted_startup_then_later_transcription_and_reload_restart_cleanly() {
    let code = format!(
        r#"
import os,pathlib,time
attempts=pathlib.Path('attempts')
first=not attempts.exists()
with attempts.open('a') as log: log.write(f'{{os.getpid()}}\n')
if first: time.sleep(600)
{DECODE}
"#
    );
    let fixture = Fixture::new(&code);
    let mut model = fixture.load();
    let attempts = fixture.runtime().join("attempts");
    let started = Instant::now();
    let callbacks = StreamCallbacks {
        should_abort: Some(Box::new(|| attempts.exists() || started.elapsed() > Duration::from_secs(3))),
        ..Default::default()
    };
    assert!(model
        .transcribe(&[0.0; 160], TranscribeOptions::default(), callbacks)
        .unwrap_err()
        .to_string()
        .contains("aborted"));
    assert_stopped(&model);
    let first_pid = pids(&attempts)[0];
    assert_gone(first_pid);
    for _ in 0..2 {
        let result = model
            .transcribe(&[0.0; 160], TranscribeOptions::default(), StreamCallbacks::default())
            .unwrap();
        assert_eq!(result.text(), "restarted");
        assert_ne!(model.child.as_ref().unwrap().id(), first_pid);
    }
    assert_eq!(pids(&attempts).len(), 2, "successful requests reuse the ready worker");
    let second_pid = model.child.as_ref().unwrap().id();
    model.abort_runtime();
    model.abort_runtime();
    assert_stopped(&model);
    assert_gone(second_pid);
    drop(model);
    let mut reloaded = fixture.load();
    assert_stopped(&reloaded);
    assert_eq!(
        reloaded
            .transcribe(&[0.0; 160], TranscribeOptions::default(), StreamCallbacks::default())
            .unwrap()
            .text(),
        "restarted"
    );
    let third_pid = reloaded.child.as_ref().unwrap().id();
    drop(reloaded);
    assert_gone(third_pid);
    assert_eq!(pids(&attempts).len(), 3);
}

#[test]
fn dropping_worker_unblocks_full_event_queue_and_joins_readers() {
    let fixture = Fixture::new(
        r#"
import json,os,pathlib,subprocess,sys
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'])
pathlib.Path('pids').write_text(f'{os.getpid()} {child.pid}')
print(json.dumps({'type':'ready'}),flush=True)
while True: print(json.dumps({'type':'progress','value':0}),flush=True)
"#,
    );
    let mut model = fixture.load();
    model.ensure_started(&mut StreamCallbacks::default()).unwrap();
    thread::sleep(Duration::from_millis(150));
    let worker_pids = pids(&fixture.runtime().join("pids"));
    let started = Instant::now();
    drop(model);
    assert!(started.elapsed() < Duration::from_secs(2));
    for pid in worker_pids {
        assert_gone(pid);
    }
}

#[test]
fn preaborted_request_stops_an_already_ready_worker() {
    let fixture = Fixture::new(DECODE);
    let mut model = fixture.load();
    model.ensure_started(&mut StreamCallbacks::default()).unwrap();
    let pid = model.child.as_ref().unwrap().id();
    let callbacks = StreamCallbacks {
        should_abort: Some(Box::new(|| true)),
        ..Default::default()
    };
    assert!(model.transcribe(&[0.0], TranscribeOptions::default(), callbacks).is_err());
    assert_stopped(&model);
    assert_gone(pid);
}

#[test]
fn startup_errors_are_bounded_and_do_not_expose_stderr_credentials() {
    let fixture = Fixture::new(
        r#"
import json,sys
sys.stderr.write('https://user:private_hf_password@example.com\n'*10000)
sys.stderr.flush()
print(json.dumps({'type':'error','message':'runtime startup failed: '+'x'*25000}),flush=True)
"#,
    );
    let mut model = fixture.load();
    let error = model.ensure_started(&mut StreamCallbacks::default()).unwrap_err();
    let rendered = format!("{error:#}");
    assert!(rendered.contains("runtime startup failed"));
    assert!(rendered.len() < 5000, "startup errors must be bounded");
    assert!(!rendered.contains("private_hf_password"));
    assert_stopped(&model);
    assert!(model.diagnostics.lock().unwrap().len() <= MAX_DIAGNOSTIC_BYTES);
}

#[test]
fn installed_interpreter_flags_and_hf_auth_cache_environment_are_preserved() {
    let code = format!(
        r#"
import json,os,pathlib,sys
keys=('HF_HOME','HF_TOKEN_PATH','HF_TOKEN','PYTHONPATH','PYTHONSAFEPATH','PYTHONUNBUFFERED','HF_HUB_DISABLE_PROGRESS_BARS','TOKENIZERS_PARALLELISM')
env={{key:os.environ[key] for key in keys if key in os.environ}}
pathlib.Path('launch.json').write_text(json.dumps({{'args':sys.argv[1:], 'env':env}}))
{DECODE}
"#
    );
    let mut fixture = Fixture::new(&code);
    let cache = fixture.directory.path().join("login-cache");
    fixture.set_env("HF_HOME", None);
    fixture.set_env("HF_TOKEN_PATH", None);
    fixture.set_env("XDG_CACHE_HOME", Some(cache.clone().into_os_string()));
    fixture.set_env("HF_TOKEN", Some("test-private-token".into()));
    fixture.set_env("PYTHONPATH", Some("untrusted-python-path".into()));
    let mut model = Model::load(
        fixture.marker().to_str().unwrap(),
        ContextOptions {
            gpu_device: 2,
            no_gpu: true,
        },
    )
    .unwrap();
    model.ensure_started(&mut StreamCallbacks::default()).unwrap();
    let capture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.runtime().join("launch.json")).unwrap()).unwrap();
    let args = capture["args"].as_array().unwrap();
    assert_eq!(args[0], "-I");
    assert_eq!(args[1], "-u");
    assert_eq!(args[2], fixture.runtime().join("audio8_runtime.py").to_str().unwrap());
    assert_eq!(args[3], "--manifest");
    assert_eq!(args[4], fixture.marker().to_str().unwrap());
    assert_eq!(args[5], "--gpu-device");
    assert_eq!(args[6], "2");
    assert_eq!(args[7], "--no-gpu");
    let env = &capture["env"];
    assert_eq!(env["HF_HOME"], fixture.runtime().join("hf").to_str().unwrap());
    assert_eq!(env["HF_TOKEN_PATH"], cache.join("huggingface/token").to_str().unwrap());
    assert_eq!(env["HF_TOKEN"], "test-private-token");
    assert!(env.get("PYTHONPATH").is_none());
    assert_eq!(env["PYTHONSAFEPATH"], "1");
    assert_eq!(env["PYTHONUNBUFFERED"], "1");
    assert_eq!(env["HF_HUB_DISABLE_PROGRESS_BARS"], "1");
    assert_eq!(env["TOKENIZERS_PARALLELISM"], "false");
    assert_eq!(
        std::fs::read_to_string(fixture.runtime().join("audio8_runtime.py")).unwrap(),
        SCRIPT
    );
    model.abort_runtime();
    let explicit_home = fixture.directory.path().join("explicit-hf");
    let explicit_token = fixture.directory.path().join("explicit-token");
    fixture.set_env("HF_HOME", Some(explicit_home.clone().into_os_string()));
    fixture.set_env("HF_TOKEN_PATH", Some(explicit_token.clone().into_os_string()));
    model.ensure_started(&mut StreamCallbacks::default()).unwrap();
    let capture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.runtime().join("launch.json")).unwrap()).unwrap();
    assert_eq!(capture["env"]["HF_HOME"], explicit_home.to_str().unwrap());
    assert_eq!(capture["env"]["HF_TOKEN_PATH"], explicit_token.to_str().unwrap());
}

struct ServerProcess(Child);
impl Drop for ServerProcess {
    fn drop(&mut self) {
        unsafe {
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "build vibe-server and set VIBE_AUDIO8_TEST_SERVER to the binary path"]
async fn real_http_model_load_is_fast_and_does_not_launch_interpreter() {
    use std::os::unix::process::CommandExt;
    let fixture = Fixture::new("import pathlib,time; pathlib.Path('launched').touch(); time.sleep(600)");
    let binary = std::env::var_os("VIBE_AUDIO8_TEST_SERVER").expect("set VIBE_AUDIO8_TEST_SERVER to a built vibe-server binary");
    let mut server = ServerProcess(
        Command::new(binary)
            .args(["serve", "--host", "127.0.0.1", "--port", "0", "--exit-with-parent", "false"])
            .current_dir(fixture.directory.path())
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let stdout = server.0.stdout.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut line = String::new();
        BufReader::new(stdout).read_line(&mut line).unwrap();
        let _ = sender.send(line);
    });
    let line = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("server did not become ready");
    reader.join().unwrap();
    let ready: serde_json::Value = serde_json::from_str(&line).unwrap();
    let base = format!("http://127.0.0.1:{}", ready["port"].as_u64().unwrap());
    let client = reqwest::Client::builder().timeout(Duration::from_secs(2)).build().unwrap();
    for _ in 0..2 {
        let started = Instant::now();
        let response = client
            .post(format!("{base}/v1/models/load"))
            .header("content-type", "application/json")
            .body(serde_json::json!({"path":fixture.marker(), "no_gpu":true}).to_string())
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success(), "{}", response.text().await.unwrap());
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(
            !fixture.runtime().exists(),
            "HTTP load must not create runtime files or launch any child"
        );
        let response = client.delete(format!("{base}/v1/models")).send().await.unwrap();
        assert!(response.status().is_success());
    }
    drop(server);
}
