//! Synthetic workers only: live ordering, bounded upload failures and HTTP duplex cancellation.
use super::*;
use crate::audio8::cancellation_tests::Fixture;
use bytes::Bytes;
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::Instant,
};

const WORKER: &str = r#"
import json,os,pathlib,select,sys
fd=sys.stdin.fileno()
def emit(event): print(json.dumps(event),flush=True)
def control():
    line=bytearray()
    while True:
        byte=os.read(fd,1)
        if not byte: return None
        if byte==b'\n': return json.loads(line)
        line.extend(byte)
def exact(count):
    data=bytearray()
    while len(data)<count:
        part=os.read(fd,count-len(data))
        if not part: raise RuntimeError('truncated synthetic frame')
        data.extend(part)
    return data
def audit(event):
    with pathlib.Path('live-controls').open('a') as log: log.write(json.dumps(event)+'\n')
with pathlib.Path('live-worker-pids').open('a') as log: log.write(str(os.getpid())+'\n')
emit({'type':'ready'})
while True:
    header=control()
    if header is None: break
    assert header['type']=='stream_start'
    assert header['language'] in ('en','zh')
    audit(header)
    # CALIBRATION_HOOK
    assert not select.select([fd],[],[],0.05)[0], 'PCM arrived before live readiness'
    emit({'type':'calibration','realtime':True,'seconds':0.01})
    emit({'type':'stream_ready'})
    samples=0
    while True:
        header=control()
        assert header is not None
        audit(header)
        if header['type']=='stream_end':
            emit({'type':'segment','start':0,'end':(samples+159)//160,'text':'synthetic tail'})
            emit({'type':'result','segments':1})
            break
        assert header['type']=='stream_chunk'
        count=header['sample_count']
        assert 0<count<=2560
        exact(count*4)
        samples+=count
        emit({'type':'partial','text':'synthetic partial','segment_index':0})
"#;

fn runtime(model: &Model) -> PathBuf {
    model.manifest.parent().unwrap().join(".audio8-runtime")
}

fn pcm(samples: usize) -> Bytes {
    Bytes::from(vec![0u8; samples * 4])
}

fn audit(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn pids(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|pid| pid.parse().unwrap())
        .collect()
}

#[test]
fn disconnected_client_after_worker_load_reaps_runtime_before_calibration() {
    let fixture = Fixture::new(WORKER);
    let mut model = fixture.load();
    let (_sender, pcm) = tokio::sync::mpsc::channel(1);
    let result = model.transcribe_live(
        "en",
        4,
        pcm,
        |_| Err(anyhow::anyhow!("synthetic client disconnected")),
        || false,
    );
    assert!(result.is_err());
    assert_stopped(&model);
}

fn assert_stopped(model: &Model) {
    assert!(model.child.is_none() && model.input.is_none() && model.events.is_none());
    assert!(model.readers.is_empty(), "reader threads must be joined");
}

fn assert_gone(pid: u32) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if unsafe { libc::kill(pid as i32, 0) } == -1 {
            assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
            return;
        }
        assert!(Instant::now() < deadline, "synthetic live worker {pid} survived cancellation");
        thread::sleep(Duration::from_millis(20));
    }
}

fn run_live(model: &mut Model, frames: Vec<Result<Bytes, String>>) -> anyhow::Result<Vec<Value>> {
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    let mut sender = Some(sender);
    let mut frames = Some(frames);
    let mut events = Vec::new();
    let started = Instant::now();
    model.transcribe_live(
        "en",
        2,
        receiver,
        |event| {
            if event["type"] == "ready" {
                let sender = sender.take().unwrap();
                for frame in frames.take().unwrap() {
                    sender.try_send(frame).unwrap();
                }
                drop(sender); // graceful capture EOF, never an abort
            }
            events.push(event);
            Ok(())
        },
        || started.elapsed() > Duration::from_secs(3),
    )?;
    Ok(events)
}

#[test]
fn live_readiness_precedes_even_prequeued_pcm_and_eof_flushes_tail_once() {
    let fixture = Fixture::new(WORKER);
    let mut model = fixture.load();
    let controls = runtime(&model).join("live-controls");
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    sender.try_send(Ok(pcm(2560))).unwrap();
    sender.try_send(Ok(pcm(17))).unwrap();
    drop(sender);
    let mut events = Vec::new();
    let started = Instant::now();
    model
        .transcribe_live(
            "zh",
            3,
            receiver,
            |event| {
                if event["type"] == "ready" {
                    assert_eq!(audit(&controls).len(), 1, "no PCM controls may precede readiness");
                }
                events.push(event);
                Ok(())
            },
            || started.elapsed() > Duration::from_secs(3),
        )
        .unwrap();
    let kinds: Vec<_> = events.iter().map(|event| event["type"].as_str().unwrap()).collect();
    assert_eq!(
        kinds,
        ["loading", "calibration", "ready", "partial", "partial", "segment", "result"]
    );
    let segment = events.iter().find(|event| event["type"] == "segment").unwrap();
    assert_eq!(segment["end"], 17);
    assert_eq!(segment["segment_index"], 0);
    assert_eq!(events.last().unwrap()["segments"], 1);
    let controls = audit(&controls);
    assert_eq!(controls[0], json!({"type":"stream_start", "language":"zh", "threads":3}));
    assert_eq!(controls.iter().filter(|event| event["type"] == "stream_end").count(), 1);
    assert_eq!(
        controls
            .iter()
            .filter_map(|event| event["sample_count"].as_u64())
            .collect::<Vec<_>>(),
        [2560, 17]
    );
    let pid = model.child.as_ref().unwrap().id();
    drop(model);
    assert_gone(pid);
}

#[test]
fn consecutive_live_sessions_reuse_the_worker_and_flush_each_tail_once() {
    let fixture = Fixture::new(WORKER);
    let mut model = fixture.load();
    let mut pid = None;
    for samples in [160, 17] {
        let events = run_live(&mut model, vec![Ok(pcm(samples))]).unwrap();
        let current = model.child.as_ref().unwrap().id();
        assert_eq!(current, *pid.get_or_insert(current));
        assert_eq!(events.iter().filter(|event| event["type"] == "segment").count(), 1);
        assert_eq!(events.iter().filter(|event| event["type"] == "result").count(), 1);
        assert_eq!(events.last().unwrap()["segments"], 1);
    }
    let controls = audit(&runtime(&model).join("live-controls"));
    assert_eq!(controls.iter().filter(|event| event["type"] == "stream_start").count(), 2);
    assert_eq!(controls.iter().filter(|event| event["type"] == "stream_end").count(), 2);
    assert_eq!(pids(&runtime(&model).join("live-worker-pids")).len(), 1);
    drop(model);
    assert_gone(pid.unwrap());
}

#[test]
fn calibration_cancellation_reaps_the_group_and_a_later_live_request_restarts() {
    let hook = r#"    import subprocess,time
    if not pathlib.Path('calibration-attempt').exists():
        pathlib.Path('calibration-attempt').touch()
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)'])
        pathlib.Path('calibrating-pids').write_text(f'{os.getpid()} {child.pid}')
        time.sleep(600)"#;
    let code = WORKER.replace("    # CALIBRATION_HOOK", hook);
    let fixture = Fixture::new(&code);
    let mut model = fixture.load();
    let pid_path = runtime(&model).join("calibrating-pids");
    let (_sender, receiver) = tokio::sync::mpsc::channel(4);
    let started = Instant::now();
    let mut events = Vec::new();
    let error = model
        .transcribe_live(
            "en",
            2,
            receiver,
            |event| {
                events.push(event);
                Ok(())
            },
            || pid_path.exists() || started.elapsed() > Duration::from_secs(3),
        )
        .unwrap_err();
    assert!(error.to_string().contains("aborted"));
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(!events.iter().any(|event| event["type"] == "ready"));
    assert_stopped(&model);
    let cancelled = pids(&pid_path);
    assert_eq!(cancelled.len(), 2);
    for pid in &cancelled {
        assert_gone(*pid);
    }
    assert_eq!(
        unsafe { libc::waitpid(cancelled[0] as i32, std::ptr::null_mut(), libc::WNOHANG) },
        -1
    );
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::ECHILD));
    let events = run_live(&mut model, vec![Ok(pcm(17))]).unwrap();
    assert_eq!(events.last().unwrap()["type"], "result");
    let restarted = model.child.as_ref().unwrap().id();
    assert_ne!(restarted, cancelled[0]);
    drop(model);
    assert_gone(restarted);
}

#[test]
fn malformed_pcm_and_explicit_input_overload_fail_without_hanging_and_can_restart() {
    for (frame, expected) in [
        (Ok(Bytes::new()), "Invalid Audio8 live PCM frame"),
        (Ok(Bytes::from_static(&[0, 0, 0])), "Invalid Audio8 live PCM frame"),
        (Ok(pcm(2561)), "Invalid Audio8 live PCM frame"),
        (
            Err("Audio8 live input overload: capture queue is full".to_owned()),
            "input overload",
        ),
    ] {
        let fixture = Fixture::new(WORKER);
        let mut model = fixture.load();
        let started = Instant::now();
        let error = run_live(&mut model, vec![frame]).unwrap_err();
        assert!(error.to_string().contains(expected), "unexpected live failure classification");
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_stopped(&model);
        let first = pids(&runtime(&model).join("live-worker-pids"))[0];
        assert_gone(first);
        run_live(&mut model, vec![Ok(pcm(17))]).unwrap();
        let restarted = model.child.as_ref().unwrap().id();
        assert_ne!(first, restarted);
        drop(model);
        assert_gone(restarted);
    }
}

#[test]
fn non_realtime_calibration_fails_closed_before_upload() {
    let code = WORKER.replace("'realtime':True", "'realtime':False");
    let fixture = Fixture::new(&code);
    let mut model = fixture.load();
    let error = run_live(&mut model, vec![Ok(pcm(17))]).unwrap_err();
    assert!(error.to_string().contains("startup event"));
    assert_stopped(&model);
    let controls = audit(&runtime(&model).join("live-controls"));
    assert_eq!(controls.len(), 1);
    assert_gone(pids(&runtime(&model).join("live-worker-pids"))[0]);
}

struct ServerProcess {
    child: Child,
    workers: PathBuf,
}
impl Drop for ServerProcess {
    fn drop(&mut self) {
        // Model workers have their own process groups. Also clean those groups
        // on test panic, rather than only killing the outer CLI process.
        if let Ok(workers) = std::fs::read_to_string(&self.workers) {
            for pid in workers.split_whitespace().filter_map(|pid| pid.parse::<i32>().ok()) {
                unsafe {
                    libc::kill(-pid, libc::SIGKILL);
                }
            }
        }
        unsafe {
            libc::kill(-(self.child.id() as i32), libc::SIGKILL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

type HttpStream = std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
struct HttpEvents {
    stream: HttpStream,
    buffer: Vec<u8>,
}
impl HttpEvents {
    async fn next(&mut self) -> Option<Value> {
        use futures_util::StreamExt;
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if let Some(end) = self.buffer.iter().position(|byte| *byte == b'\n') {
                    let line: Vec<_> = self.buffer.drain(..=end).collect();
                    return Some(serde_json::from_slice(&line).expect("invalid live NDJSON envelope"));
                }
                match self.stream.next().await {
                    Some(Ok(bytes)) => self.buffer.extend_from_slice(&bytes),
                    Some(Err(_)) => panic!("live HTTP response disconnected unexpectedly"),
                    None => {
                        assert!(self.buffer.is_empty());
                        return None;
                    }
                }
            }
        })
        .await
        .expect("live HTTP event timed out")
    }
    async fn ready(&mut self) {
        let mut calibrated = false;
        loop {
            match self.next().await.expect("live stream ended before readiness")["type"]
                .as_str()
                .unwrap()
            {
                "loading" => {}
                "calibration" => calibrated = true,
                "ready" => {
                    assert!(calibrated);
                    return;
                }
                _ => panic!("live HTTP failed before readiness"),
            }
        }
    }
    async fn finish(&mut self) -> Vec<Value> {
        let mut events = Vec::new();
        while let Some(event) = self.next().await {
            assert_ne!(event["type"], "error", "synthetic live HTTP request failed");
            events.push(event);
        }
        assert_eq!(events.iter().filter(|event| event["type"] == "segment").count(), 1);
        assert_eq!(events.iter().filter(|event| event["type"] == "result").count(), 1);
        events
    }
}

async fn open_http_live(
    client: &reqwest::Client,
    base: &str,
) -> (tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>, HttpEvents) {
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    let response = tokio::time::timeout(
        Duration::from_secs(3),
        client
            .post(format!("{base}/v1/audio/live?language=en&n_threads=2"))
            .header("content-type", "application/octet-stream")
            .body(reqwest::Body::wrap_stream(tokio_stream::wrappers::ReceiverStream::new(
                receiver,
            )))
            .send(),
    )
    .await
    .expect("HTTP live did not respond before upload")
    .unwrap();
    assert!(response.status().is_success(), "live HTTP request was rejected");
    assert_eq!(response.headers()["content-type"], "application/x-ndjson");
    (
        sender,
        HttpEvents {
            stream: Box::pin(response.bytes_stream()),
            buffer: Vec::new(),
        },
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "build vibe-server and set VIBE_AUDIO8_TEST_SERVER to the binary path"]
async fn real_http_live_is_duplex_and_response_drop_releases_the_model_lease() {
    use std::os::unix::process::CommandExt;
    let fixture = Fixture::new(WORKER);
    let model = fixture.load();
    let marker = model.manifest.clone();
    let directory = runtime(&model);
    let workers = directory.join("live-worker-pids");
    drop(model); // marker-only, no local worker; all work below crosses HTTP.
    let binary = std::env::var_os("VIBE_AUDIO8_TEST_SERVER").expect("set VIBE_AUDIO8_TEST_SERVER to a built server binary");
    let mut server = ServerProcess {
        child: Command::new(binary)
            .args(["serve", "--host", "127.0.0.1", "--port", "0", "--exit-with-parent", "false"])
            .current_dir(marker.parent().unwrap())
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
        workers: workers.clone(),
    };
    let stdout = server.child.stdout.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut line = String::new();
        BufReader::new(stdout).read_line(&mut line).unwrap();
        let _ = sender.send(line);
    });
    let line = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("test server did not become ready");
    reader.join().unwrap();
    let ready: Value = serde_json::from_str(&line).unwrap();
    let base = format!("http://127.0.0.1:{}", ready["port"].as_u64().unwrap());
    let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build().unwrap();
    let load = || {
        client
            .post(format!("{base}/v1/models/load"))
            .header("content-type", "application/json")
            .body(json!({"path":marker, "no_gpu":true}).to_string())
    };
    assert!(load().send().await.unwrap().status().is_success());
    assert!(!directory.exists(), "HTTP model load must remain lazy");

    let (upload, mut response) = open_http_live(&client, &base).await;
    response.ready().await;
    assert_eq!(audit(&directory.join("live-controls")).len(), 1);
    let frame = pcm(2560 + 17);
    upload.send(Ok(frame.slice(..3))).await.unwrap();
    upload.send(Ok(frame.slice(3..))).await.unwrap();
    drop(upload);
    let events = response.finish().await;
    assert_eq!(events.iter().find(|event| event["type"] == "segment").unwrap()["end"], 17);
    assert_eq!(pids(&workers).len(), 1);

    let (upload, mut response) = open_http_live(&client, &base).await;
    response.ready().await;
    assert_eq!(pids(&workers).len(), 1, "consecutive HTTP sessions reuse the runtime");
    let cancelled_pid = pids(&workers)[0];
    drop(response); // keep the upload sender alive: this is response cancellation, not EOF.
    tokio::time::timeout(Duration::from_secs(3), upload.closed())
        .await
        .expect("response drop did not close the upload transport");
    drop(upload);
    let started = Instant::now();
    let loaded = tokio::time::timeout(Duration::from_secs(3), load().send())
        .await
        .expect("cancelled HTTP live request stranded the model lease")
        .unwrap();
    assert!(loaded.status().is_success());
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_gone(cancelled_pid);

    let (upload, mut response) = open_http_live(&client, &base).await;
    response.ready().await;
    upload.send(Ok(pcm(17))).await.unwrap();
    drop(upload);
    response.finish().await;
    let restarted = pids(&workers);
    assert_eq!(restarted.len(), 2);
    assert_ne!(restarted[0], restarted[1]);

    let (upload, mut response) = open_http_live(&client, &base).await;
    response.ready().await;
    upload.send(Ok(Bytes::from_static(&[0, 0, 0]))).await.unwrap();
    drop(upload);
    let mut errors = 0;
    while let Some(event) = response.next().await {
        assert_ne!(event["type"], "result", "malformed PCM must not be finalized as success");
        if event["type"] == "error" {
            assert!(event["message"].as_str().unwrap().contains("incomplete f32 sample"));
            errors += 1;
        }
    }
    assert_eq!(errors, 1, "malformed HTTP PCM must report an explicit error");
    assert_gone(restarted[1]);
    let unloaded = client.delete(format!("{base}/v1/models")).send().await.unwrap();
    assert!(unloaded.status().is_success());
    assert_gone(restarted[1]);
    drop(server);
}
