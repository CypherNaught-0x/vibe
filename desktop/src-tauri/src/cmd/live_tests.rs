use super::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::Duration;

fn send_event(stream: &mut std::net::TcpStream, event: Value) {
    let line = format!("{event}\n");
    write!(stream, "{:x}\r\n{line}\r\n", line.len()).unwrap();
    stream.flush().unwrap();
}

#[test]
fn session_targeted_stops_do_not_affect_newer_capture() {
    assert!(matches_session("null", "current"));
    assert!(matches_session(r#"{"sessionId":"current"}"#, "current"));
    assert!(!matches_session(r#"{"sessionId":"stale"}"#, "current"));
}

#[test]
fn calibration_rejects_slow_backend_and_invalid_capacity() {
    let mut profile: Profile = serde_json::from_value(json!({
        "type": "calibration", "window_tokens": 32, "pcm_capacity_samples": 15_360,
        "step_seconds": 0.02, "realtime": false, "device": "cpu",
    }))
    .unwrap();
    assert!(profile.validate().unwrap_err().to_string().contains("too slow"));
    profile.realtime = true;
    assert!(profile.validate().is_ok());
    profile.pcm_capacity_samples = 0;
    assert!(profile.validate().is_err());
}

#[test]
fn one_active_session_and_cleanup_allows_next_session() {
    let first = Lease::acquire().unwrap();
    assert!(Lease::acquire().is_err());
    drop(first);
    let next = Lease::acquire().unwrap();
    cancel_live_recording();
    assert_eq!(next.0.state(), CANCEL);
}

#[test]
fn segment_bounds_are_centiseconds_and_text_is_not_joined_with_extra_spaces() {
    let a = serde_json::from_value::<LiveSegment>(json!({"start": 42, "end": 78, "text": " hello"}))
        .unwrap()
        .into_segment()
        .unwrap();
    let b = serde_json::from_value::<LiveSegment>(json!({"start": 78, "end": 100, "text": "world"}))
        .unwrap()
        .into_segment()
        .unwrap();
    assert_eq!((a.start, a.stop, b.start, b.stop), (42, 78, 78, 100));
    assert_eq!([a.text, b.text].concat(), " helloworld");
    assert!(
        serde_json::from_value::<LiveSegment>(json!({"start": 42, "end": 41, "text": "bad"}))
            .unwrap()
            .into_segment()
            .is_err()
    );
}

#[tokio::test]
async fn full_duplex_http_waits_for_ready_uploads_f32le_and_eof_flushes_result() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            headers.push_str(&line);
            if line == "\r\n" {
                break;
            }
        }
        assert!(headers.starts_with("POST /v1/audio/live?language=en&n_threads=4 HTTP/1.1"));
        assert!(headers
            .to_ascii_lowercase()
            .contains("content-type: application/octet-stream"));
        assert!(headers.to_ascii_lowercase().contains("transfer-encoding: chunked"));
        let mut response = stream;
        response.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").unwrap();
        send_event(&mut response, json!({"type": "loading"}));
        send_event(
            &mut response,
            json!({"type": "calibration", "window_tokens": 32, "pcm_capacity_samples": 15_360, "step_seconds": 0.02, "realtime": true}),
        );
        // Calibration is not enough: no PCM may be uploaded before the ready event.
        reader.get_ref().set_read_timeout(Some(Duration::from_millis(80))).unwrap();
        let error = reader.read(&mut [0u8; 1]).unwrap_err();
        assert!(matches!(
            error.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
        ));
        reader.get_ref().set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        send_event(&mut response, json!({"type": "ready"}));
        let mut uploaded = Vec::new();
        loop {
            let mut size = String::new();
            reader.read_line(&mut size).unwrap();
            let count = usize::from_str_radix(size.trim().split(';').next().unwrap(), 16).unwrap();
            if count == 0 {
                reader.read_line(&mut String::new()).unwrap();
                break;
            }
            let start = uploaded.len();
            uploaded.resize(start + count, 0);
            reader.read_exact(&mut uploaded[start..]).unwrap();
            let mut newline = [0u8; 2];
            reader.read_exact(&mut newline).unwrap();
            assert_eq!(newline, *b"\r\n");
        }
        assert_eq!(
            uploaded,
            [0.25f32, -0.5, 1.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        // Tail outputs are sent only once the request producer has gone away (EOF).
        send_event(
            &mut response,
            json!({"type": "partial", "segment_index": 0, "text": " hello"}),
        );
        send_event(
            &mut response,
            json!({"type": "segment", "segment_index": 0, "start": 42, "end": 78, "text": " hello"}),
        );
        send_event(&mut response, json!({"type": "result", "segments": 1, "samples": 3}));
        response.write_all(b"0\r\n\r\n").unwrap();
    });
    let (upload, receiver) = mpsc::channel(2);
    let response = open_response(
        &reqwest::Client::new(),
        &base,
        "en",
        Some(4),
        receiver,
        Arc::new(Control::default()),
    )
    .await
    .unwrap();
    let bytes = response.bytes_stream().map(|r| r.map_err(std::io::Error::other));
    let mut lines = FramedRead::new(StreamReader::new(bytes), LinesCodec::new());
    let mut upload = Some(upload);
    let mut calibrated = false;
    let mut saw_result = false;
    let mut segments = Vec::new();
    while let Some(line) = lines.next().await {
        let event: Value = serde_json::from_str(&line.unwrap()).unwrap();
        match event["type"].as_str().unwrap() {
            "calibration" => {
                serde_json::from_value::<Profile>(event).unwrap().validate().unwrap();
                calibrated = true;
            }
            "ready" => {
                assert!(calibrated);
                let mut tx = upload.take().unwrap();
                tx.try_send(Ok(bytes::Bytes::from(
                    [0.25f32, -0.5, 1.0]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )))
                .unwrap();
                drop(tx); // normal stop's upload EOF
            }
            "segment" => segments.push(serde_json::from_value::<LiveSegment>(event).unwrap().into_segment().unwrap()),
            "result" => {
                assert_eq!(event["segments"], 1);
                saw_result = true;
            }
            "loading" | "partial" => {}
            other => panic!("unexpected event: {other}"),
        }
    }
    assert!(saw_result);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].stop, 78);
    tokio::task::spawn_blocking(move || server.join().unwrap()).await.unwrap();
}

#[tokio::test]
async fn stop_while_http_startup_is_pending_cancels_and_closes_request() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (accepted, wait_for_accept) = tokio::sync::oneshot::channel();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut reader = BufReader::new(stream);
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
        }
        accepted.send(()).unwrap();
        // No response headers, calibration or ready. Cancellation must not wait for them.
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).unwrap();
        assert!(bytes.is_empty() || bytes == b"0\r\n\r\n");
    });
    let control = Arc::new(Control::default());
    let stopping = control.clone();
    let stop = tokio::spawn(async move {
        wait_for_accept.await.unwrap();
        stopping.stop();
    });
    let (upload, receiver) = mpsc::channel(2);
    let client = reqwest::Client::new();
    let stopped = tokio::select! {
        biased;
        _ = control.changed.notified() => {
            control.abort_upload();
            true
        },
        response = open_response(&client, &base, "en", None, receiver, control.clone()) => {
            panic!("Startup unexpectedly completed: {response:?}")
        }
    };
    drop(upload);
    assert!(stopped);
    assert_eq!(control.state(), STOP);
    stop.await.unwrap();
    tokio::task::spawn_blocking(move || server.join().unwrap()).await.unwrap();
}
