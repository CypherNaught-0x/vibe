//! RAM-only duplex PCM capture. Return headers/events before reading the body:
//! the client starts capture only after model startup + calibration succeed.
use super::{error, AppState};
use crate::engine::Engine;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde::Deserialize;
use std::{convert::Infallible, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const FRAME_BYTES: usize = 2560 * 4;
const MAX_HTTP_FRAME_BYTES: usize = 256 * 1024;

#[derive(Deserialize)]
pub(super) struct LiveOptions {
    #[serde(default = "default_language")]
    language: String,
    #[serde(default = "default_threads")]
    n_threads: usize,
}
fn default_language() -> String {
    "en".into()
}
fn default_threads() -> usize {
    4
}

pub(super) async fn live(
    State(state): State<AppState>,
    Query(options): Query<LiveOptions>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    if !matches!(options.language.as_str(), "en" | "zh") || options.n_threads > 256 {
        return error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Audio8 live requires language en/zh and 0..256 threads",
        );
    }
    if headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()) != Some("application/octet-stream") {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request",
            "Send 16kHz mono little-endian f32 PCM as application/octet-stream",
        );
    }
    let mut lease = match state.unload_timeout.try_acquire(Arc::clone(&state.inner)) {
        Ok(lease) => lease,
        Err(_) => return error(StatusCode::CONFLICT, "model_busy", "The transcription model is in use"),
    };
    if !matches!(lease.ctx, Some(Engine::Audio8(_))) {
        return error(
            StatusCode::BAD_REQUEST,
            "unsupported_engine",
            "Live PCM capture currently requires an Audio8 model",
        );
    }
    let (events, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(32);
    let (pcm_tx, pcm_rx) = mpsc::channel(4);
    // The first body item ensures HTTP/1 peers get the response before uploading.
    let _ = send(&events, serde_json::json!({"type":"loading"}));
    let producer_events = events.clone();
    let producer = tokio::spawn(async move {
        let mut stream = body.into_data_stream();
        let mut tail = Vec::with_capacity(FRAME_BYTES);
        loop {
            let next = tokio::select! {
                _ = producer_events.closed() => return,
                next = stream.next() => next,
            };
            match next {
                Some(Ok(bytes)) if bytes.len() <= MAX_HTTP_FRAME_BYTES => {
                    for part in bytes.chunks(FRAME_BYTES) {
                        tail.extend_from_slice(part);
                        let complete = tail.len() / 4 * 4;
                        // Never retain a view into a larger transport allocation.
                        let ready = Bytes::copy_from_slice(&tail[..complete]);
                        tail.drain(..complete);
                        if !ready.is_empty() && pcm_tx.send(Ok(ready)).await.is_err() {
                            return;
                        }
                    }
                }
                Some(Ok(_)) => {
                    let _ = pcm_tx
                        .send(Err("Audio8 live HTTP frame exceeded its memory bound".into()))
                        .await;
                    return;
                }
                Some(Err(_)) => {
                    let _ = pcm_tx.send(Err("Audio8 live upload disconnected".into())).await;
                    return;
                }
                None => {
                    if !tail.is_empty() {
                        let _ = pcm_tx
                            .send(Err("Audio8 live upload ended with an incomplete f32 sample".into()))
                            .await;
                    }
                    return; // producer drop closes PCM; worker flushes model tail
                }
            }
        }
    });
    tokio::task::spawn_blocking(move || {
        let Some(Engine::Audio8(model)) = lease.ctx.as_mut() else {
            return;
        };
        let result = model.transcribe_live(
            &options.language,
            options.n_threads,
            pcm_rx,
            |event| {
                send(&events, event).map_err(|_| anyhow::anyhow!("Audio8 live client disconnected or consumed events too slowly"))
            },
            || events.is_closed(),
        );
        if let Err(err) = result {
            // No PCM/transcript values are included in these protocol errors.
            let _ = send(&events, serde_json::json!({"type":"error", "message":err.to_string()}));
        }
        producer.abort();
    });
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        Body::from_stream(ReceiverStream::new(receiver)),
    )
        .into_response()
}

fn send(tx: &mpsc::Sender<Result<Bytes, Infallible>>, value: serde_json::Value) -> Result<(), ()> {
    let mut line = serde_json::to_vec(&value).map_err(|_| ())?;
    line.push(b'\n');
    tx.try_send(Ok(Bytes::from(line))).map_err(|_| ())
}
