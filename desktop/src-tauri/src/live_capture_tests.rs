use super::*;
use futures::StreamExt;

fn pcm(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn shared(ring: MixRing, resamplers: Vec<Resampler>) -> Arc<Shared> {
    Arc::new(Shared {
        producer: Mutex::new(Producer {
            ring,
            resamplers,
            started: None,
            accepting: false,
        }),
        wake: Condvar::new(),
        peak: AtomicU32::new(0),
        abandoned: AtomicBool::new(false),
        control: Arc::new(Control::default()),
    })
}

#[test]
fn mix_waits_for_all_sources_and_preserves_start_offsets_gaps_and_wraparound() {
    let mut ring = MixRing::new(8, 2).unwrap();
    for frame in 0..3 {
        ring.mix(frame, 0.25).unwrap();
    }
    ring.horizons[0] = 3;
    assert!(ring.take_chunk().is_none()); // loopback has not arrived yet
    ring.mix(2, 0.25).unwrap();
    ring.mix(3, 0.25).unwrap();
    ring.horizons[1] = 4;
    assert_eq!(pcm(&ring.take_chunk().unwrap()), vec![0.25, 0.25, 0.5]);
    assert!(ring.take_chunk().is_none()); // mic may still modify frame 3
    ring.mix(5, 0.5).unwrap();
    ring.horizons = vec![6, 6];
    assert_eq!(pcm(&ring.take_chunk().unwrap()), vec![0.25, 0.0, 0.5]);
    for frame in 6..14 {
        ring.mix(frame, 0.75).unwrap();
    }
    ring.horizons = vec![14, 14];
    assert_eq!(pcm(&ring.take_chunk().unwrap()), vec![0.75; 8]);
    assert_eq!(ring.samples.len(), 8);
}

#[test]
fn overflow_and_late_device_frames_fail_instead_of_overwriting_pcm() {
    let mut ring = MixRing::new(2, 1).unwrap();
    ring.mix(0, 0.25).unwrap();
    ring.mix(1, 0.5).unwrap();
    assert!(ring.mix(2, 0.75).unwrap_err().to_string().contains("overflow"));
    ring.horizons[0] = 2;
    assert_eq!(pcm(&ring.take_chunk().unwrap()), vec![0.25, 0.5]);
    assert!(ring.mix(1, 1.0).unwrap_err().to_string().contains("fatal lag"));
    assert!(ring.mix(usize::MAX, 0.0).is_err());
}

#[test]
fn shared_resampler_handles_fractional_rates_and_final_partial_without_wav() {
    let mut ring = MixRing::new(16_001, 1).unwrap();
    let mut resampler = Resampler::new(44_100, 2, 1).unwrap();
    for _ in 0..100 {
        resampler.push(&[0.5f32; 882], &mut ring).unwrap();
    }
    assert_eq!(resampler.position(), 16_000);
    ring.horizons[0] = resampler.position();
    let mut output = Vec::new();
    while let Some(bytes) = ring.take_chunk() {
        output.extend(pcm(&bytes));
    }
    assert_eq!(output, vec![0.5; 16_000]);
    let mut ring = MixRing::new(8, 1).unwrap();
    let mut resampler = Resampler::new(48_000, 2, 1).unwrap();
    resampler.push(&[0.25f32, 0.75], &mut ring).unwrap();
    assert_eq!(resampler.position(), 0);
    resampler.flush(&mut ring).unwrap();
    resampler.flush(&mut ring).unwrap(); // idempotent: no extra tail frame
    assert_eq!(resampler.position(), 1);
    ring.finished = true;
    assert_eq!(pcm(&ring.take_chunk().unwrap()), vec![0.5]);
}

#[test]
fn worker_flushes_tail_then_closes_upload_with_eof() {
    let mut ring = MixRing::new(8, 1).unwrap();
    let mut resampler = Resampler::new(48_000, 1, 1).unwrap();
    resampler.push(&[0.75f32], &mut ring).unwrap();
    resampler.flush(&mut ring).unwrap();
    ring.finished = true;
    let shared = shared(ring, vec![resampler]);
    shared.control.stop();
    let (tx, mut rx) = mpsc::channel(2);
    let worker = std::thread::spawn(move || upload_worker(shared, tx, None));
    futures::executor::block_on(async {
        assert_eq!(pcm(&rx.next().await.unwrap().unwrap()), vec![0.75]);
        assert!(rx.next().await.is_none());
    });
    worker.join().unwrap();
}

#[test]
fn cancellation_unblocks_worker_even_when_upload_is_backpressured() {
    let mut ring = MixRing::new(CHUNK_SAMPLES * 8, 1).unwrap();
    for frame in 0..CHUNK_SAMPLES * 8 {
        ring.mix(frame, 0.1).unwrap();
    }
    ring.horizons[0] = CHUNK_SAMPLES * 8;
    let shared = shared(ring, vec![]);
    let (tx, _rx) = mpsc::channel(0);
    let worker_shared = shared.clone();
    let worker = std::thread::spawn(move || upload_worker(worker_shared, tx, None));
    std::thread::sleep(Duration::from_millis(40));
    shared.control.cancel();
    worker.join().unwrap();
    assert_eq!(shared.control.state(), CANCEL);
}

#[tokio::test]
async fn stop_during_loading_is_not_lost_before_first_poll() {
    let control = Control::default();
    control.stop(); // listener fired before the HTTP/model select was first polled
    let selected = tokio::select! {
        biased;
        _ = control.changed.notified() => true,
        _ = futures::future::pending::<()>() => false,
    };
    assert!(selected);
    assert_eq!(control.state(), STOP);
    control.cancel();
    control.stop();
    assert_eq!(control.state(), CANCEL); // stop cannot downgrade a hard cancellation
}
