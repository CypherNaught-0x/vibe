use super::*;

fn drained(mixer: &mut TimelineMixer) -> Vec<f32> {
    let mut out = Vec::new();
    mixer.take_ready(&mut out);
    out
}

#[test]
fn unknown_or_missing_formats_mean_opus() {
    assert_eq!(RecordingFormat::parse(None), RecordingFormat::Opus);
    assert_eq!(RecordingFormat::parse(Some("ogg-vorbis")), RecordingFormat::Opus);
    assert_eq!(RecordingFormat::parse(Some(" WAV ")), RecordingFormat::Wav);
    for (name, format) in [
        ("opus", RecordingFormat::Opus),
        ("wav", RecordingFormat::Wav),
        ("flac", RecordingFormat::Flac),
        ("mp3", RecordingFormat::Mp3),
        ("m4a", RecordingFormat::M4a),
    ] {
        assert_eq!(RecordingFormat::parse(Some(name)), format);
        assert_eq!(format.extension(), name);
    }
}

#[test]
fn a_single_source_is_released_immediately_and_downmixed() {
    let mut mixer = TimelineMixer::new(1, 100, 1000);
    mixer.push(0, 2, &[1.0f32, 0.0, 0.5, 0.5, f32::NAN, 0.0]).unwrap();
    assert_eq!(drained(&mut mixer), vec![0.5, 0.5, 0.0]);
    assert!(drained(&mut mixer).is_empty());
}

#[test]
fn sources_are_summed_by_sample_index_and_wait_for_each_other() {
    let mut mixer = TimelineMixer::new(2, 100, 1000);
    mixer.push(0, 1, &[0.1f32, 0.2, 0.3]).unwrap();
    // The second source has not delivered yet: nothing is final.
    assert!(drained(&mut mixer).is_empty());
    mixer.push(1, 1, &[0.01f32, 0.02]).unwrap();
    let out = drained(&mut mixer);
    assert_eq!(out.len(), 2);
    assert!((out[0] - 0.11).abs() < 1e-6 && (out[1] - 0.22).abs() < 1e-6);
    mixer.push(1, 1, &[0.03f32]).unwrap();
    assert!((drained(&mut mixer)[0] - 0.33).abs() < 1e-6);
    // Integer samples are scaled like the WAV writer's float view of them.
    mixer.push(0, 1, &[i16::MAX]).unwrap();
    mixer.push(1, 1, &[0i16]).unwrap();
    assert!((drained(&mut mixer)[0] - 1.0).abs() < 1e-3);
}

#[test]
fn a_silent_source_stops_holding_the_output_back_and_rejoins_at_the_leading_edge() {
    let mut mixer = TimelineMixer::new(2, 4, 1000);
    mixer.push(0, 1, &[1.0f32; 10]).unwrap();
    // Source 1 never delivered: everything older than the lag window is released as is.
    assert_eq!(drained(&mut mixer), vec![1.0; 6]);
    // Its next buffer is the latest audio, so it lands on the newest frames, not in the past.
    mixer.push(1, 1, &[0.5f32, 0.5]).unwrap();
    mixer.push(0, 1, &[1.0f32; 2]).unwrap();
    let mut rest = Vec::new();
    mixer.take_all(&mut rest);
    assert_eq!(rest, vec![1.0, 1.0, 1.5, 1.5, 1.0, 1.0]);
}

#[test]
fn take_all_keeps_the_longest_source() {
    let mut mixer = TimelineMixer::new(2, 100, 1000);
    mixer.push(0, 1, &[0.25f32; 5]).unwrap();
    mixer.push(1, 1, &[0.25f32; 2]).unwrap();
    let mut out = Vec::new();
    mixer.take_all(&mut out);
    assert_eq!(out, vec![0.5, 0.5, 0.25, 0.25, 0.25]);
}

#[test]
fn an_undrained_mixer_reports_overflow_instead_of_growing() {
    let mut mixer = TimelineMixer::new(1, 100, 4);
    assert!(mixer.push(0, 1, &[0.0f32; 4]).is_ok());
    assert!(mixer.push(0, 1, &[0.0f32; 1]).is_err());
}

fn ffmpeg_available() -> bool {
    if crate::ffmpeg::find_ffmpeg_path().is_none() {
        eprintln!("skipping: ffmpeg not found");
        return false;
    }
    true
}

/// Decode any stored recording back to 16 kHz mono and count its samples.
fn decoded_samples(path: &Path) -> u32 {
    let wav = path.with_extension("check.wav");
    crate::ffmpeg::transcode_recording(&[path.to_path_buf()], RecordingFormat::Wav.output_args(), &wav).unwrap();
    let samples = hound::WavReader::open(&wav).unwrap().len();
    std::fs::remove_file(wav).ok();
    samples
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vibe-recording-encoder-{name}-{}", crate::ffmpeg::random_string(6)));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn live_encoder_writes_every_format_from_two_sources() {
    if !ffmpeg_available() {
        return;
    }
    let dir = temp_dir("formats");
    let rate = 48_000;
    for format in [
        RecordingFormat::Opus,
        RecordingFormat::Wav,
        RecordingFormat::Flac,
        RecordingFormat::Mp3,
        RecordingFormat::M4a,
    ] {
        let output = dir.join(format!("live.{}", format.extension()));
        let encoder = match LiveEncoder::start(rate, &[1, 2], format, output.clone()) {
            Ok(encoder) => encoder,
            Err(error) => panic!("{format:?}: {error:#}"),
        };
        let (mic, system) = (encoder.sink(0), encoder.sink(1));
        // Two seconds in 10 ms callbacks: a mono tone and a stereo one.
        for block in 0..200 {
            let mono: Vec<f32> = (0..480).map(|i| ((block * 480 + i) as f32 * 0.05).sin() * 0.3).collect();
            let stereo: Vec<f32> = mono.iter().flat_map(|s| [*s, -*s * 0.5]).collect();
            mic.push(&mono);
            system.push(&stereo);
        }
        let stored = encoder.finish().unwrap_or_else(|error| panic!("{format:?}: {error:#}"));
        let seconds = decoded_samples(&stored) as f32 / 16_000.0;
        // Lossy codecs pad the edges a little.
        assert!((1.9..2.2).contains(&seconds), "{format:?}: {seconds}s");
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_failed_live_encoder_reports_and_leaves_no_partial_file() {
    if !ffmpeg_available() {
        return;
    }
    let dir = temp_dir("failed");
    let output = dir.join("missing-folder").join("live.opus");
    let encoder = LiveEncoder::start(48_000, &[1], RecordingFormat::Opus, output.clone()).unwrap();
    encoder.sink(0).push(&[0.0f32; 4800]);
    assert!(encoder.finish().is_err());
    assert!(!output.exists());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn transcode_mixes_two_captures_into_the_stored_format() {
    if !ffmpeg_available() {
        return;
    }
    let dir = temp_dir("transcode");
    let write = |name: &str, rate: u32, channels: u16, seconds: u32| {
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..rate * seconds * channels as u32 {
            writer.write_sample((i as f32 * 0.01).sin() * 0.2).unwrap();
        }
        writer.finalize().unwrap();
        path
    };
    // Different rates are exactly the case the live encoder leaves to this path.
    let inputs = [write("mic.wav", 48_000, 1, 2), write("system.wav", 44_100, 2, 1)];
    let output = dir.join("stored.opus");
    crate::ffmpeg::transcode_recording(&inputs, RecordingFormat::Opus.output_args(), &output).unwrap();
    let seconds = decoded_samples(&output) as f32 / 16_000.0;
    assert!((1.9..2.2).contains(&seconds), "{seconds}s");
    std::fs::remove_dir_all(dir).ok();
}

/// Stop latency of a realistic long recording: feed `VIBE_BENCH_MINUTES` (default 60) of two
/// sources as fast as ffmpeg takes it, let the pump settle as it would in real time, then time
/// `finish`. `cargo test --release recording_encoder::tests::bench -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_stop_latency() {
    if !ffmpeg_available() {
        return;
    }
    let minutes: usize = std::env::var("VIBE_BENCH_MINUTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let dir = temp_dir("bench");
    for format in [RecordingFormat::Opus, RecordingFormat::Wav] {
        let output = dir.join(format!("bench.{}", format.extension()));
        let encoder = LiveEncoder::start(48_000, &[1, 2], format, output).unwrap();
        let (mic, system) = (encoder.sink(0), encoder.sink(1));
        let fed = Instant::now();
        let mono: Vec<f32> = (0..480).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let stereo: Vec<f32> = mono.iter().flat_map(|s| [*s, *s]).collect();
        for block in 0..minutes * 60 * 100 {
            mic.push(&mono);
            system.push(&stereo);
            // Keep the backlog below the overflow guard, as real-time capture would.
            if block % 1000 == 999 {
                while encoder.mixer.lock().unwrap().frames.len() > 48_000 * 5 {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        while !encoder.mixer.lock().unwrap().frames.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
        }
        // Real time gives ffmpeg the whole recording to catch up; give it a moment here.
        std::thread::sleep(Duration::from_secs(2));
        let encode_time = fed.elapsed();
        let stop = Instant::now();
        let stored = encoder.finish().unwrap();
        println!(
            "{format:?}: {minutes} min encoded live in {encode_time:?} (background), stop took {:?}, {} bytes",
            stop.elapsed(),
            std::fs::metadata(&stored).unwrap().len()
        );
    }
    std::fs::remove_dir_all(dir).ok();
}
