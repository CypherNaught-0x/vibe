//! Disk-free decoding. Non-streamable containers fail rather than spilling audio to disk.
use std::io::{Cursor, Read, Write};
use std::process::{Command, Stdio};

use anyhow::{bail, Context};

use super::{find_ffmpeg, try_read_native_wav, ReadOptions};

const MAX_DECODED_BYTES: u64 = 256 * 1024 * 1024;

pub(super) fn decode(bytes: Vec<u8>, options: ReadOptions) -> anyhow::Result<Vec<f32>> {
    if !options.enhance_audio {
        if let Ok(samples) = try_read_native_wav(&mut Cursor::new(&bytes)) {
            return Ok(samples);
        }
    }
    let ffmpeg = find_ffmpeg()?;
    // Replay the RAM buffer to discover all tracks. Never materialise an input file.
    let probe = run(&ffmpeg, &bytes, &["-hide_banner", "-nostdin", "-i", "pipe:0"], 1)?;
    let tracks = String::from_utf8_lossy(&probe.stderr)
        .lines()
        .filter(|line| line.trim_start().starts_with("Stream #") && line.contains("Audio:"))
        .count()
        .max(1);
    let mut args = vec![
        "-hide_banner".to_string(),
        "-loglevel".into(),
        "error".into(),
        "-nostdin".into(),
        "-i".into(),
        "pipe:0".into(),
    ];
    let enhance = "silenceremove=stop_periods=-1:stop_duration=0.7:stop_threshold=-45dB";
    if tracks > 1 {
        let inputs: String = (0..tracks).map(|i| format!("[0:a:{i}]")).collect();
        let mut graph = format!("{inputs}amix=inputs={tracks}:duration=longest:normalize=0[mix]");
        let output = if options.enhance_audio {
            graph.push_str(&format!(";[mix]{enhance}[out]"));
            "[out]"
        } else {
            "[mix]"
        };
        args.extend(["-filter_complex".into(), graph, "-map".into(), output.into()]);
    } else if options.enhance_audio {
        args.extend(["-af".into(), enhance.into()]);
    }
    args.extend(["-ar", "16000", "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"].map(String::from));
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let decoded = run(&ffmpeg, &bytes, &refs, MAX_DECODED_BYTES)?;
    if !decoded.status.success() {
        bail!("Audio cannot be decoded without disk storage; convert it to a streamable WAV/FLAC/WebM file first");
    }
    if decoded.stdout.is_empty() || decoded.stdout.len() % 4 != 0 {
        bail!("Disk-free audio decoder returned no complete samples");
    }
    let samples: Vec<f32> = decoded
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if samples.iter().any(|sample| !sample.is_finite()) {
        bail!("Audio decoder returned invalid samples");
    }
    Ok(samples)
}

struct Output {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: std::process::ExitStatus,
}

fn run(executable: &std::path::Path, input: &[u8], args: &[&str], limit: u64) -> anyhow::Result<Output> {
    let mut child = Command::new(executable)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to start disk-free audio decoder")?;
    let mut stdin = child.stdin.take().context("Audio decoder stdin missing")?;
    let stdout = child.stdout.take().context("Audio decoder stdout missing")?;
    let mut stderr = child.stderr.take().context("Audio decoder stderr missing")?;
    std::thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(input));
        let errors = scope.spawn(move || {
            let mut result = Vec::new();
            let mut buffer = [0u8; 4096];
            loop {
                let count = stderr.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                let keep = count.min(65536usize.saturating_sub(result.len()));
                result.extend_from_slice(&buffer[..keep]);
            }
            Ok::<_, std::io::Error>(result)
        });
        let mut result = Vec::new();
        let read = stdout.take(limit + 1).read_to_end(&mut result);
        if read.is_err() || result.len() as u64 > limit {
            let _ = child.kill();
        }
        let status = child.wait()?;
        let _ = writer.join(); // A probe may close stdin early; BrokenPipe is expected.
        let stderr = errors
            .join()
            .map_err(|_| anyhow::anyhow!("Audio decoder stderr reader failed"))??;
        read?;
        if result.len() as u64 > limit {
            bail!("Disk-free decoded audio exceeds the 256 MiB limit");
        }
        Ok(Output {
            stdout: result,
            stderr,
            status,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_wav_needs_no_external_decoder() {
        let bytes = include_bytes!("../../../../fixtures/short.wav").to_vec();
        let expected = try_read_native_wav(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(decode(bytes, ReadOptions::default()).unwrap(), expected);
    }

    #[test]
    fn stereo_wav_is_downmixed_through_pipes() {
        if find_ffmpeg().is_err() {
            return;
        }
        let mut buffer = Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut buffer, spec).unwrap();
        for _ in 0..1600 {
            writer.write_sample(8192i16).unwrap();
            writer.write_sample(8192i16).unwrap();
        }
        writer.finalize().unwrap();
        let samples = decode(buffer.into_inner(), ReadOptions::default()).unwrap();
        assert_eq!(samples.len(), 1600);
        // FFmpeg's stereo-to-mono matrix uses equal-power, not arithmetic-average gains.
        assert!((samples[800] - 0.25 * std::f32::consts::SQRT_2).abs() < 0.01);
    }
}
