//! The `nemo-speech serve` child: started through a guard so it never outlives this server,
//! bound to localhost with a random API key, and fed audio in memory over chat completions.
use super::speakers::{single_tagged_speaker, speaker_runs, split_text, TimedWord};
use super::{MAX_CHUNK_SECONDS, SAMPLE_RATE};
use anyhow::{bail, Context as _};
use base64::Engine as _;
use serde::Deserialize;
use std::{
    collections::VecDeque,
    hash::{BuildHasher as _, Hasher as _},
    io::{BufRead, BufReader, Cursor},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use whisper_rs::Segment;

const DIAGNOSTIC_LINES: usize = 40;

pub(super) struct Server {
    /// The guard process; closing `guard_input` makes it stop `nemo-speech`.
    child: Child,
    guard_input: Option<ChildStdin>,
    base: String,
    key: String,
    lines: Arc<Mutex<VecDeque<String>>>,
}

impl Server {
    pub(super) fn start(
        binary: &Path,
        decoder: &Path,
        encoder: &Path,
        device: &str,
        aborted: &mut dyn FnMut() -> bool,
    ) -> anyhow::Result<Self> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
        let key = random_key();
        let context = context_tokens(MAX_CHUNK_SECONDS as f64 * 1.2).to_string();
        // The guard stops nemo-speech when this process dies, even when it is killed outright:
        // the 20+ GB child must never outlive it.
        let mut command = Command::new(std::env::current_exe().context("could not locate vibe-server")?);
        command
            .arg(GUARD_ARG)
            .arg(binary)
            .arg("serve")
            .arg("--asr-model")
            .arg(decoder)
            .arg("--salm-encoder")
            .arg(encoder)
            .args(["--host", "127.0.0.1", "--port", &port.to_string(), "--api-key", &key])
            .args(["--no-ui", "--no-warmup", "--device", device])
            .args(["--read-timeout", "3600", "--write-timeout", "3600"])
            // One request at a time: no KV cache is reserved for parallel sequences.
            .args(["--asr.salm.max_sequences", "1"])
            .args(["--asr.salm.n_ctx_per_sequence", &context, "--asr.salm.max_context", &context])
            .env("NO_COLOR", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        tracing::info!(binary = %binary.display(), port, device, "starting NeMo-Speech.cpp");
        let mut child = command
            .spawn()
            .with_context(|| format!("could not start {}", binary.display()))?;
        let guard_input = child.stdin.take();
        let lines = Arc::new(Mutex::new(VecDeque::new()));
        for stream in [
            child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            child.stderr.take().map(|s| Box::new(s) as _),
        ]
        .into_iter()
        .flatten()
        {
            let lines = Arc::clone(&lines);
            thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if line.contains("unused tensor") {
                        continue;
                    }
                    tracing::debug!("nemo-speech: {line}");
                    let mut lines = lines.lock().unwrap_or_else(|error| error.into_inner());
                    if lines.len() == DIAGNOSTIC_LINES {
                        lines.pop_front();
                    }
                    lines.push_back(line);
                }
            });
        }
        let mut server = Self {
            child,
            guard_input,
            base: format!("http://127.0.0.1:{port}"),
            key,
            lines,
        };
        server.wait_ready(aborted)?;
        Ok(server)
    }

    fn wait_ready(&mut self, aborted: &mut dyn FnMut() -> bool) -> anyhow::Result<()> {
        let limit = std::env::var("VIBE_NEMO_SPEECH_STARTUP_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20 * 60);
        let started = Instant::now();
        let url = format!("{}/ready", self.base);
        // The decoder is memory-mapped: a first load after a restart reads it all from disk.
        loop {
            if aborted() {
                bail!("Nemotron startup cancelled");
            }
            let ready = off_runtime(|| {
                reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(2))
                    .build()
                    .and_then(|client| client.get(&url).send())
                    .and_then(|response| response.json::<serde_json::Value>())
                    .is_ok_and(|body| body["ready"] == true)
            });
            if ready {
                tracing::info!(seconds = started.elapsed().as_secs(), "NeMo-Speech.cpp ready");
                return Ok(());
            }
            if self.child.try_wait()?.is_some() {
                bail!("NeMo-Speech.cpp stopped while loading the model: {}", self.diagnostics());
            }
            if started.elapsed().as_secs() > limit {
                bail!(
                    "NeMo-Speech.cpp did not become ready within {limit} s: {}",
                    self.diagnostics()
                );
            }
            thread::sleep(Duration::from_millis(250));
        }
    }

    pub(super) fn is_running(&mut self) -> bool {
        self.child.try_wait().is_ok_and(|status| status.is_none())
    }

    pub(super) fn diagnostics(&self) -> String {
        let lines = self.lines.lock().unwrap_or_else(|error| error.into_inner());
        let tail: Vec<&str> = lines.iter().rev().take(8).rev().map(String::as_str).collect();
        if tail.is_empty() {
            "no output".to_owned()
        } else {
            tail.join(" | ")
        }
    }

    /// One chat-completions request, streamed so generated text can drive progress. Runs on a
    /// worker thread so cancellation never waits for the model; the caller then drops the server.
    pub(super) fn transcribe(
        &self,
        samples: &[f32],
        instruction: &str,
        language: Option<&str>,
        progress: &mut dyn FnMut(usize),
        aborted: &mut dyn FnMut() -> bool,
    ) -> anyhow::Result<Transcription> {
        let body = request_body(samples, instruction, language)?;
        let url = format!("{}/v1/chat/completions", self.base);
        let key = self.key.clone();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let progress_sender = sender.clone();
            let result = stream_request(&url, &key, body, |chars| {
                let _ = progress_sender.send(Message::Progress(chars));
            });
            let _ = sender.send(Message::Done(result));
        });
        loop {
            if aborted() {
                bail!("Nemotron transcription aborted");
            }
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(Message::Progress(chars)) => progress(chars),
                Ok(Message::Done(result)) => {
                    return result
                        .map_err(|error| error.context(format!("Nemotron transcription failed ({})", self.diagnostics())))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("Nemotron request thread stopped"),
            }
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // End of input tells the guard to stop nemo-speech; kill the guard only if it hangs,
        // since killing it first would orphan the model process.
        drop(self.guard_input.take());
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if !matches!(self.child.try_wait(), Ok(None)) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        tracing::warn!("NeMo-Speech.cpp guard did not stop in time; killing it");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// argv[1] that turns vibe-server into the guard of one `nemo-speech` process.
pub const GUARD_ARG: &str = "__nemo-speech-guard";

/// Run `<binary> <args...>` until it exits or until stdin closes, which happens when the
/// server that started the guard exits, crashes or is killed. Then stop it and exit.
pub fn run_guard(mut args: impl Iterator<Item = std::ffi::OsString>) -> ! {
    use std::sync::atomic::{AtomicBool, Ordering};
    let Some(binary) = args.next() else {
        eprintln!("{GUARD_ARG}: missing program");
        std::process::exit(2);
    };
    let mut command = Command::new(&binary);
    command.args(args).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            eprintln!("could not start {}: {error}", Path::new(&binary).display());
            std::process::exit(1);
        }
    };
    let parent_gone = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&parent_gone);
    thread::spawn(move || {
        let _ = std::io::copy(&mut std::io::stdin(), &mut std::io::sink());
        flag.store(true, Ordering::Relaxed);
    });
    loop {
        match child.try_wait() {
            Ok(Some(status)) => std::process::exit(status.code().unwrap_or(1)),
            Ok(None) if parent_gone.load(Ordering::Relaxed) => {
                let _ = child.kill();
                let _ = child.wait();
                std::process::exit(0);
            }
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(_) => std::process::exit(1),
        }
    }
}

enum Message {
    Progress(usize),
    Done(anyhow::Result<Transcription>),
}

fn request_body(samples: &[f32], instruction: &str, language: Option<&str>) -> anyhow::Result<Vec<u8>> {
    let mut wav = Cursor::new(Vec::with_capacity(samples.len() * 2 + 44));
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut wav, spec)?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }
    let audio = base64::engine::general_purpose::STANDARD.encode(wav.into_inner());
    let mut body = serde_json::json!({
        "model": "salm",
        "stream": true,
        "timestamp_granularities": ["word"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
        ]}],
    });
    if let Some(language) = language {
        body["language"] = language.into();
    }
    Ok(serde_json::to_vec(&body)?)
}

fn stream_request(url: &str, key: &str, body: Vec<u8>, mut progress: impl FnMut(usize)) -> anyhow::Result<Transcription> {
    let response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()?
        .post(url)
        .bearer_auth(key)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().unwrap_or_default();
        bail!("HTTP {status}: {}", error_message(&text));
    }
    let mut generated = 0;
    for line in BufReader::new(response).lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let chunk: StreamChunk = serde_json::from_str(data).context("NeMo-Speech.cpp sent an unreadable event")?;
        if let Some(error) = chunk.error {
            bail!("{}", error.message);
        }
        if let Some(transcription) = chunk.transcription {
            return Ok(transcription);
        }
        let text: usize = chunk
            .choices
            .iter()
            .filter_map(|choice| choice.delta.content.as_deref())
            .map(str::len)
            .sum();
        if text > 0 {
            generated += text;
            progress(generated);
        }
    }
    bail!("NeMo-Speech.cpp ended the stream without a transcript")
}

fn error_message(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value["error"]["message"].as_str().map(str::to_owned))
        .unwrap_or_else(|| body.chars().take(300).collect())
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<Choice>,
    transcription: Option<Transcription>,
    error: Option<ErrorBody>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ErrorBody {
    message: String,
}

#[derive(Deserialize, Debug)]
pub(super) struct Transcription {
    #[serde(default)]
    text: String,
    #[serde(default)]
    duration: f64,
    #[serde(default)]
    subtitle_cues: Vec<Cue>,
    /// Each word's time and the speaker the encoder's speaker branch heard say it.
    #[serde(default)]
    words: Vec<TimedWord>,
}

#[derive(Deserialize, Debug)]
struct Cue {
    start: f64,
    end: f64,
    text: String,
    /// The cue's words, as indices into [`Transcription::words`].
    first_word: Option<usize>,
    last_word: Option<usize>,
}

impl Transcription {
    /// Subtitle cues as segments, in centiseconds; the whole text when the alignment failed.
    /// With `diarize`, a cue is split where its words change speaker and every piece carries
    /// its speaker.
    pub(super) fn segments(&self, offset: f64, diarize: bool) -> Vec<Segment> {
        let segment = |start: f64, end: f64, text: &str, speaker: Option<usize>| {
            let text = clean_text(text);
            (!text.is_empty()).then(|| Segment {
                start: ((start + offset) * 100.0).round() as i64,
                end: ((end.max(start) + offset) * 100.0).round() as i64,
                text,
                no_speech_prob: 0.0,
                // Word timings only steer speaker splits here; segments carry none.
                words: Vec::new(),
                speaker: speaker.filter(|_| diarize),
            })
        };
        if self.subtitle_cues.is_empty() {
            // Without the alignment there is no time to split at; the model's own tags still
            // name the speaker when only one speaks.
            let speaker = single_tagged_speaker(&self.text);
            return segment(0.0, self.duration, &self.text, speaker).into_iter().collect();
        }
        let mut segments = Vec::new();
        for cue in &self.subtitle_cues {
            let words = match (cue.first_word, cue.last_word) {
                (Some(first), Some(last)) if diarize && first <= last => self.words.get(first..=last).unwrap_or_default(),
                _ => &[],
            };
            let runs = speaker_runs(words);
            if runs.len() <= 1 {
                let speaker = runs
                    .first()
                    .map(|run| run.speaker)
                    .or_else(|| single_tagged_speaker(&cue.text));
                segments.extend(segment(cue.start, cue.end, &cue.text, speaker));
                continue;
            }
            let pieces = split_text(&cue.text, words, &runs);
            for (index, (run, text)) in runs.iter().zip(pieces).enumerate() {
                let start = if index == 0 { cue.start } else { words[run.first].start };
                let end = if index == runs.len() - 1 {
                    cue.end
                } else {
                    words[run.last].end
                };
                segments.extend(segment(start, end, &text, Some(run.speaker)));
            }
        }
        segments
    }
}

/// Text without the speaker and event tags the model writes (`<spk:0>`), on one line.
fn clean_text(text: &str) -> String {
    let text = strip_tags(text).split_whitespace().collect::<Vec<_>>().join(" ");
    // Like Whisper, a segment starts with the space that separates it from the previous one,
    // except in scripts written without spaces.
    match text.chars().next() {
        Some(first) if !unspaced_script(first) => format!(" {text}"),
        _ => text,
    }
}

pub(super) fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for c in text.chars() {
        match c {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth > 0 => {}
            _ => out.push(c),
        }
    }
    out
}

pub(super) fn unspaced_script(c: char) -> bool {
    matches!(c as u32,
        0x0E00..=0x0E7F // Thai
        | 0x3000..=0x30FF // CJK punctuation, Hiragana, Katakana
        | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF // CJK ideographs
        | 0xFF00..=0xFFEF) // full-width forms
}

/// KV cache cells for one request of `seconds`: 12.5 tokens per second of audio plus up to 6
/// per second of transcript, with room for the prompt.
fn context_tokens(seconds: f64) -> u32 {
    let tokens = (seconds.max(1.0) * 19.0).ceil() as u32 + 2_048;
    tokens.max(8_192).div_ceil(256) * 256
}

/// Run blocking HTTP on its own thread: `reqwest::blocking` panics on a tokio worker, and the
/// CLI transcribes on one.
fn off_runtime<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    thread::scope(|scope| scope.spawn(work).join().expect("HTTP thread panicked"))
}

fn random_key() -> String {
    let state = std::collections::hash_map::RandomState::new();
    (0..2)
        .map(|round| {
            let mut hasher = state.build_hasher();
            hasher.write_u64(round);
            hasher.write_u128(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
            );
            hasher.write_u32(std::process::id());
            format!("{:016x}", hasher.finish())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_follow_cues_without_tags_and_shift_by_offset() {
        let transcription: Transcription = serde_json::from_str(
            r#"{"text":"<spk:0> Hallo zusammen, heute.","duration":6.3,"subtitle_cues":[
                {"start":0.0,"end":4.197,"text":"<spk:0> Hallo zusammen,\nheute"},
                {"start":4.197,"end":6.13,"text":"sprechen wir."},
                {"start":6.2,"end":6.3,"text":"<spk:1>"}]}"#,
        )
        .unwrap();
        let segments = transcription.segments(10.0, false);
        assert_eq!(segments.len(), 2);
        assert_eq!((segments[0].start, segments[0].end), (1000, 1420));
        assert_eq!(segments[0].text, " Hallo zusammen, heute");
        assert_eq!((segments[1].start, segments[1].end), (1420, 1613));
    }

    #[test]
    fn chinese_segments_get_no_leading_space() {
        assert_eq!(clean_text("<spk:0>你好。"), "你好。");
        assert_eq!(clean_text("<spk:0>"), "");
    }

    #[test]
    fn unaligned_transcripts_become_one_segment() {
        let transcription: Transcription = serde_json::from_str(r#"{"text":"<spk:0> Hi there.","duration":2.5}"#).unwrap();
        let segments = transcription.segments(0.0, false);
        assert_eq!(segments.len(), 1);
        assert_eq!(
            (segments[0].start, segments[0].end, segments[0].text.as_str()),
            (0, 250, " Hi there.")
        );
    }

    fn word(word: &str, start: f64, end: f64, speaker: usize) -> String {
        format!(r#"{{"word":"{word}","start":{start},"end":{end},"speaker":"spk:{speaker}","speaker_slot":{speaker}}}"#)
    }

    fn spans(segments: &[Segment]) -> Vec<(i64, i64, &str, Option<usize>)> {
        segments
            .iter()
            .map(|s| (s.start, s.end, s.text.as_str(), s.speaker))
            .collect()
    }

    /// The shape NeMo-Speech.cpp 0.3.0 returned for a two-voice recording, trimmed.
    fn two_speakers() -> Transcription {
        let words = [
            word("Hello", 0.0, 0.478, 0),
            word("Daniel,", 0.478, 1.116, 0),
            word("Yes,", 4.08, 4.861, 1),
            word("I", 4.861, 5.021, 1),
            word("read", 5.021, 5.26, 1),
            word("it.", 5.26, 5.419, 1),
            word("Great.", 8.81, 9.404, 0),
        ];
        serde_json::from_str(&format!(
            r#"{{"text":"<spk:0> Hello Daniel, <spk:1> Yes, I read it. <spk:0> Great.","duration":11.87,
                "words":[{}],
                "subtitle_cues":[
                    {{"first_word":0,"last_word":1,"start":0.0,"end":3.93,"text":"Hello Daniel,"}},
                    {{"first_word":2,"last_word":6,"start":4.08,"end":11.69,"text":"Yes, I read\nit. Great."}}],
                "diarization_timestamps":[{{"speaker":0,"start":0.0,"end":3.93}}]}}"#,
            words.join(",")
        ))
        .unwrap()
    }

    #[test]
    fn native_speakers_label_cues_and_split_them_where_the_speaker_changes() {
        let segments = two_speakers().segments(1.0, true);
        assert_eq!(
            spans(&segments),
            [
                (100, 493, " Hello Daniel,", Some(0)),
                (508, 642, " Yes, I read it.", Some(1)),
                (981, 1269, " Great.", Some(0)),
            ]
        );
    }

    #[test]
    fn without_diarization_cues_stay_whole_and_unlabelled() {
        let segments = two_speakers().segments(0.0, false);
        assert_eq!(
            spans(&segments),
            [(0, 393, " Hello Daniel,", None), (408, 1169, " Yes, I read it. Great.", None)]
        );
    }

    #[test]
    fn an_unaligned_transcript_keeps_a_speaker_only_when_one_is_tagged() {
        let one: Transcription = serde_json::from_str(r#"{"text":"<spk:1> Hi. <spk:1> There.","duration":2.0}"#).unwrap();
        assert_eq!(one.segments(0.0, true)[0].speaker, Some(1));
        assert_eq!(one.segments(0.0, false)[0].speaker, None);
        let two: Transcription = serde_json::from_str(r#"{"text":"<spk:0> Hi. <spk:1> Yo.","duration":2.0}"#).unwrap();
        assert_eq!(two.segments(0.0, true)[0].speaker, None);
    }

    #[test]
    fn context_covers_a_full_chunk() {
        assert!(context_tokens(MAX_CHUNK_SECONDS as f64 * 1.2) as f64 >= MAX_CHUNK_SECONDS as f64 * 19.0);
        assert_eq!(context_tokens(1.0), 8_192);
    }
}
