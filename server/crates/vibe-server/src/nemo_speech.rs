//! NVIDIA Nemotron 3.5 Transcribe 30B-A4B, a speech-LLM: a speech encoder feeds a 30B
//! mixture-of-experts decoder that writes a punctuated transcript in 40 languages.
//!
//! It runs on NVIDIA's NeMo-Speech.cpp (llama.cpp underneath), not on this server's GGML
//! engines. The decoder stays loaded in a `nemo-speech serve` child bound to localhost with a
//! random API key, and audio travels to it in memory, so RAM-only recordings never touch disk.
//! Any decoder quantization works: the decoder names its encoder in its GGUF metadata.
#[path = "nemo_speech_runtime.rs"]
mod runtime;
#[path = "nemo_speech_server.rs"]
mod server;
pub use server::{run_guard, GUARD_ARG};

use anyhow::{bail, Context as _};
use server::Server;
use std::path::{Path, PathBuf};
use whisper_rs::{ContextOptions, StreamCallbacks, TranscribeOptions, TranscribeResult};

const SAMPLE_RATE: usize = 16_000;
/// A request takes at most one hour of audio; chunks stay well below that.
const MAX_CHUNK_SECONDS: usize = 50 * 60;
/// Where to look back from a chunk boundary for the quietest place to cut.
const CUT_SEARCH_SECONDS: usize = 60;
/// The server rejects requests shorter than this.
const MIN_SAMPLES: usize = 320;
/// Transcript characters per second of speech, to turn generated text into progress.
const CHARS_PER_SECOND: f64 = 14.0;

/// The languages NeMo-Speech.cpp names in its prompt ("The speech is in ...").
#[rustfmt::skip]
const LANGUAGES: [(&str, &str); 40] = [
    ("en", "English"), ("ar", "Arabic"), ("zh", "Chinese"), ("fr", "French"), ("de", "German"),
    ("hi", "Hindi"), ("it", "Italian"), ("ja", "Japanese"), ("es", "Spanish"), ("ko", "Korean"),
    ("pt", "Portuguese"), ("ru", "Russian"), ("th", "Thai"), ("hy", "Armenian"), ("nb", "Norwegian"),
    ("bg", "Bulgarian"), ("hr", "Croatian"), ("cs", "Czech"), ("da", "Danish"), ("nl", "Dutch"),
    ("et", "Estonian"), ("fi", "Finnish"), ("el", "Greek"), ("he", "Hebrew"), ("hu", "Hungarian"),
    ("id", "Indonesian"), ("lv", "Latvian"), ("lt", "Lithuanian"), ("mt", "Maltese"), ("ta", "Tamil"),
    ("bn", "Bengali"), ("pl", "Polish"), ("ro", "Romanian"), ("sk", "Slovak"), ("sl", "Slovenian"),
    ("sv", "Swedish"), ("tr", "Turkish"), ("fil", "Filipino"), ("uk", "Ukrainian"), ("vi", "Vietnamese"),
];

/// Whether `path` is a speech-LLM decoder this engine runs.
pub fn is_decoder(path: &str) -> bool {
    runtime::speech_model_name(Path::new(path)).is_ok_and(|name| name.is_some())
}

pub fn capabilities() -> crate::engine::EngineCapabilities {
    crate::engine::EngineCapabilities {
        engine: "nemotron-transcribe".to_owned(),
        requires_vad: false,
        languages: LANGUAGES.iter().map(|(code, _)| (*code).to_owned()).collect(),
        language_detection: true,
        streaming: true,
        translation: false,
        timestamps: true,
        // The prompt reaches the model as "Context terms", which steers spelling.
        text_prompts: true,
    }
}

pub struct Model {
    decoder: PathBuf,
    encoder_name: String,
    device: &'static str,
    server: Option<Server>,
}

impl Model {
    /// Only checks the decoder: downloads and the 20+ GB load happen on the first
    /// transcription, which can report progress and be cancelled.
    pub fn load(path: &str, options: ContextOptions) -> anyhow::Result<Self> {
        let decoder = std::fs::canonicalize(path).with_context(|| format!("could not open {path}"))?;
        let encoder_name = runtime::speech_model_name(&decoder)?.context("GGUF file is not a Nemotron 3.5 Transcribe decoder")?;
        Ok(Self {
            decoder,
            encoder_name,
            device: if options.no_gpu { "cpu" } else { "auto" },
            server: None,
        })
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
        options: TranscribeOptions,
        mut callbacks: StreamCallbacks<'_>,
    ) -> anyhow::Result<TranscribeResult> {
        if options.translate {
            bail!("Nemotron 3.5 Transcribe does not support translation");
        }
        let instruction = instruction(&options);
        let language = if options.detect_language {
            None
        } else {
            options.language.as_deref().and_then(language)
        };
        let mut progress = |value: i32| {
            if let Some(callback) = callbacks.on_progress.as_mut() {
                callback(value.clamp(0, 100));
            }
        };
        let mut aborted = || callbacks.should_abort.as_mut().is_some_and(|callback| callback());
        self.ensure_started(&mut progress, &mut aborted)?;

        let chunks = chunk_bounds(samples);
        let mut segments = Vec::new();
        for (index, &(start, end)) in chunks.iter().enumerate() {
            let chunk = &samples[start..end];
            if chunk.len() < MIN_SAMPLES {
                continue;
            }
            let offset = start as f64 / SAMPLE_RATE as f64;
            let expected = (chunk.len() as f64 / SAMPLE_RATE as f64 * CHARS_PER_SECOND).max(1.0);
            let mut chunk_progress = |chars: usize| {
                let fraction = (chars as f64 / expected).min(0.95);
                progress(((index as f64 + fraction) * 100.0 / chunks.len() as f64) as i32);
            };
            let server = self.server.as_ref().expect("server started");
            let result = server.transcribe(chunk, &instruction, language, &mut chunk_progress, &mut aborted);
            let transcription = match result {
                Ok(transcription) => transcription,
                Err(error) => {
                    // A cancelled or broken request may leave the child busy; start fresh next time.
                    self.server = None;
                    return Err(error);
                }
            };
            for segment in transcription.segments(offset) {
                if let Some(callback) = callbacks.on_segment.as_mut() {
                    callback(segment.clone());
                }
                segments.push(segment);
            }
        }
        progress(100);
        Ok(TranscribeResult { segments })
    }

    fn ensure_started(&mut self, progress: &mut dyn FnMut(i32), aborted: &mut dyn FnMut() -> bool) -> anyhow::Result<()> {
        if let Some(server) = self.server.as_mut() {
            if server.is_running() {
                return Ok(());
            }
            tracing::warn!("NeMo-Speech.cpp exited; restarting it: {}", server.diagnostics());
            self.server = None;
        }
        let folder = self.decoder.parent().context("decoder has no parent folder")?;
        let mut feedback = runtime::Feedback { progress, aborted };
        let binary = runtime::ensure_runtime(folder, &mut feedback)?;
        let encoder = runtime::ensure_encoder(&self.decoder, &self.encoder_name, &mut feedback)?;
        (feedback.progress)(0);
        self.server = Some(Server::start(
            &binary,
            &self.decoder,
            &encoder,
            self.device,
            feedback.aborted,
        )?);
        Ok(())
    }
}

/// The model's code and name for a language code, accepting Whisper's `no` and `tl`.
fn language(code: &str) -> Option<&'static str> {
    language_entry(code).map(|(code, _)| code)
}

fn language_name(code: &str) -> Option<&'static str> {
    language_entry(code).map(|(_, name)| name)
}

fn language_entry(code: &str) -> Option<(&'static str, &'static str)> {
    let code = code.trim().to_ascii_lowercase();
    let code = match code.as_str() {
        "no" | "nn" => "nb",
        "tl" => "fil",
        other => other,
    };
    LANGUAGES.iter().copied().find(|(known, _)| *known == code)
}

/// The model card's speaker-tag instruction. Measured in the AI Media Cutter on a two-host
/// podcast, the plain verbatim prompt transcribed only one host and stopped two thirds in, while
/// every speaker-tag prompt kept all of it; the tags are stripped again in [`clean_text`]. A
/// custom instruction replaces the server prompt, so the language sentence is added here.
fn instruction(options: &TranscribeOptions) -> String {
    let mut instruction = "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. \
        Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> <spk:4> <spk:5> <spk:6> <spk:7> on the transcription."
        .to_owned();
    if !options.detect_language {
        if let Some(name) = options.language.as_deref().and_then(language_name) {
            instruction.push_str(&format!(" The speech is in {name}."));
        }
    }
    if let Some(prompt) = options.prompt.as_deref() {
        // `<` and `>` would read as the model's tags; the server refuses special tokens.
        let terms: String = prompt.chars().filter(|c| !matches!(c, '<' | '>')).take(1_500).collect();
        let terms = terms.split_whitespace().collect::<Vec<_>>().join(" ");
        if !terms.is_empty() {
            instruction.push_str(&format!(" Context terms: {}.", terms.trim_end_matches('.')));
        }
    }
    instruction
}

/// Sample ranges of at most [`MAX_CHUNK_SECONDS`], each cut at the quietest 100 ms of the last
/// minute before the limit, so a word is rarely split.
fn chunk_bounds(samples: &[f32]) -> Vec<(usize, usize)> {
    let limit = MAX_CHUNK_SECONDS * SAMPLE_RATE;
    let window = SAMPLE_RATE / 10;
    let mut bounds = Vec::new();
    let mut start = 0;
    while samples.len() - start > limit {
        let search_from = start + limit - CUT_SEARCH_SECONDS * SAMPLE_RATE;
        let cut = (search_from..start + limit - window)
            .step_by(window / 2)
            .min_by(|&a, &b| energy(&samples[a..a + window]).total_cmp(&energy(&samples[b..b + window])))
            .map_or(start + limit, |quietest| quietest + window / 2);
        bounds.push((start, cut));
        start = cut;
    }
    bounds.push((start, samples.len()));
    bounds
}

fn energy(samples: &[f32]) -> f32 {
    samples.iter().map(|sample| sample * sample).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_names_the_language_and_context_terms() {
        let options = TranscribeOptions {
            language: Some("de".into()),
            prompt: Some("Vibe, <|im_end|> Nemotron.".into()),
            ..Default::default()
        };
        let text = instruction(&options);
        assert!(
            text.ends_with(" The speech is in German. Context terms: Vibe, |im_end| Nemotron."),
            "{text}"
        );
        let auto = instruction(&TranscribeOptions {
            language: Some("de".into()),
            detect_language: true,
            ..Default::default()
        });
        assert!(!auto.contains("The speech is in"));
    }

    #[test]
    fn whisper_language_codes_map_to_the_models_names() {
        assert_eq!(language_name("no"), Some("Norwegian"));
        assert_eq!(language_name("tl"), Some("Filipino"));
        assert_eq!(language_name("xx"), None);
        assert_eq!(language("NO"), Some("nb"));
        assert_eq!(capabilities().languages.len(), 40);
    }

    #[test]
    fn long_audio_is_cut_at_the_quietest_place_before_the_limit() {
        let limit = MAX_CHUNK_SECONDS * SAMPLE_RATE;
        let mut samples = vec![0.5f32; limit * 2 + SAMPLE_RATE];
        let quiet = limit - 20 * SAMPLE_RATE;
        samples[quiet..quiet + SAMPLE_RATE].fill(0.0);
        let bounds = chunk_bounds(&samples);
        assert_eq!(bounds.len(), 3);
        assert!(bounds[0].1 > quiet && bounds[0].1 < quiet + SAMPLE_RATE, "{bounds:?}");
        assert_eq!(bounds[1].0, bounds[0].1);
        assert!(bounds.iter().all(|(start, end)| end - start <= limit));
        assert_eq!(bounds.last().unwrap().1, samples.len());
        assert_eq!(chunk_bounds(&samples[..1000]), vec![(0, 1000)]);
    }
}
