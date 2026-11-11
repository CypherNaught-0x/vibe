//! Speakers from Nemotron 3.5 Transcribe's own diarization. With word timestamps,
//! NeMo-Speech.cpp labels every word with the speaker the encoder's speaker branch heard
//! (`"speaker": "spk:0"`, up to eight); these helpers cut subtitle cues where that changes.
use super::server::{strip_tags, unspaced_script};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(super) struct TimedWord {
    #[serde(default)]
    pub(super) word: String,
    pub(super) start: f64,
    pub(super) end: f64,
    /// `spk:N`, numbered from 0.
    speaker: Option<String>,
    speaker_slot: Option<usize>,
}

impl TimedWord {
    pub(super) fn speaker(&self) -> Option<usize> {
        self.speaker
            .as_deref()
            .and_then(|label| label.strip_prefix("spk:")?.parse().ok())
            .or(self.speaker_slot)
    }
}

/// Consecutive words of one speaker, as indices into the cue's words.
#[derive(Debug, PartialEq)]
pub(super) struct Run {
    pub(super) first: usize,
    pub(super) last: usize,
    pub(super) speaker: usize,
}

/// A word without a speaker (the alignment could not place it) stays with the run before
/// it, or with the first run when it opens the cue.
pub(super) fn speaker_runs(words: &[TimedWord]) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (index, word) in words.iter().enumerate() {
        match (word.speaker(), runs.last_mut()) {
            (Some(speaker), Some(run)) if run.speaker != speaker => runs.push(Run {
                first: index,
                last: index,
                speaker,
            }),
            (_, Some(run)) => run.last = index,
            (Some(speaker), None) => runs.push(Run {
                first: 0,
                last: index,
                speaker,
            }),
            (None, None) => {}
        }
    }
    runs
}

/// The cue text cut where each run after the first begins, so punctuation and spacing stay as
/// the model wrote them. When a word cannot be found in order (overlapping speakers interleave
/// by time), the runs are rebuilt from their words instead.
pub(super) fn split_text(text: &str, words: &[TimedWord], runs: &[Run]) -> Vec<String> {
    let text: String = strip_tags(text).split_whitespace().collect::<Vec<_>>().join(" ");
    let mut positions = Vec::with_capacity(words.len());
    let mut cursor = 0;
    for word in words {
        let found = (!word.word.is_empty())
            .then(|| text[cursor..].find(word.word.as_str()))
            .flatten();
        match found {
            Some(at) => {
                positions.push(cursor + at);
                cursor += at + word.word.len();
            }
            None => {
                let separator = if words.iter().any(|word| word.word.chars().next().is_some_and(unspaced_script)) {
                    ""
                } else {
                    " "
                };
                return runs
                    .iter()
                    .map(|run| {
                        words[run.first..=run.last]
                            .iter()
                            .map(|word| word.word.as_str())
                            .collect::<Vec<_>>()
                            .join(separator)
                    })
                    .collect();
            }
        }
    }
    let mut cuts: Vec<usize> = runs.iter().skip(1).map(|run| positions[run.first]).collect();
    cuts.insert(0, 0);
    cuts.push(text.len());
    cuts.windows(2).map(|range| text[range[0]..range[1]].to_owned()).collect()
}

/// The speaker of `<spk:N>` tags when they all name the same one.
pub(super) fn single_tagged_speaker(text: &str) -> Option<usize> {
    let mut speakers = text
        .split("<spk:")
        .skip(1)
        .filter_map(|rest| rest.split_once('>')?.0.trim().parse::<usize>().ok());
    let first = speakers.next()?;
    speakers.all(|speaker| speaker == first).then_some(first)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unplaced_words_stay_with_the_speaker_before_them() {
        let words: Vec<TimedWord> = serde_json::from_str(
            r#"[{"word":"um","start":0,"end":1},{"word":"a","start":1,"end":2,"speaker":"spk:2"},
                {"word":"b","start":2,"end":3},{"word":"c","start":3,"end":4,"speaker_slot":1}]"#,
        )
        .unwrap();
        assert_eq!(
            speaker_runs(&words),
            [
                Run {
                    first: 0,
                    last: 2,
                    speaker: 2
                },
                Run {
                    first: 3,
                    last: 3,
                    speaker: 1
                }
            ]
        );
        assert!(speaker_runs(&words[..1]).is_empty());
    }

    #[test]
    fn words_out_of_text_order_rebuild_the_pieces_from_the_words() {
        let words: Vec<TimedWord> = serde_json::from_str(
            r#"[{"word":"yes","start":0,"end":1,"speaker":"spk:1"},{"word":"no","start":0.5,"end":1,"speaker":"spk:0"}]"#,
        )
        .unwrap();
        let runs = speaker_runs(&words);
        assert_eq!(split_text("No. Yes.", &words, &runs), ["yes", "no"]);
        let chinese: Vec<TimedWord> = serde_json::from_str(
            r#"[{"word":"你","start":0,"end":1,"speaker":"spk:0"},{"word":"好","start":1,"end":2,"speaker":"spk:1"}]"#,
        )
        .unwrap();
        assert_eq!(split_text("你好。", &chinese, &speaker_runs(&chinese)), ["你", "好。"]);
    }
}
