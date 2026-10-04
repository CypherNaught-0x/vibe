use eyre::{bail, Context, ContextCompat, Result};
use serde::{Deserialize, Serialize};
use std::{io::Read, path::Path};

pub const MODEL_FILENAME: &str = "Audio8-ASR-Infinite.audio8.json";
const CHECKPOINT: &str = "Edge0/Audio8-ASR-Infinite";
// Verified against https://huggingface.co/api/models/Edge0/Audio8-ASR-Infinite.
const REVISION: &str = "7476824bc222e4ad509d286e8cae8b8d3f371129";
const MAX_MARKER_BYTES: u64 = 4096;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelMarker {
    engine: String,
    checkpoint: String,
    revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    artifacts: Option<String>,
}

fn pinned_marker() -> ModelMarker {
    ModelMarker {
        engine: "audio8".into(),
        checkpoint: CHECKPOINT.into(),
        revision: REVISION.into(),
        backend: None,
        artifacts: None,
    }
}

fn validate_marker_bytes(bytes: &[u8]) -> Result<()> {
    let marker: ModelMarker = serde_json::from_slice(bytes).context("Invalid Audio8 model marker")?;
    if marker.engine != "audio8" || marker.checkpoint != CHECKPOINT || marker.revision != REVISION {
        bail!("Audio8 requires the supported, pinned Edge0/Audio8-ASR-Infinite checkpoint");
    }
    match marker.backend.as_deref().unwrap_or("torch") {
        "torch" if marker.artifacts.is_none() => {}
        "onnx" | "mlx" => {
            let relative = Path::new(marker.artifacts.as_deref().context("Converted Audio8 requires artifacts")?);
            if relative.as_os_str().is_empty()
                || relative
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_) | std::path::Component::CurDir))
            {
                bail!("Audio8 artifacts must be a relative subdirectory beside the marker");
            }
            if marker.backend.as_deref() == Some("mlx") && !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                bail!("Audio8 MLX requires macOS on Apple Silicon");
            }
        }
        _ => bail!("Unsupported Audio8 backend/selection fields"),
    }
    Ok(())
}

pub(crate) fn validate_model_marker(path: &Path) -> Result<()> {
    let file = std::fs::File::open(path).context("Cannot open Audio8 model marker")?;
    if !file.metadata()?.is_file() {
        bail!("Audio8 model marker must be a regular file");
    }
    let mut bytes = Vec::new();
    file.take(MAX_MARKER_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_MARKER_BYTES {
        bail!("Audio8 model marker is too large");
    }
    validate_marker_bytes(&bytes)?;
    let marker: ModelMarker = serde_json::from_slice(&bytes)?;
    if let Some(relative) = marker.artifacts {
        let parent = path.parent().unwrap_or(Path::new(".")).canonicalize()?;
        let directory = parent
            .join(relative)
            .canonicalize()
            .context("Audio8 converted artifacts are missing")?;
        if !directory.is_dir() || !directory.starts_with(&parent) {
            bail!("Audio8 artifacts must remain beside the marker");
        }
    }
    Ok(())
}

/// Add only a local descriptor: the runtime installs dependencies and weights on first load.
/// No checkpoint, revision or remote-code setting is accepted from the frontend.
#[tauri::command]
pub async fn add_audio8_model(app_handle: tauri::AppHandle) -> crate::error::CommandResult<String> {
    which::which("uv")
        .context("Audio8 requires uv installed and available on PATH. Install it from https://docs.astral.sh/uv/")?;
    let folder = super::app::get_models_folder(app_handle)?;
    std::fs::create_dir_all(&folder).context("Cannot create models folder")?;
    let path = folder.join(MODEL_FILENAME);
    if std::fs::symlink_metadata(&path).is_ok_and(|metadata| !metadata.file_type().is_file()) {
        return Err(eyre::eyre!("Cannot replace Audio8 marker: the destination is not a regular file").into());
    }
    if validate_model_marker(&path).is_err() {
        std::fs::write(&path, serde_json::to_vec_pretty(&pinned_marker())?).context("Cannot write Audio8 model marker")?;
    }
    Ok(path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::{pinned_marker, validate_marker_bytes, CHECKPOINT, REVISION};
    use serde_json::json;

    #[test]
    fn accepts_only_the_pinned_marker() {
        assert!(validate_marker_bytes(&serde_json::to_vec(&pinned_marker()).unwrap()).is_ok());
        for marker in [
            json!({"engine": "whisper", "checkpoint": CHECKPOINT, "revision": REVISION}),
            json!({"engine": "audio8", "checkpoint": "someone/other-model", "revision": REVISION}),
            json!({"engine": "audio8", "checkpoint": CHECKPOINT, "revision": "main"}),
            json!({"engine": "audio8", "checkpoint": CHECKPOINT, "revision": REVISION, "trust_remote_code": true}),
        ] {
            assert!(validate_marker_bytes(&serde_json::to_vec(&marker).unwrap()).is_err());
        }
    }

    #[test]
    fn rejects_partial_or_invalid_json() {
        assert!(validate_marker_bytes(b"{}").is_err());
        assert!(validate_marker_bytes(b"not json").is_err());
    }
}
