//! Embedded portable workers. A marker may select only pinned built-in code.
use anyhow::{bail, Context as _};
use std::{
    io::Write,
    path::{Component, Path},
};

pub(super) fn validate_selection(backend: Option<&str>, artifacts: Option<&str>, marker: &Path) -> anyhow::Result<()> {
    match backend.unwrap_or("torch") {
        "torch" if artifacts.is_none() => Ok(()),
        "onnx" | "mlx" => {
            let relative = Path::new(artifacts.context("Converted Audio8 marker requires artifacts")?);
            if relative.as_os_str().is_empty()
                || relative
                    .components()
                    .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
            {
                bail!("Audio8 artifacts must be a relative subdirectory beside the marker");
            }
            let parent = marker.parent().unwrap_or(Path::new(".")).canonicalize()?;
            let directory = parent
                .join(relative)
                .canonicalize()
                .context("Audio8 converted artifacts are missing")?;
            if !directory.is_dir() || !directory.starts_with(&parent) {
                bail!("Audio8 artifacts must remain beside the marker");
            }
            if backend == Some("mlx") && !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                bail!("Audio8 MLX requires macOS on Apple Silicon");
            }
            Ok(())
        }
        _ => bail!("Unsupported Audio8 backend/selection fields"),
    }
}

pub(super) fn install(directory: &Path, backend: &str) -> anyhow::Result<std::path::PathBuf> {
    let (entry, files): (&str, &[(&str, &str)]) = match backend {
        "onnx" => (
            "audio8_onnx_runtime.py",
            &[
                ("audio8_onnx_runtime.py", include_str!("audio8_onnx_runtime.py")),
                ("audio8_onnx/backend.py", include_str!("audio8_onnx/backend.py")),
            ],
        ),
        "mlx" => (
            "audio8_mlx_runtime.py",
            &[
                ("audio8_mlx_runtime.py", include_str!("audio8_mlx_runtime.py")),
                ("audio8_mlx/backend.py", include_str!("audio8_mlx/backend.py")),
                ("audio8_mlx/common.py", include_str!("audio8_mlx/common.py")),
                ("audio8_mlx/model.py", include_str!("audio8_mlx/model.py")),
            ],
        ),
        "torch" => return Ok(directory.join("audio8_runtime.py")),
        _ => bail!("Unsupported Audio8 backend"),
    };
    for (name, source) in files.iter().copied().chain(std::iter::once((
        "audio8_portable_runtime.py",
        include_str!("audio8_portable_runtime.py"),
    ))) {
        let target = directory.join(name);
        std::fs::create_dir_all(target.parent().unwrap())?;
        let mut file = tempfile::NamedTempFile::new_in(target.parent().unwrap())?;
        file.write_all(source.as_bytes())?;
        file.persist(target)
            .map_err(|err| err.error)
            .context("Could not install built-in Audio8 backend")?;
    }
    Ok(directory.join(entry))
}
