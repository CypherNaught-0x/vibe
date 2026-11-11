use chrono::Local;
use eyre::{bail, ContextCompat, Result};
use rand::distr::Alphanumeric;
use rand::Rng;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use which::which;

pub fn get_local_time() -> String {
    let now = Local::now();
    now.format("%Y-%m-%d %H-%M-%S").to_string()
}

pub fn random_string(length: usize) -> String {
    rand::rng().sample_iter(&Alphanumeric).take(length).map(char::from).collect()
}

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(not(windows))]
const EXECUTABLE_NAME: &str = "ffmpeg";

#[cfg(windows)]
const EXECUTABLE_NAME: &str = "ffmpeg.exe";

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

pub fn get_vibe_temp_folder() -> PathBuf {
    use chrono::Local;
    let current_datetime = Local::now();
    let formatted_datetime = current_datetime.format("%Y-%m-%d").to_string();
    let dir = std::env::temp_dir().join(format!("vibe_temp_{}", formatted_datetime));
    if std::fs::create_dir_all(&dir).is_ok() {
        return dir;
    }
    std::env::temp_dir()
}

pub fn find_ffmpeg_path() -> Option<PathBuf> {
    if let Ok(path) = which(EXECUTABLE_NAME) {
        return Some(path);
    }

    let cwd = std::env::current_dir().ok()?;
    let ffmpeg_in_cwd = cwd.join(EXECUTABLE_NAME);
    if ffmpeg_in_cwd.is_file() && ffmpeg_in_cwd.exists() {
        return Some(ffmpeg_in_cwd);
    }

    if let Ok(exe_path) = std::env::current_exe() {
        let exe_folder = exe_path.parent()?;
        let ffmpeg_in_exe_folder = exe_folder.join(EXECUTABLE_NAME);
        if ffmpeg_in_exe_folder.exists() {
            return Some(ffmpeg_in_exe_folder);
        }
        #[cfg(target_os = "macos")]
        {
            let resources_folder = exe_folder.join("../Resources");
            let ffmpeg_in_resources = resources_folder.join(EXECUTABLE_NAME);
            if ffmpeg_in_resources.exists() {
                return Some(ffmpeg_in_resources);
            }
        }
    }

    None
}

fn command() -> Result<Command> {
    let ffmpeg_path = find_ffmpeg_path().context("ffmpeg not found")?;
    let mut cmd = Command::new(ffmpeg_path);
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    Ok(cmd)
}

/// Mono float PCM at `rate` arrives on stdin and is encoded into `output` as it comes.
/// `mixed`: several sources were summed upstream, so cap the peaks like the old merge did.
pub fn spawn_live_encoder(rate: u32, mixed: bool, output_args: &[&str], output: &Path) -> Result<Child> {
    let mut cmd = command()?;
    let rate = rate.to_string();
    cmd.args(["-f", "f32le", "-ar", rate.as_str(), "-ac", "1", "-i", "pipe:0"]);
    if mixed {
        cmd.args(["-af", MIX_LIMITER]);
    }
    cmd.args(output_args).arg(output);
    cmd.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
    tracing::debug!("live encoder: {:?}", cmd);
    Ok(cmd.spawn()?)
}

/// Peak cap after summing sources; `amix` with `normalize=0` adds them, which can clip.
const MIX_LIMITER: &str = "alimiter=limit=0.97";

/// One pass from the raw captures to the stored file: mix (when there are two), downmix,
/// resample and encode. Replaces the old merge-to-WAV + normalize pair, which wrote and
/// re-read a full intermediate WAV.
pub fn transcode_recording(inputs: &[PathBuf], output_args: &[&str], output: &Path) -> Result<()> {
    if inputs.is_empty() {
        bail!("no recording inputs");
    }
    let mut cmd = command()?;
    for input in inputs {
        cmd.arg("-i").arg(input);
    }
    let graph;
    if inputs.len() > 1 {
        // amix averages by default, which costs 6 dB on a two-input mix. Sum instead and cap
        // the peaks, since the mic and the meeting can both be loud at once.
        graph = format!(
            "amix=inputs={}:duration=longest:dropout_transition=0:normalize=0,{MIX_LIMITER}",
            inputs.len()
        );
        cmd.args(["-filter_complex", graph.as_str()]);
    }
    cmd.args(output_args).arg(output);
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
    tracing::debug!("transcode recording: {:?}", cmd);

    let result = cmd.output()?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        bail!("ffmpeg failed: {}", stderr.chars().take(1000).collect::<String>().trim());
    }
    if std::fs::metadata(output).map(|m| m.len()).unwrap_or(0) == 0 {
        bail!("ffmpeg produced no output");
    }
    Ok(())
}
