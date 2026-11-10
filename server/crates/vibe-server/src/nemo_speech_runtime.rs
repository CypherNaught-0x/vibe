//! Files the Nemotron 3.5 Transcribe engine needs besides the decoder the user picked:
//! NVIDIA's NeMo-Speech.cpp release for this platform and the speech encoder the decoder names.
//! Both are pinned by version and SHA-256 and fetched on first use, next to the decoder.
use anyhow::{bail, Context as _};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

pub const RUNTIME_VERSION: &str = "0.3.0";
const RUNTIME_RELEASES: &str = "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download";
const ENCODER_REPO: &str = "AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF";
/// The repository commit the encoder was checked against (2026-11-02).
const ENCODER_REVISION: &str = "dbd06c5cad7a2f33c8e06be2fb3ef814dbd9aa59";
const SPEECH_MODEL_KEY: &str = "nemo_speech.speech_model";

/// Encoders whose download is pinned; any other one must already sit next to the decoder.
const KNOWN_ENCODERS: [PinnedFile; 1] = [PinnedFile {
    name: "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf",
    sha256: "4221029c8a565c4a80c507ffa760c716e595acf663f9f4273ce69910050e7ba2",
    size: 1_670_410_080,
}];

struct PinnedFile {
    name: &'static str,
    sha256: &'static str,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimeArchive {
    id: &'static str,
    sha256: &'static str,
    size: u64,
    zip: bool,
}

const RUNTIMES: [RuntimeArchive; 7] = [
    RuntimeArchive {
        id: "macos-aarch64-metal",
        sha256: "2b972ed8b5dba07492212acf8cb75cbf34ecb141198ffa1997febe84b7c4d9ed",
        size: 7_239_335,
        zip: false,
    },
    RuntimeArchive {
        id: "macos-x86_64-cpu",
        sha256: "97e98aa5eb82dffa4f513f26336b5301d792c9fb43f720ac8f9520a0e92ffe3f",
        size: 7_742_012,
        zip: false,
    },
    RuntimeArchive {
        id: "linux-x86_64-cuda",
        sha256: "eca00a1f67d3ca200ddba11dd71c4f3d67e044cc0939d17d073978e40b995917",
        size: 703_984_313,
        zip: false,
    },
    RuntimeArchive {
        id: "linux-x86_64-cpu",
        sha256: "255f12b722ab15b2a38a51cf125d56e3fa97058b528f7e895b39d6a473deb06e",
        size: 9_605_101,
        zip: false,
    },
    RuntimeArchive {
        id: "linux-aarch64-cpu",
        sha256: "56858eaf50d19b05902c043f65d0057352bdc107ae380c41053bfbddfc204bf1",
        size: 9_064_856,
        zip: false,
    },
    RuntimeArchive {
        id: "windows-x86_64-cuda",
        sha256: "d0d32ae1bf6ee0534767a47b2aa0e9e2ff6fe39643465c0baae9301d04f6ed42",
        size: 508_625_181,
        zip: true,
    },
    RuntimeArchive {
        id: "windows-x86_64-cpu",
        sha256: "4ed9740e361e4653f56c3dce7d1e81883a1c8b831c1bd7d01567fff1130b5b05",
        size: 8_694_509,
        zip: true,
    },
];

impl RuntimeArchive {
    fn name(&self) -> String {
        format!("nemo-speech-{RUNTIME_VERSION}-{}", self.id)
    }

    fn file_name(&self) -> String {
        format!("{}.{}", self.name(), if self.zip { "zip" } else { "tar.gz" })
    }
}

/// Reports download progress (0-100) and asks whether to stop.
pub struct Feedback<'a> {
    pub progress: &'a mut dyn FnMut(i32),
    pub aborted: &'a mut dyn FnMut() -> bool,
}

/// The encoder file a speech-LLM decoder names, or `None` for any other GGUF.
pub fn speech_model_name(path: &Path) -> anyhow::Result<Option<String>> {
    let name = gguf_string(path, SPEECH_MODEL_KEY)?;
    if let Some(name) = &name {
        // The name is joined to the decoder's folder, so it must stay a bare file name.
        if name.is_empty() || Path::new(name).file_name().is_none_or(|file| file != name.as_str()) {
            bail!("decoder names an invalid speech encoder '{name}'");
        }
    }
    Ok(name)
}

/// The decoder's encoder next to it, downloading a pinned one when it is missing.
pub fn ensure_encoder(decoder: &Path, name: &str, feedback: &mut Feedback<'_>) -> anyhow::Result<PathBuf> {
    let folder = decoder.parent().context("decoder has no parent folder")?;
    let path = folder.join(name);
    let known = KNOWN_ENCODERS.iter().find(|file| file.name == name);
    if path.is_file() {
        if let Some(file) = known {
            let size = path.metadata()?.len();
            if size != file.size {
                bail!(
                    "{} is {size} bytes instead of {}; delete it so it downloads again",
                    path.display(),
                    file.size
                );
            }
        }
        return Ok(path);
    }
    let Some(file) = known else {
        bail!("speech encoder {name} is missing: place it next to {}", decoder.display());
    };
    let url = format!("https://huggingface.co/{ENCODER_REPO}/resolve/{ENCODER_REVISION}/{name}?download=1");
    tracing::info!(url, "downloading Nemotron speech encoder");
    download(&url, &path, file.sha256, file.size, feedback)?;
    Ok(path)
}

/// `nemo-speech` from `VIBE_NEMO_SPEECH_BIN`, or the pinned release installed under `folder`.
pub fn ensure_runtime(folder: &Path, feedback: &mut Feedback<'_>) -> anyhow::Result<PathBuf> {
    if let Some(binary) = std::env::var_os("VIBE_NEMO_SPEECH_BIN") {
        let binary = PathBuf::from(binary);
        if !binary.is_file() {
            bail!("VIBE_NEMO_SPEECH_BIN points to {}, which is not a file", binary.display());
        }
        return Ok(binary);
    }
    let runtime = runtime_for(std::env::consts::OS, std::env::consts::ARCH, has_nvidia_gpu()).with_context(|| {
        format!(
            "NeMo-Speech.cpp has no release for {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let root = folder.join("nemo-speech").join(runtime.name());
    if let Some(binary) = find_binary(&root, 3) {
        return Ok(binary);
    }
    std::fs::create_dir_all(root.parent().expect("runtime root has a parent"))?;
    let archive = root.with_file_name(runtime.file_name());
    let url = format!("{RUNTIME_RELEASES}/v{RUNTIME_VERSION}/{}", runtime.file_name());
    tracing::info!(url, "downloading NeMo-Speech.cpp");
    download(&url, &archive, runtime.sha256, runtime.size, feedback)?;
    let staging = root.with_extension("partial");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    // bsdtar (macOS, Windows 10+) and GNU tar both read these archives; Windows' tar also reads zip.
    let output = Command::new("tar")
        .arg("-xf")
        .arg(&archive)
        .arg("-C")
        .arg(&staging)
        .output()
        .context("could not run tar to unpack NeMo-Speech.cpp")?;
    if !output.status.success() {
        let _ = std::fs::remove_dir_all(&staging);
        bail!(
            "unpacking NeMo-Speech.cpp failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let _ = std::fs::remove_dir_all(&root);
    std::fs::rename(&staging, &root)?;
    let _ = std::fs::remove_file(&archive);
    find_binary(&root, 3).context("the NeMo-Speech.cpp release holds no nemo-speech binary")
}

fn runtime_for(os: &str, arch: &str, nvidia: bool) -> Option<RuntimeArchive> {
    let id = match (os, arch, nvidia) {
        ("macos", "aarch64", _) => "macos-aarch64-metal",
        ("macos", "x86_64", _) => "macos-x86_64-cpu",
        ("linux", "x86_64", true) => "linux-x86_64-cuda",
        ("linux", "x86_64", false) => "linux-x86_64-cpu",
        ("linux", "aarch64", _) => "linux-aarch64-cpu",
        ("windows", "x86_64", true) => "windows-x86_64-cuda",
        ("windows", "x86_64", false) => "windows-x86_64-cpu",
        _ => return None,
    };
    RUNTIMES.iter().copied().find(|runtime| runtime.id == id)
}

fn has_nvidia_gpu() -> bool {
    if cfg!(target_os = "macos") {
        return false;
    }
    let mut command = Command::new("nvidia-smi");
    command.arg("-L");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    command
        .output()
        .is_ok_and(|output| output.status.success() && !output.stdout.is_empty())
}

fn binary_name() -> &'static str {
    if cfg!(windows) {
        "nemo-speech.exe"
    } else {
        "nemo-speech"
    }
}

/// The CLI inside an extracted release (`<name>/bin/nemo-speech`).
fn find_binary(root: &Path, depth: usize) -> Option<PathBuf> {
    let mut folders = Vec::new();
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            folders.push(path);
        } else if path.file_name().is_some_and(|name| name == binary_name()) {
            return Some(path);
        }
    }
    folders.sort();
    (depth > 0)
        .then(|| folders.iter().find_map(|folder| find_binary(folder, depth - 1)))
        .flatten()
}

/// Download to `destination` through a `.part` file, keeping it only when size and SHA-256 match.
/// The transfer runs on its own thread: blocking HTTP panics on a tokio worker (the CLI path).
fn download(url: &str, destination: &Path, sha256: &str, size: u64, feedback: &mut Feedback<'_>) -> anyhow::Result<()> {
    let partial = PathBuf::from(format!("{}.part", destination.display()));
    let cancelled = AtomicBool::new(false);
    let (sender, receiver) = mpsc::channel();
    let result = thread::scope(|scope| {
        let worker = scope.spawn(|| fetch(url, &partial, size, &cancelled, &sender));
        let mut reported = -1;
        loop {
            if (feedback.aborted)() {
                cancelled.store(true, Ordering::Relaxed);
            }
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(percent) if percent != reported => {
                    reported = percent;
                    (feedback.progress)(percent);
                }
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if worker.is_finished() {
                break;
            }
        }
        worker
            .join()
            .unwrap_or_else(|_| Err(anyhow::anyhow!("download thread panicked")))
    })
    .and_then(|(written, digest)| {
        if cancelled.load(Ordering::Relaxed) {
            bail!("download of {} cancelled", file_name(destination));
        }
        if written != size || digest != sha256 {
            bail!(
                "{} failed verification ({written} bytes, sha256 {digest})",
                file_name(destination)
            );
        }
        Ok(std::fs::rename(&partial, destination)?)
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result.with_context(|| format!("could not download {url}"))
}

/// Stream `url` into `partial`, returning its length and SHA-256.
fn fetch(
    url: &str,
    partial: &Path,
    size: u64,
    cancelled: &AtomicBool,
    progress: &mpsc::Sender<i32>,
) -> anyhow::Result<(u64, String)> {
    let mut response = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()?
        .get(url)
        .send()?
        .error_for_status()?;
    let mut file = File::create(partial)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut written = 0u64;
    while !cancelled.load(Ordering::Relaxed) {
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        hasher.update(&buffer[..read]);
        written += read as u64;
        let _ = progress.send((written.saturating_mul(100) / size.max(1)).min(100) as i32);
    }
    file.flush()?;
    let digest = hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect();
    Ok((written, digest))
}

fn file_name(path: &Path) -> String {
    path.file_name().unwrap_or_default().to_string_lossy().into_owned()
}

/// One string value from a GGUF header, without loading any tensor.
fn gguf_string(path: &Path, wanted: &str) -> anyhow::Result<Option<String>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let _tensors = read_u64(&mut reader)?;
    let pairs = read_u64(&mut reader)?;
    for _ in 0..pairs {
        let key = read_string(&mut reader)?;
        let kind = read_u32(&mut reader)?;
        if key == wanted {
            if kind != GGUF_STRING {
                bail!("GGUF key {wanted} is not a string");
            }
            return Ok(Some(read_string(&mut reader)?));
        }
        skip_value(&mut reader, kind)?;
    }
    Ok(None)
}

const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;
const MAX_GGUF_STRING: u64 = 64 << 20;

fn scalar_size(kind: u32) -> anyhow::Result<i64> {
    Ok(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => bail!("unknown GGUF value type {kind}"),
    })
}

fn skip_value(reader: &mut BufReader<File>, kind: u32) -> anyhow::Result<()> {
    match kind {
        GGUF_STRING => {
            let length = read_u64(reader)?;
            if length > MAX_GGUF_STRING {
                bail!("corrupt GGUF string length");
            }
            reader.seek_relative(length as i64)?;
        }
        GGUF_ARRAY => {
            let element = read_u32(reader)?;
            let count = read_u64(reader)?;
            if element == GGUF_STRING || element == GGUF_ARRAY {
                for _ in 0..count {
                    skip_value(reader, element)?;
                }
            } else {
                let bytes = scalar_size(element)?
                    .checked_mul(i64::try_from(count)?)
                    .context("corrupt GGUF array length")?;
                reader.seek(SeekFrom::Current(bytes))?;
            }
        }
        _ => reader.seek_relative(scalar_size(kind)?)?,
    }
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> anyhow::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> anyhow::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string(reader: &mut impl Read) -> anyhow::Result<String> {
    let length = read_u64(reader)?;
    if length > MAX_GGUF_STRING {
        bail!("corrupt GGUF string length");
    }
    let mut bytes = vec![0u8; length as usize];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend((value.len() as u64).to_le_bytes());
        out.extend(value.as_bytes());
    }

    fn gguf(pairs: &[(&str, Vec<u8>)]) -> tempfile::NamedTempFile {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend((pairs.len() as u64).to_le_bytes());
        for (key, value) in pairs {
            string(&mut out, key);
            out.extend(value);
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&out).unwrap();
        file
    }

    fn typed_string(value: &str) -> Vec<u8> {
        let mut out = GGUF_STRING.to_le_bytes().to_vec();
        string(&mut out, value);
        out
    }

    #[test]
    fn finds_the_speech_model_after_arrays_and_scalars() {
        let mut tokens = GGUF_ARRAY.to_le_bytes().to_vec();
        tokens.extend(GGUF_STRING.to_le_bytes());
        tokens.extend(2u64.to_le_bytes());
        string(&mut tokens, "<s>");
        string(&mut tokens, "hello");
        let mut widths = GGUF_ARRAY.to_le_bytes().to_vec();
        widths.extend(4u32.to_le_bytes());
        widths.extend(3u64.to_le_bytes());
        widths.extend([0u8; 12]);
        let mut flag = 7u32.to_le_bytes().to_vec();
        flag.push(1);
        let file = gguf(&[
            ("general.architecture", typed_string("nemotron_h_moe")),
            ("tokenizer.ggml.tokens", tokens),
            ("nemotron_h_moe.feed_forward_length", widths),
            ("flag", flag),
            (SPEECH_MODEL_KEY, typed_string("Encoder-F16.gguf")),
        ]);
        assert_eq!(speech_model_name(file.path()).unwrap().as_deref(), Some("Encoder-F16.gguf"));
    }

    #[test]
    fn other_ggufs_have_no_speech_model() {
        let file = gguf(&[("general.architecture", typed_string("parakeet"))]);
        assert_eq!(speech_model_name(file.path()).unwrap(), None);
    }

    #[test]
    fn rejects_encoder_names_that_leave_the_folder() {
        for name in ["../escape.gguf", "/abs.gguf", ""] {
            let file = gguf(&[(SPEECH_MODEL_KEY, typed_string(name))]);
            assert!(speech_model_name(file.path()).is_err(), "{name}");
        }
    }

    #[test]
    fn picks_cuda_only_with_an_nvidia_gpu() {
        assert_eq!(runtime_for("macos", "aarch64", false).unwrap().id, "macos-aarch64-metal");
        assert_eq!(runtime_for("linux", "x86_64", true).unwrap().id, "linux-x86_64-cuda");
        assert_eq!(runtime_for("windows", "x86_64", false).unwrap().id, "windows-x86_64-cpu");
        assert!(runtime_for("windows", "aarch64", false).is_none());
    }
}
