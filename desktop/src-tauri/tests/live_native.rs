//! Compiles the live modules independently of binary command registration. No audio device
//! is opened: hardware/sidecar boundaries are stubs, while CPAL types, HTTP and resampling
//! use the real dependencies and production implementations.
#![allow(dead_code)]

#[path = "../src/error.rs"]
mod error;
mod config {
    pub const STORE_FILENAME: &str = "store.bin";
}
mod ffmpeg {
    pub fn get_local_time() -> String {
        "live-test".into()
    }
}
mod meeting_prompt {
    pub fn recording_started(_: &tauri::AppHandle) {}
    pub fn recording_stopped(_: &tauri::AppHandle) {}
}
mod cmd {
    pub mod files {
        pub fn sanitize_filename_stem(value: &str) -> String {
            value.into()
        }
    }
    pub mod server_cmd {
        pub async fn load_model(_: tauri::AppHandle, _: String, _: Option<i32>, _: u32, _: Option<bool>) -> crate::error::CommandResult<String> {
            Err(eyre::eyre!("The unit harness must not load a physical Audio8 model").into())
        }
    }
    pub mod audio {
        use cpal::traits::DeviceTrait;
        use cpal::{FromSample, Sample};
        #[derive(serde::Serialize, serde::Deserialize)]
        pub struct AudioDevice {
            pub is_default: bool,
            pub is_input: bool,
            pub id: String,
            pub name: String,
        }
        pub(crate) struct StreamHandle(pub(crate) cpal::Stream);
        unsafe impl Send for StreamHandle {}
        unsafe impl Sync for StreamHandle {}
        pub(crate) struct LevelMeter;
        impl LevelMeter {
            pub(crate) fn new(_: tauri::AppHandle) -> Self {
                Self
            }
            pub(crate) fn push(&self, _: f32) {}
        }
        pub(crate) fn buffer_peak<T: Sample>(input: &[T]) -> f32
        where
            f32: FromSample<T>,
        {
            input.iter().map(|v| f32::from_sample(*v).abs()).fold(0.0, f32::max)
        }
        pub(crate) fn get_output_device_and_config(
            _: &cpal::Host,
            _: &AudioDevice,
        ) -> eyre::Result<(cpal::Device, cpal::SupportedStreamConfig)> {
            eyre::bail!("The unit harness must not open a loopback device")
        }
        // Keep trait imports/type checking aligned with production CPAL configuration.
        fn _input_config(device: &cpal::Device) -> eyre::Result<cpal::SupportedStreamConfig> {
            Ok(device.default_input_config()?)
        }
    }
}
mod setup {
    pub struct ServerState {
        pub process: Option<ServerProcess>,
    }
    pub struct ServerProcess;
    impl ServerProcess {
        pub fn client(&self) -> reqwest::Client {
            reqwest::Client::new()
        }
        pub fn base_url(&self) -> String {
            "http://127.0.0.1:1".into()
        }
    }
}
#[path = "../src/cmd/live.rs"]
mod live;
#[path = "../src/live_capture.rs"]
mod live_capture;
#[path = "../src/private_audio.rs"]
mod private_audio;
#[path = "../src/transcript.rs"]
mod transcript;
