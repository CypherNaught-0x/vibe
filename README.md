<p align="center">
  <a target="blank" href="https://github.com/thewh1teagle/vibe">
    <img
        width="96px"
        alt="vibe++ logo"
        src="./design/logo.png"
    />
  </a>
</p>

<h1 align="center">vibe++ - Transcribe on your own!</h1>

<p align="center">
  <strong>⌨️ Transcribe audio and video right on your device</strong>
  <br/>
</p>

<p align="center">
  <a target="_blank" href="https://thewh1teagle.github.io/vibe/">
    🔗 Download upstream Vibe
  </a>
    &nbsp; | &nbsp; Give it a Star ⭐ | &nbsp;
    <a target="_blank" href="https://thewh1teagle.github.io/vibe/?action=support-vibe">Support the project 🤝</a>
</p>

<hr />

## About vibe++

vibe++ is a fork of [thewh1teagle/vibe](https://github.com/thewh1teagle/vibe), kept rebased on the latest upstream `main`. It adds:

- 🔴 **Live transcription (Audio8)**: captions appear while you speak. Microphone audio streams at 16 kHz to a local Audio8 server, which shows draft and final captions as they settle. Backends: Torch, ONNX and MLX. MLX loads weights as 8-bit, which on an M3 Pro is fast enough for real time. Audio8 only supports English and Chinese.
- 🧠 **RAM-only recording ("Do not store audio")**: recordings stay in memory and never touch disk. They are transcribed from RAM and discarded afterwards. URL downloads are disabled in this mode.
- 🗑️ **Delete the original transcript after AI processing**: keep only the AI summary. New recordings, live or file-based, are summarized automatically when this option and the summary task are both on.
- 🦙 **Nemotron 3.5 Transcribe 30B-A4B**: NVIDIA's speech-LLM via [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp). It works with any decoder quantization, covers 40 languages plus auto-detect, and keeps audio in memory. The runtime and encoder are downloaded on first use and checked by SHA-256.
- 📁 **Folders for transcripts**: group projects into named folders in the sidebar and move them between folders. Pick a folder with its **+** button and new transcriptions, recordings and live captions are saved there until you clear it. Folders are ordinary subfolders of the projects folder, so they also show up in your file manager.
- 🛠️ **Fixes**: stopping a recording twice (button, hotkey or shortcut) no longer crashes the app. On macOS, you can move the window by dragging the top bar, including the empty parts of the transcript toolbar.

Builds of vibe++ are not published. To build it yourself, see [docs/BUILDING.md](docs/BUILDING.md).

## Screenshots

<p align="center">
	<a target="_blank" href="https://thewh1teagle.github.io/vibe/">
    	<img width=720 src="./design/app-preview-light.png">
	</a>
</p>

# Features 🌟

- 🌍 Transcribe almost every language
- 🔒 Ultimate privacy: fully offline transcription, no data ever leaves your device
- 🎨 User friendly design
- 🎙️ Transcribe audio / video
- 🎶 Option to transcribe audio from popular websites (YouTube, Vimeo, Facebook, Twitter and more!)
- 📂 Batch transcribe multiple files!
- 📝 Support `SRT`, `VTT`, `TXT`, `HTML`, `PDF`, `JSON`, `DOCX` formats
- 👀 Realtime preview
- 🤖 Supports Whisper, Nemotron 3.5, and Parakeet TDT v3 models
- ✨ Summarize transcripts: Get quick, multilingual summaries using the Claude API
- 🧠 Ollama support: Do local AI analysis and batch summaries with Ollama
- 🌐 Translate to English from any language
- 🖨️ Print transcript directly to any printer
- 🔄 Automatic updates
- 💻 Optimized for `GPU` (`macOS`, `Windows`, `Linux`)
- 🎮 Optimized for `Nvidia` / `AMD` / `Intel` GPUs! (`Vulkan`/`CoreML`)
- 🔧 Total Freedom: Customize Models Easily via Settings
- ⚙️ Model arguments for advanced users
- ⏳ Transcribe system audio
- 🎤 Transcribe from microphone
- 🖥️ CLI support: Use Vibe directly from the command line interface! (see `--help`)
- 👥 Speaker diarization
- 🎬 Stable timestamps mode for subtitle/movie-grade timing (VAD-backed, slower)
- 📱 Record from your phone: scan a QR code once, then your computer does the transcribing and sends the text straight back — works on iOS and Android with nothing to install from an app store
- 📥 Integrate custom models from your own site: Use `vibe://download/?url=<model url>`
- 📹 Choose caption length optimized for videos / reels
- ⚡ HTTP API with Swagger docs and agent skills

# Supported platforms 🖥️

`MacOS`
`Windows`
`Linux`

# Contribute 🤝

## Developers

Get started with local development [here](docs/BUILDING.md).

Ideas, bug reports and translation fixes are welcome — please [open an issue](https://github.com/thewh1teagle/vibe/issues).
Pull requests are limited to a few collaborators. See [CONTRIBUTE.md](CONTRIBUTE.md).

# Community

[![Discord](https://img.shields.io/badge/chat-discord-7289da.svg)](https://discord.gg/EcxWSstQN8)

# Roadmap 🛣️

You can see the roadmap in [Vibe-Roadmap](https://github.com/users/thewh1teagle/projects/5/views/1)

# Add translation 🌐

See the [translation guide](docs/TRANSLATIONS.md).

# Docs 📄

see [Vibe Docs](https://github.com/thewh1teagle/vibe/tree/main/docs)

# I want to know more!

Medium [post](https://medium.com/@thewh1teagle/creating-vibe-multilingual-audio-transcription-872ab6d9dbb0)

# Issue report

You can open [new issue](https://github.com/thewh1teagle/vibe/issues/new?assignees=octocat&labels=bug&projects=&template=bug_report.yaml&title=[Short+title]) and it's recommend to check [DEBUG.md](docs/DEBUG.md) first.

# Privacy Policy 🔒

Your privacy is important to us. Please review our [Privacy Policy](http://thewh1teagle.github.io/vibe/?action=open-privacy-policy) to understand how we handle your data.

# Credits

Thanks for [tauri.app](https://tauri.app/) for making the best apps framework I ever seen

Thanks for [github.com/whisper.cpp](https://github.com/ggerganov/whisper.cpp) for outstanding interface for the AI model.

Thanks for [openai.com](https://openai.com/) for their amazing [Whisper model](https://openai.com/research/whisper)

Thanks for [github.com](https://github.com/) for their support in open source projects, providing infastructure completely free.

And for all the amazing open source frameworks and libraries which this project uses...
