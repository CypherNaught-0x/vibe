# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "audio8-asr-infinite @ git+https://github.com/Edge0-AI/Audio8-ASR-Infinite@c8ba8eea829be0339e8d7757f8ca52dac06e1e32",
#   "torch==2.13.0", "transformers==5.13.0", "numpy==2.5.3", "safetensors==0.8.0",
#   "huggingface-hub==1.33.0", "tokenizers==0.22.2",
# ]
# ///
"""Generate independent pinned Torch reference, then EXIT before MLX loads.

Default: small deterministic numerical fixtures (<10MB, no checkpoint load).
--full: actual checkpoint/short.wav token reference; REQUIRES parent's heavy slot.
Torch reference and MLX run in separate UV environments because mlx-lm 0.31.3
imports fail with Transformers 5.13's changed AutoTokenizer.register signature.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio8_mlx.common import CHECKPOINT, REVISION, PACKAGE_REVISION, WORKSPACE, workspace_output


def tiny(output):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import VoxtralRealtimeFeatureExtractor
    from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
    from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import Audio8ASRInfiniteForConditionalGeneration
    original = json.loads((WORKSPACE / "plans/audio8/artifacts/hf/hub/models--Edge0--Audio8-ASR-Infinite/blobs/ebde73c2762b0e7bce8d6bb6b121048534c29458").read_text())
    original.update(hidden_size=16, projection_size=128, semantic_vad_horizons_seconds=None, vocab_size=32,
                    bos_token_id=1, eos_token_id=2, pad_token_id=0)
    original["audio_config"].update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                    num_attention_heads=2, num_key_value_heads=2, head_dim=8, sliding_window=11)
    original["text_config"].update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
                                   layer_types=["full_attention"] * 2, pad_token_id=0,
                                   bos_token_id=1, eos_token_id=2)
    (output / "tiny-config.json").write_text(json.dumps(original, indent=2))
    extractor = VoxtralRealtimeFeatureExtractor(feature_size=128, n_fft=400, hop_length=160,
                                                sampling_rate=16000, global_log_mel_max=1.5)
    rng = np.random.default_rng(23)
    features = [rng.normal(0, 0.1, (1, 128, count)).astype(np.float32) for count in (32, 16, 16)]
    samples = (np.sin(np.arange(3480) * 0.17) * 0.2).astype(np.float32)
    mel = extractor([samples], sampling_rate=16000, return_tensors="pt")["input_features"].numpy()
    for label, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        torch.manual_seed(7)
        model = Audio8ASRInfiniteForConditionalGeneration(Audio8ASRInfiniteConfig(**original)).eval().to(dtype=dtype)
        # Independent small reference deliberately spans a sliding-cache boundary.
        arrays = {"samples": samples, "mel": mel}
        with torch.inference_mode():
            condition = model.build_t_cond(3, batch_size=1, device=torch.device("cpu"), dtype=dtype, frame_len=8)
            arrays["t_cond"] = condition.float().numpy()
            audio_cache = text_cache = None
            position = 0
            for index, feature in enumerate(features):
                arrays[f"features_{index}"] = feature
                conv = model.audio_tower.embedder(torch.tensor(feature, dtype=dtype))
                arrays[f"conv_{index}"] = conv.float().numpy()
                hidden, outputs = model.get_audio_tower_hidden_states(
                    encoder_inputs_embeds=conv, past_key_values=audio_cache, use_cache=True,
                    position_ids=torch.arange(position, position + conv.shape[1])[None], return_outputs=True)
                position += conv.shape[1]
                audio_cache = outputs.past_key_values
                arrays[f"hidden_{index}"] = hidden.float().numpy()
                grouped = model.group_audio_hidden_states(hidden, frame_len=8)
                audio = model.multi_modal_projector(grouped)
                arrays[f"audio_{index}"] = audio.float().numpy()
                ids = torch.arange(audio.shape[1])[None] + 3 + index
                arrays[f"ids_{index}"] = ids.numpy()
                embeds = model.build_text_inputs_embeds(input_ids=ids, audio_embeds=audio)
                text_output = model.forward_language_model_with_delay(inputs_embeds=embeds, past_key_values=text_cache,
                                                                      use_cache=True, t_cond=condition, logits_to_keep=1)
                text_cache = text_output.past_key_values
                arrays[f"logits_{index}"] = text_output.logits.float().numpy()
            for frame_len in (4, 6, 8):
                source = torch.arange(1 * frame_len * 2 * 16, dtype=dtype).reshape(1, frame_len * 2, 16) / 100
                arrays[f"group_input_{frame_len}"] = source.float().numpy()
                arrays[f"group_project_{frame_len}"] = model.multi_modal_projector(
                    model.group_audio_hidden_states(source, frame_len=frame_len)).float().numpy()
        weights = {key: value.contiguous() for key, value in model.state_dict().items() if key != "language_model.lm_head.weight"}
        save_file(weights, str(output / f"tiny-{label}.safetensors"))
        np.savez(output / f"tiny-{label}-reference.npz", **arrays)
    print(json.dumps({"reference": "pinned Torch tiny layers", "output": str(output)}))


def full(output, hf_cache):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoFeatureExtractor
    from audio8_asr_infinite.modeling.configuration_audio8_asr_infinite import Audio8ASRInfiniteConfig
    from audio8_asr_infinite.modeling.modeling_audio8_asr_infinite import (
        Audio8ASRInfiniteForConditionalGeneration, resolve_qwen_streaming_special_token_ids,
        resolve_qwen_language_token_id)
    from audio8_asr_infinite.streaming_inference import simulated_streaming_greedy_decode_batch
    common = dict(revision=REVISION, cache_dir=str(hf_cache), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, **common)
    extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT, **common)
    config = Audio8ASRInfiniteConfig.from_pretrained(CHECKPOINT, **common)
    config.semantic_vad_horizons_seconds = None
    print(json.dumps({"stage": "torch-reference-load", "pid": os.getpid()}), flush=True)
    model = Audio8ASRInfiniteForConditionalGeneration.from_pretrained(
        CHECKPOINT, **common, config=config, torch_dtype=torch.float16,
        use_safetensors=True, attn_implementation="sdpa").to("mps").eval()

    class AudioConfig:
        sampling_rate = 16000
        raw_audio_samples_per_token = 2560
        streaming_n_left_pad_tokens = 9

    class CpuFrontend:
        def __call__(self, windows, **kwargs):
            if hasattr(windows, "detach"):
                windows = windows.detach().cpu().numpy()
            return extractor(windows, **kwargs)

    fixture = WORKSPACE / "server/fixtures/short.wav"
    with wave.open(str(fixture), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("Full parity fixture must be mono 16kHz PCM16")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    numeric = {}
    audio_count = text_count = 0

    def capture_audio(_module, _args, kwargs, outputs):
        nonlocal audio_count
        if audio_count < 3:
            numeric[f"conv_{audio_count}"] = kwargs["inputs_embeds"].detach().float().cpu().numpy()
            numeric[f"hidden_{audio_count}"] = outputs.last_hidden_state.detach().float().cpu().numpy()
        audio_count += 1

    def capture_logits(_module, _args, logits):
        nonlocal text_count
        if text_count < 3:
            numeric[f"logits_{text_count}"] = logits.detach().float().cpu().numpy()
        text_count += 1

    audio_hook = model.audio_tower.register_forward_hook(capture_audio, with_kwargs=True)
    text_hook = model.language_model.lm_head.register_forward_hook(capture_logits)
    print(json.dumps({"stage": "torch-reference-decode", "pid": os.getpid(), "audio_seconds": len(samples) / 16000}), flush=True)
    started = time.perf_counter()
    result = simulated_streaming_greedy_decode_batch(
        model=model, tokenizer=tokenizer, feature_extractor=CpuFrontend(), waveforms=[samples],
        language_token_ids=[resolve_qwen_language_token_id(tokenizer, "en")],
        special_ids=resolve_qwen_streaming_special_token_ids(tokenizer), audio_config=AudioConfig(),
        num_delay_tokens=3, frame_len=8, right_pad_text_tokens=10, dtype=torch.float16,
        device=torch.device("mps"), max_new_tokens=None)[0]
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    audio_hook.remove()
    text_hook.remove()
    np.savez(output / "full-numerical-reference.npz", **numeric)
    (output / "full-reference.json").write_text(json.dumps({"checkpoint": CHECKPOINT, "revision": REVISION,
        "package_revision": PACKAGE_REVISION, "fixture": "server/fixtures/short.wav", "device": "mps", "dtype": "float16",
        "inference_seconds": elapsed, **result}, indent=2))
    print(json.dumps({"reference": "pinned pretrained FP16", "tokens": len(result["generated_token_ids"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--hf-cache", type=Path, default=WORKSPACE / "plans/audio8/artifacts/hf/hub")
    args = parser.parse_args()
    output = workspace_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.full:
        full(output, args.hf_cache)
    else:
        tiny(output)


if __name__ == "__main__":
    main()
