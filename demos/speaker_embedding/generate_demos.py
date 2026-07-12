"""Generate speaker embedding demo audio files.

Produces:
  1. Voice transition: SLERP from female to male at 11 steps (0.0 to 1.0)
  2. Emotion steering: Same voice with different emotion instructions

Requires a running Qwen3-TTS Base server on localhost:8091.
"""

import base64
import json
import os
import sys

import httpx
import numpy as np

API_BASE = os.environ.get("API_BASE", "http://localhost:8091")
TEXT = "The future of voice synthesis is here. Every shade of expression, every timbre, captured in a single embedding."

# Reference audio paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
GRADIO_AUDIO = os.path.join(
    REPO_ROOT,
    ".venv/lib/python3.12/site-packages/gradio/media_assets/audio",
)
FEMALE_AUDIO = os.path.join(GRADIO_AUDIO, "cate_blanch.mp3")
MALE_AUDIO = os.path.join(GRADIO_AUDIO, "heath_ledger.mp3")


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two vectors."""
    v0_norm = v0 / (np.linalg.norm(v0) + 1e-10)
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    if abs(dot) > 0.9995:
        # Nearly parallel, fall back to lerp
        return (1 - t) * v0 + t * v1
    omega = np.arccos(dot)
    return (np.sin((1 - t) * omega) / np.sin(omega)) * v0 + (
        np.sin(t * omega) / np.sin(omega)
    ) * v1


def extract_embedding(audio_path: str) -> list[float]:
    """Extract speaker embedding by sending ref_audio to the speech API
    with x_vector_only_mode, then reading the embedding from the response.

    Since the API doesn't have a dedicated extraction endpoint, we use the
    upload endpoint and retrieve the cached embedding.
    """
    print(f"  Extracting embedding from {os.path.basename(audio_path)}...")

    # Upload as a named voice, which triggers ECAPA-TDNN extraction
    voice_name = f"_tmp_extract_{os.path.basename(audio_path).replace('.', '_')}"

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Upload to voices endpoint
    resp = httpx.post(
        f"{API_BASE}/v1/audio/voices",
        files={"audio_sample": (os.path.basename(audio_path), audio_data)},
        data={"consent": "agree", "name": voice_name},
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"  Upload failed: {resp.text}")
        # Fall back: generate a short clip with ref_audio to trigger extraction
        # and use a synthetic embedding
        raise RuntimeError(f"Upload failed: {resp.status_code} {resp.text}")

    print(f"  Uploaded voice '{voice_name}', now generating with it to extract embedding...")

    # Generate a short speech with this voice to get the embedding cached
    # We'll use x_vector_only_mode with the uploaded voice
    speech_resp = httpx.post(
        f"{API_BASE}/v1/audio/speech",
        json={
            "input": "test",
            "voice": voice_name,
            "task_type": "Base",
            "x_vector_only_mode": True,
            "response_format": "wav",
        },
        timeout=120,
    )

    if speech_resp.status_code != 200:
        print(f"  Speech gen failed ({speech_resp.status_code}), trying ref_audio approach...")
        # Direct ref_audio approach
        audio_b64 = base64.b64encode(audio_data).decode()
        speech_resp = httpx.post(
            f"{API_BASE}/v1/audio/speech",
            json={
                "input": "test",
                "task_type": "Base",
                "ref_audio": audio_b64,
                "ref_text": "test reference",
                "x_vector_only_mode": True,
                "response_format": "wav",
            },
            timeout=120,
        )

    # For now, use the offline extraction approach
    return _extract_offline(audio_path)


_encoder_cache = None


def _get_encoder():
    """Load and cache the ECAPA-TDNN speaker encoder."""
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    sys.path.insert(0, REPO_ROOT)

    import torch
    from transformers import AutoConfig

    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSSpeakerEncoder,
    )

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)

    model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
    encoder.eval()

    # Load weights from the single safetensors file
    model_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
    )
    # Find the snapshot
    import glob

    safetensor_files = glob.glob(
        os.path.join(model_dir, "snapshots/*/model.safetensors")
    )
    if not safetensor_files:
        from huggingface_hub import hf_hub_download

        safetensor_files = [hf_hub_download(model_path, "model.safetensors")]

    from safetensors.torch import load_file

    state_dict = load_file(safetensor_files[0])
    encoder_state = {
        k.replace("speaker_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("speaker_encoder.")
    }
    encoder.load_state_dict(encoder_state, strict=False)
    print(f"  Loaded {len(encoder_state)} encoder weights")

    _encoder_cache = encoder
    return encoder


def _extract_offline(audio_path: str) -> list[float]:
    """Extract embedding offline using the ECAPA-TDNN model directly."""
    import torch

    encoder = _get_encoder()

    import librosa

    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        mel_spectrogram,
    )

    # Load at 24kHz as the mel spectrogram function expects
    wav, sr = librosa.load(audio_path, sr=24000)
    mels = mel_spectrogram(
        torch.from_numpy(wav).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)

    with torch.no_grad():
        emb = encoder(mels)
    return emb.squeeze().tolist()


def generate_speech(
    text: str,
    speaker_embedding: list[float],
    output_path: str,
    response_format: str = "wav",
) -> bool:
    """Generate speech using speaker_embedding passthrough."""
    print(f"  Generating: {os.path.basename(output_path)}...")
    resp = httpx.post(
        f"{API_BASE}/v1/audio/speech",
        json={
            "input": text,
            "task_type": "Base",
            "speaker_embedding": speaker_embedding,
            "response_format": response_format,
        },
        timeout=180,
    )
    if resp.status_code != 200:
        print(f"  FAILED ({resp.status_code}): {resp.text[:200]}")
        return False
    with open(output_path, "wb") as f:
        f.write(resp.content)
    size_kb = len(resp.content) / 1024
    print(f"  OK ({size_kb:.1f} KB)")
    return True


def demo_voice_transition(
    emb_female: list[float],
    emb_male: list[float],
    output_dir: str,
):
    """Generate voice transition from female to male at 11 SLERP ratios."""
    print("\n=== Voice Transition (Female → Male) ===")
    ratios = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    v0 = np.array(emb_female)
    v1 = np.array(emb_male)

    for ratio in ratios:
        blended = slerp(v0, v1, ratio)
        label = f"transition_{ratio:.1f}"
        path = os.path.join(output_dir, f"{label}.wav")
        generate_speech(TEXT, blended.tolist(), path)


def demo_emotion_steering(
    base_embedding: list[float],
    output_dir: str,
):
    """Generate same voice with different emotion/style instructions.

    Uses the instructions field to steer prosody while keeping
    the same speaker embedding.
    """
    print("\n=== Emotion Steering (Same Voice, Different Emotions) ===")
    emotions = {
        "neutral": "Speak in a calm, neutral tone.",
        "happy": "Speak with joy and enthusiasm, as if sharing wonderful news.",
        "sad": "Speak with a melancholic, sorrowful tone, slowly and softly.",
        "angry": "Speak with intensity and frustration, raising your voice.",
        "whisper": "Speak in a soft, intimate whisper.",
        "excited": "Speak with high energy and excitement, fast-paced.",
    }

    emotion_text = "I can't believe what just happened. This changes everything we thought we knew."

    for name, instruction in emotions.items():
        path = os.path.join(output_dir, f"emotion_{name}.wav")
        print(f"  Generating: emotion_{name}.wav (instruction: {instruction[:40]}...)...")
        resp = httpx.post(
            f"{API_BASE}/v1/audio/speech",
            json={
                "input": emotion_text,
                "task_type": "Base",
                "speaker_embedding": base_embedding,
                "instructions": instruction,
                "response_format": "wav",
            },
            timeout=180,
        )
        if resp.status_code != 200:
            print(f"  FAILED ({resp.status_code}): {resp.text[:200]}")
            continue
        with open(path, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  OK ({size_kb:.1f} KB)")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Check server
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        print(f"Server health: {r.status_code}")
    except Exception as e:
        print(f"Server not reachable at {API_BASE}: {e}")
        sys.exit(1)

    # Step 1: Extract embeddings
    print("\n=== Extracting Speaker Embeddings ===")
    emb_female = _extract_offline(FEMALE_AUDIO)
    print(f"  Female embedding: dim={len(emb_female)}")

    emb_male = _extract_offline(MALE_AUDIO)
    print(f"  Male embedding: dim={len(emb_male)}")

    # Save embeddings
    from safetensors.numpy import save_file as save_safetensors_np

    save_safetensors_np(
        {"speaker_embedding": np.array(emb_female, dtype=np.float32)},
        os.path.join(output_dir, "embedding_female.safetensors"),
    )
    save_safetensors_np(
        {"speaker_embedding": np.array(emb_male, dtype=np.float32)},
        os.path.join(output_dir, "embedding_male.safetensors"),
    )
    print("  Saved embeddings to output/")

    # Step 2: Voice transition
    demo_voice_transition(emb_female, emb_male, output_dir)

    # Step 3: Emotion steering (using female voice as base)
    demo_emotion_steering(emb_female, output_dir)

    # Step 4: Emotion steering with blended voice (50/50)
    print("\n=== Emotion Steering (Blended 50/50 Voice) ===")
    blended_50 = slerp(np.array(emb_female), np.array(emb_male), 0.5).tolist()
    blended_dir = os.path.join(output_dir, "blended_emotion")
    os.makedirs(blended_dir, exist_ok=True)
    demo_emotion_steering(blended_50, blended_dir)

    print(f"\n=== Done! Output files in {output_dir} ===")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".wav"):
            size = os.path.getsize(os.path.join(output_dir, f)) / 1024
            print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
