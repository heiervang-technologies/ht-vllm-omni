"""End-to-end test for the entropy guardrail sentinel trailer.

Requires a running vllm-omni server with Qwen3-TTS and entropy guardrail support.
Set VLLM_BASE_URL env var (default: http://localhost:30185).

Usage:
    python tests/e2e/test_entropy_guardrail_e2e.py [--base-url URL]
"""

import argparse
import base64
import json
import os
import struct
import sys
from pathlib import Path

import requests

SENTINEL_MAGIC = b"\x00" * 8 + b"VLLMMETA"
DEFAULT_BASE_URL = "http://localhost:30185"
VOICE_PATH = Path.home() / ".local/share/tts/voices/hai.wav"


def load_ref_audio_base64(path: Path) -> str:
    """Load audio file as base64 data URI."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def parse_sentinel_trailer(data: bytes) -> dict | None:
    """Parse sentinel trailer from end of PCM stream.

    Returns the JSON metadata dict if found, None otherwise.
    """
    idx = data.rfind(SENTINEL_MAGIC)
    if idx == -1:
        return None
    offset = idx + len(SENTINEL_MAGIC)
    if len(data) < offset + 4:
        return None
    json_len = struct.unpack("<I", data[offset : offset + 4])[0]
    json_start = offset + 4
    if len(data) < json_start + json_len:
        return None
    payload = data[json_start : json_start + json_len]
    return json.loads(payload.decode("utf-8"))


def test_normal_completion(base_url: str, ref_audio: str) -> bool:
    """Test that normal (non-guardrail) completion has NO sentinel."""
    print("\n=== Test 1: Normal completion (no guardrail trigger) ===")
    resp = requests.post(
        f"{base_url}/v1/audio/speech",
        json={
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "input": "Hello world.",
            "task_type": "Base",
            "ref_audio": ref_audio,
            "x_vector_only_mode": True,
            "response_format": "pcm",
            "stream": True,
            "entropy_guardrail": True,
            "entropy_threshold_high": 8.0,  # very permissive — should NOT trigger
            "entropy_threshold_low": 0.01,
            "entropy_window": 50,
            "max_new_tokens": 200,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    # Check for X-TTS-Trailer header
    has_trailer_header = resp.headers.get("X-TTS-Trailer") == "supported"
    print(f"  X-TTS-Trailer header: {has_trailer_header}")

    data = b""
    for chunk in resp.iter_content(chunk_size=4096):
        data += chunk
    print(f"  Received {len(data)} bytes of audio")

    meta = parse_sentinel_trailer(data)
    if meta is None:
        print("  PASS: No sentinel trailer (normal completion)")
        return True
    else:
        print(f"  FAIL: Unexpected sentinel trailer: {meta}")
        return False


def test_guardrail_trigger_streaming(base_url: str, ref_audio: str) -> bool:
    """Test that tight guardrail thresholds trigger and produce sentinel."""
    print("\n=== Test 2: Streaming with tight guardrail (should trigger) ===")
    resp = requests.post(
        f"{base_url}/v1/audio/speech",
        json={
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "input": "This is a longer sentence to give the model time to potentially trigger the entropy guardrail with very tight thresholds set for testing purposes.",
            "task_type": "Base",
            "ref_audio": ref_audio,
            "x_vector_only_mode": True,
            "response_format": "pcm",
            "stream": True,
            "entropy_guardrail": True,
            "entropy_threshold_high": 2.0,  # very tight — should trigger quickly
            "entropy_threshold_low": 2.0,  # impossible range — guarantees trigger
            "entropy_window": 2,
            "max_new_tokens": 2000,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    has_trailer_header = resp.headers.get("X-TTS-Trailer") == "supported"
    print(f"  X-TTS-Trailer header: {has_trailer_header}")

    data = b""
    for chunk in resp.iter_content(chunk_size=4096):
        data += chunk
    print(f"  Received {len(data)} bytes")

    meta = parse_sentinel_trailer(data)
    if meta is not None:
        print("  PASS: Sentinel trailer found!")
        print(f"    reason: {meta.get('reason')}")
        print(f"    step: {meta.get('step')}")
        print(f"    entropy: {meta.get('entropy')}")
        print(f"    threshold: {meta.get('threshold')}")
        print(f"    window: {meta.get('window')}")
        # Strip trailer and check audio is still valid PCM
        idx = data.rfind(SENTINEL_MAGIC)
        audio_only = data[:idx]
        print(f"    Audio bytes (without trailer): {len(audio_only)}")
        return True
    else:
        print("  FAIL: No sentinel trailer found (guardrail may not have triggered)")
        print("  Note: this could mean metadata didn't survive the pipeline")
        return False


def test_guardrail_nonstreaming(base_url: str, ref_audio: str) -> bool:
    """Test that non-streaming response gets X-TTS-Guardrail header."""
    print("\n=== Test 3: Non-streaming with tight guardrail (header check) ===")
    resp = requests.post(
        f"{base_url}/v1/audio/speech",
        json={
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "input": "Testing the non-streaming guardrail header response.",
            "task_type": "Base",
            "ref_audio": ref_audio,
            "x_vector_only_mode": True,
            "response_format": "wav",
            "stream": False,
            "entropy_guardrail": True,
            "entropy_threshold_high": 2.0,
            "entropy_threshold_low": 2.0,
            "entropy_window": 2,
            "max_new_tokens": 2000,
        },
        timeout=120,
    )
    resp.raise_for_status()

    guardrail_header = resp.headers.get("X-TTS-Guardrail")
    print(f"  X-TTS-Guardrail header: {guardrail_header}")
    print(f"  Received {len(resp.content)} bytes of audio")

    if guardrail_header:
        meta = json.loads(guardrail_header)
        print("  PASS: Guardrail header found!")
        print(f"    reason: {meta.get('reason')}")
        print(f"    step: {meta.get('step')}")
        return True
    else:
        print("  FAIL: No X-TTS-Guardrail header")
        return False


def test_no_guardrail_no_trailer(base_url: str, ref_audio: str) -> bool:
    """Test that with guardrail disabled, no sentinel or header appears."""
    print("\n=== Test 4: Guardrail disabled (no trailer or header) ===")
    resp = requests.post(
        f"{base_url}/v1/audio/speech",
        json={
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "input": "Short test.",
            "task_type": "Base",
            "ref_audio": ref_audio,
            "x_vector_only_mode": True,
            "response_format": "pcm",
            "stream": True,
            "entropy_guardrail": False,
            "max_new_tokens": 200,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    has_trailer_header = resp.headers.get("X-TTS-Trailer")
    data = b""
    for chunk in resp.iter_content(chunk_size=4096):
        data += chunk

    meta = parse_sentinel_trailer(data)
    if meta is None and not has_trailer_header:
        print("  PASS: No trailer or header (guardrail disabled)")
        return True
    else:
        print(f"  FAIL: Unexpected trailer={meta}, header={has_trailer_header}")
        return False


def main():
    parser = argparse.ArgumentParser(description="E2E test for entropy guardrail sentinel")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--voice", default=str(VOICE_PATH))
    args = parser.parse_args()

    voice_path = Path(args.voice)
    if not voice_path.exists():
        print(f"Voice file not found: {voice_path}")
        sys.exit(1)

    ref_audio = load_ref_audio_base64(voice_path)
    print(f"Using server: {args.base_url}")
    print(f"Using voice: {voice_path}")

    # Check server is up
    try:
        r = requests.get(f"{args.base_url}/v1/models", timeout=5)
        r.raise_for_status()
        models = r.json()
        print(f"Server models: {[m['id'] for m in models.get('data', [])]}")
    except Exception as e:
        print(f"Server not reachable at {args.base_url}: {e}")
        sys.exit(1)

    results = []
    results.append(("Normal completion (no trigger)", test_normal_completion(args.base_url, ref_audio)))
    results.append(("Streaming guardrail trigger", test_guardrail_trigger_streaming(args.base_url, ref_audio)))
    results.append(("Non-streaming guardrail header", test_guardrail_nonstreaming(args.base_url, ref_audio)))
    results.append(("Guardrail disabled", test_no_guardrail_no_trailer(args.base_url, ref_audio)))

    print("\n" + "=" * 60)
    print("RESULTS:")
    passed = 0
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(results)} tests passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
