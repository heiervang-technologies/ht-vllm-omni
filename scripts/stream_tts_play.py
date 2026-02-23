#!/usr/bin/env python3
"""Stream TTS audio from vLLM-omni and play it in real-time.

Usage:
    # Base model with reference audio (x-vector only, no ref_text needed):
    python scripts/stream_tts_play.py "Hello, this is a streaming test."
    python scripts/stream_tts_play.py --ref-audio path/to/ref.wav "Some text."

    # Save output:
    python scripts/stream_tts_play.py --save output.wav "Save and play."
    python scripts/stream_tts_play.py --no-play --save output.wav "Just save."

    # Compare streaming vs non-streaming latency:
    python scripts/stream_tts_play.py --no-stream "Non-streaming comparison."
"""

import argparse
import base64
import io
import struct
import subprocess
import sys
import time

import numpy as np
import requests

SAMPLE_RATE = 24000
CHANNELS = 1
BIT_DEPTH = 16
BYTES_PER_SAMPLE = BIT_DEPTH // 8


def make_wav_header(data_size: int) -> bytes:
    """Create a WAV header for the given PCM data size."""
    header = io.BytesIO()
    header.write(b"RIFF")
    header.write(struct.pack("<I", 36 + data_size))
    header.write(b"WAVE")
    header.write(b"fmt ")
    header.write(struct.pack("<I", 16))  # chunk size
    header.write(struct.pack("<H", 1))  # PCM format
    header.write(struct.pack("<H", CHANNELS))
    header.write(struct.pack("<I", SAMPLE_RATE))
    header.write(struct.pack("<I", SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE))
    header.write(struct.pack("<H", CHANNELS * BYTES_PER_SAMPLE))
    header.write(struct.pack("<H", BIT_DEPTH))
    header.write(b"data")
    header.write(struct.pack("<I", data_size))
    return header.getvalue()


def make_ref_audio_data_url(path: str | None) -> str:
    """Load a reference audio file and return a base64 data URL.

    If path is None, generates a 1-second silent WAV as a placeholder
    (the model uses the x-vector from this for speaker embedding).
    """
    if path is not None:
        with open(path, "rb") as f:
            audio_bytes = f.read()
    else:
        # Generate 1s silent WAV at 16kHz (speaker encoder rate)
        sr = 16000
        samples = np.zeros(sr, dtype=np.int16)
        buf = io.BytesIO()
        buf.write(make_wav_header(len(samples) * 2))
        buf.write(samples.tobytes())
        audio_bytes = buf.getvalue()

    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def stream_and_play(
    text: str,
    base_url: str,
    model: str,
    ref_audio_path: str | None,
    play: bool,
    save_path: str | None,
    stream: bool,
):
    ref_audio_url = make_ref_audio_data_url(ref_audio_path)

    payload = {
        "input": text,
        "model": model,
        "task_type": "Base",
        "response_format": "pcm",
        "stream": stream,
        "ref_audio": ref_audio_url,
        "x_vector_only_mode": True,
    }

    print(f"[request] POST {base_url}/v1/audio/speech")
    print(f"[request] stream={stream}, task_type=Base, x_vector_only=True")
    print(f"[request] ref_audio: {'<silent placeholder>' if ref_audio_path is None else ref_audio_path}")
    print(f"[request] text: {text[:80]}{'...' if len(text) > 80 else ''}")

    t_start = time.monotonic()

    # Start audio player subprocess (reads raw s16le PCM from stdin)
    player = None
    if play:
        player = subprocess.Popen(
            [
                "sox",
                "-t",
                "raw",
                "-r",
                str(SAMPLE_RATE),
                "-e",
                "signed-integer",
                "-b",
                str(BIT_DEPTH),
                "-c",
                str(CHANNELS),
                "-",
                "-d",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    all_pcm = bytearray()
    chunk_count = 0
    first_chunk_time = None

    try:
        resp = requests.post(
            f"{base_url}/v1/audio/speech",
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        t_response = time.monotonic()
        print(f"[timing] HTTP response headers: {(t_response - t_start) * 1000:.0f}ms")

        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue

            chunk_count += 1
            now = time.monotonic()

            if first_chunk_time is None:
                first_chunk_time = now
                latency = (now - t_start) * 1000
                print(f"[timing] first audio chunk: {latency:.0f}ms ({len(chunk)} bytes)")

            all_pcm.extend(chunk)

            if player and player.stdin:
                try:
                    player.stdin.write(chunk)
                    player.stdin.flush()
                except BrokenPipeError:
                    break

            # Progress every 10 chunks
            if chunk_count % 10 == 0:
                elapsed = (now - t_start) * 1000
                duration_ms = (len(all_pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)) * 1000
                print(
                    f"[progress] {chunk_count} chunks, "
                    f"{len(all_pcm)} bytes, "
                    f"{duration_ms:.0f}ms audio, "
                    f"{elapsed:.0f}ms elapsed"
                )

    except requests.exceptions.ConnectionError as e:
        print(f"[error] Connection failed: {e}", file=sys.stderr)
        print(f"[error] Is the server running at {base_url}?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[error] HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        if player and player.stdin:
            player.stdin.close()

    t_end = time.monotonic()
    total_bytes = len(all_pcm)
    audio_duration = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) if total_bytes > 0 else 0

    print("\n[summary]")
    print(f"  chunks received: {chunk_count}")
    print(f"  total PCM bytes: {total_bytes}")
    print(f"  audio duration:  {audio_duration:.2f}s")
    print(f"  total wall time: {(t_end - t_start) * 1000:.0f}ms")
    if first_chunk_time:
        print(f"  time to first audio: {(first_chunk_time - t_start) * 1000:.0f}ms")
        if audio_duration > 0:
            print(f"  realtime factor: {(t_end - t_start) / audio_duration:.2f}x")

    # Save to file
    if save_path and total_bytes > 0:
        with open(save_path, "wb") as f:
            if save_path.endswith(".wav"):
                f.write(make_wav_header(total_bytes))
            f.write(all_pcm)
        print(f"  saved to: {save_path}")

    # Wait for player to finish
    if player:
        player.wait()


def main():
    parser = argparse.ArgumentParser(description="Stream TTS and play audio in real-time")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Model name")
    parser.add_argument("--ref-audio", metavar="PATH", help="Reference audio WAV for voice cloning (default: silent)")
    parser.add_argument("--save", metavar="PATH", help="Save audio to file (.wav or .pcm)")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio, just measure")
    parser.add_argument("--no-stream", action="store_true", help="Non-streaming mode (for comparison)")
    args = parser.parse_args()

    stream_and_play(
        text=args.text,
        base_url=args.url,
        model=args.model,
        ref_audio_path=args.ref_audio,
        play=not args.no_play,
        save_path=args.save,
        stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
