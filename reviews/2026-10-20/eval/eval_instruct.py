"""Measure whether `instructions` changes Qwen3-TTS output. Writes WAVs + a JSON table."""
import io, json, sys, time
import numpy as np, requests, soundfile as sf, librosa

URL = sys.argv[1]
TAG = sys.argv[2]
EXTRA = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
ASR = "http://localhost:30189/v1/audio/transcriptions"
TEXT = "I just found out what happened at the meeting today, and honestly, I can not believe it."
STYLES = {
    "none": None,
    "angry": "Speak in a furious, angry voice, almost shouting.",
    "sad": "Speak in a very sad, quiet, tearful voice, slowly.",
    "excited": "Speak in an extremely excited, happy, fast and energetic voice.",
    "whisper": "Whisper very softly.",
}
SEEDS = [1, 2, 3]

def feats(wav, sr):
    f0, vf, _ = librosa.pyin(wav, fmin=60, fmax=500, sr=sr)
    f0 = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=wav)[0]
    return dict(dur=len(wav) / sr, rms_db=float(20 * np.log10(rms.mean() + 1e-9)),
                f0_mean=float(f0.mean()) if f0.size else 0.0, f0_std=float(f0.std()) if f0.size else 0.0)

def asr(b):
    try:
        r = requests.post(ASR, files={"file": ("a.wav", b, "audio/wav")}, data={"model": "Qwen/Qwen3-ASR-1.7B"}, timeout=60)
        return r.json().get("text", r.text)[:200]
    except Exception as e:
        return f"asr-error {e}"

rows = []
for style, instr in STYLES.items():
    for seed in SEEDS:
        body = dict(input=TEXT, response_format="wav", seed=seed, **EXTRA)
        if instr:
            body["instructions"] = instr
        t = time.time()
        r = requests.post(URL, json=body, timeout=180)
        if r.status_code != 200:
            rows.append(dict(style=style, seed=seed, error=r.text[:300])); print(rows[-1]); continue
        wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        sf.write(f"{TAG}_{style}_{seed}.wav", wav, sr)
        row = dict(style=style, seed=seed, latency=round(time.time() - t, 2), **feats(wav, sr), asr=asr(r.content))
        rows.append(row); print(json.dumps(row))
json.dump(rows, open(f"{TAG}.json", "w"), indent=1)
