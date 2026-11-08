import io, json, sys
import numpy as np, requests, soundfile as sf, librosa
from safetensors.torch import load_file
URL = "http://127.0.0.1:18099/v1/audio/speech"
ASR = "http://localhost:30189/v1/audio/transcriptions"
V = "/home/me/.local/share/tts/voices/qwen3tts0.6b/{}.0.6b.safetensors"
emb = lambda n: next(iter(load_file(V.format(n)).values())).reshape(-1).float().tolist()
A, B = sys.argv[1], sys.argv[2]
TEXT = "The quick brown fox jumps over the lazy dog, and then it runs all the way back home again before the sun goes down."
def seg_feats(wav, sr):
    out = []
    for part in np.array_split(wav, 3):
        f0, _, _ = librosa.pyin(part, fmin=60, fmax=500, sr=sr); f0 = f0[~np.isnan(f0)]
        cent = librosa.feature.spectral_centroid(y=part, sr=sr)[0].mean()
        out.append((round(float(np.median(f0)) if f0.size else 0, 1), round(float(cent))))
    return out
cases = {
    f"{A}_only": dict(speaker_embedding=emb(A)),
    f"{B}_only": dict(speaker_embedding=emb(B)),
    f"{A}_to_{B}_default": dict(speaker_embedding=emb(A), speaker_embedding_end=emb(B)),
    f"{A}_to_{B}_48f": dict(speaker_embedding=emb(A), speaker_embedding_end=emb(B), speaker_steering_frames=48),
}
for name, extra in cases.items():
    for seed in (1, 2):
        r = requests.post(URL, json=dict(input=TEXT, task_type="Base", response_format="wav", seed=seed, **extra), timeout=300)
        if r.status_code != 200: print(name, seed, "ERR", r.text[:300]); continue
        wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        sf.write(f"steer_{name}_{seed}.wav", wav, sr)
        t = requests.post(ASR, files={"file": ("a.wav", r.content, "audio/wav")}, data={"model": "Qwen/Qwen3-ASR-1.7B"}, timeout=60).json().get("text", "")
        print(json.dumps(dict(case=name, seed=seed, dur=round(len(wav)/sr, 2), thirds_f0_centroid=seg_feats(wav, sr), asr=t[:120])))
