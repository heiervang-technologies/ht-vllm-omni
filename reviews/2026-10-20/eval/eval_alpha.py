import io, json, sys
import numpy as np, requests, soundfile as sf, librosa, torch
from safetensors.torch import load_file
URL = "http://127.0.0.1:18099/v1/audio/speech"
V = "/home/me/.local/share/tts/voices/qwen3tts0.6b/{}.0.6b.safetensors"
emb = lambda n: next(iter(load_file(V.format(n)).values())).reshape(-1).float()
A, B = emb("dark_cyber_lord"), emb("mei")
print("norms", A.norm().item(), B.norm().item(), (B - A).norm().item())
TEXT = "The quick brown fox jumps over the lazy dog, and then it runs all the way back home again before the sun goes down."
def thirds(wav, sr):
    out = []
    for part in np.array_split(wav, 3):
        f0, _, _ = librosa.pyin(part, fmin=60, fmax=500, sr=sr); f0 = f0[~np.isnan(f0)]
        out.append(round(float(np.median(f0)) if f0.size else 0, 1))
    return out
for alpha in [float(a) for a in sys.argv[1:]]:
    end = (A + alpha * (B - A)).tolist()
    r = requests.post(URL, json=dict(input=TEXT, task_type="Base", response_format="wav", seed=1,
                                     speaker_embedding=A.tolist(), speaker_embedding_end=end, max_new_tokens=400), timeout=300)
    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    sf.write(f"alpha_{alpha}.wav", wav, sr)
    print(json.dumps(dict(alpha=alpha, dur=round(len(wav)/sr, 2), f0_thirds=thirds(wav, sr))))
