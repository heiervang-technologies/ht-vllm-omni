# TTS steering: findings (2026-10-20)

Task: make our TTS steerable (emotion + voice). Three questions:

1. Can `feat/speaker-steering` (dynamic `speaker_embedding_end`) ship?
2. Do `instructions` work on the Base model we run in prod?
3. What should speaker-service / `say` expose?

Audio from every run below is on Centurion KC3:
`/run/media/me/storage/builds/ht-vllm-omni/tts-steering/eval/*.wav`.
Scripts and raw numbers are in `eval/` next to this file.

## 1. Dynamic speaker steering: does not work as designed

`origin/feat/speaker-steering` is seven months behind `ht`. All of it except
the final commit (56627d29) is already on `ht` through upstream, so the PR
ports only that commit (`feat/speaker-steering-rebase`). During the port I
changed the following:

- It accepts any start speaker the talker resolves, including uploaded
  voices like `markus` that are stored as audio.
- The ramp horizon defaults to about 4 frames per text token (measured on
  1.7B-Base). The old default of `max_new_tokens`=2048 moved a 5-second
  utterance only about 3% of the way.
- The delta goes into the talker-only `text_step` term. That keeps it out of
  the code predictor's layer-0 input.

Mechanism: the start speaker sits in one prompt position that is already in
the KV cache. Steering therefore adds `ramp * (end - start)` to **every**
decode step's input.

End-to-end test: 0.6B-Base, local server on the worktree, start
`dark_cyber_lord`, end `mei`. Embeddings come from
`~/.local/share/tts/voices/qwen3tts0.6b`. Median F0 per third of the
utterance:

| case | duration | F0 by thirds (Hz) | ASR |
|---|---|---|---|
| dark_cyber_lord only | 5.4–6.2 s | ~110 / 113 / 93 | exact |
| mei only | 6.2–6.4 s | ~262 / 231 / 212 | exact |
| full steer, default horizon | **163 s** (hit max tokens) | 418 / 398 / 289 | text, then minutes of squeak |
| full steer, 48 frames | **163 s** | ~400 / 380 / 345 | degraded / empty |

Scaled steering: end = start + α·(end − start), so the delta is α times the
full voice difference:

| α | duration | F0 thirds |
|---|---|---|
| 0.02–0.35 | 5.5–7.2 s | ~95–150, no consistent drift toward mei |
| 0.5 | 6.1 s | 180 / 136 / 109 (noisy) |
| 0.7 | 31.8 s | 96 / **428 / 428** (runaway, no EOS) |

**Conclusion:** no α gives a usable transition. Small α does nothing
measurable. At larger α the model stops emitting EOS and produces a high
squeal. A per-step additive bias does not stand in for a different speaker
token. The original branch used the same mechanism and, in addition, fed the
delta into the code predictor, so it was not working either. **Do not merge
the PR as-is.** It stays a draft so the protocol plumbing and tests can be
reused.

**What does work today, with no vllm change:** a *static* interpolated
`speaker_embedding` per request. The `slerp_*`, `*_centroid_*` and `interp_*`
files in `~/.local/share/tts/voices/qwen3tts*/` are examples. For a voice
change across an utterance, the speaker-service could split the text into
sentences and synthesize each one with `slerp(start, end, i/n)`. That uses
only the existing API.

## 2. `instructions` on Base vs CustomVoice

Qwen's model card lists instruction control only for 1.7B-CustomVoice and
1.7B-VoiceDesign. Base (both sizes) and 0.6B-CustomVoice are not listed. In
the code, vllm-omni prepends the instruct prompt for every task type, Base
included, so Base *sees* it, but the model was not trained on it.

Live prod (1.7B-Base, voice `markus`, same sentence, seeds 1–3) with ASR
through qwen3-asr:

| instructions | mean duration | mean RMS | median F0 | ASR exact |
|---|---|---|---|---|
| none | 6.69 s | −15.2 dB | 101 Hz | 3/3 |
| "furious, angry, almost shouting" | 5.12 s | −19.1 dB | 118 Hz | 3/3 |
| "very sad, quiet, tearful, slowly" | 5.81 s | −18.1 dB | 110 Hz | 3/3 |
| "extremely excited, happy, fast" | 4.67 s | −18.8 dB | 141 Hz | 3/3 |
| "whisper very softly" | 5.36 s | −18.7 dB | 119 Hz | 3/3 |

Reading:

- **Any** instruction makes Base about 20–30% faster and about 3–4 dB quieter,
  whatever its content. "Angry, shouting" comes out *quieter* than no
  instruction. That shift is a side effect of an out-of-distribution prefix,
  not style control.
- There is a weak directional signal: "excited" is the fastest and
  highest-pitched, and "sad" is the slowest of the instructed runs. Whisper
  does not whisper.
- Intelligibility is unaffected (15/15 exact transcripts).
- Verdict: on Base, `instructions` are a mild, unreliable prosody nudge, not
  emotion control. Treat them as experimental and have Markus listen
  (`live_base_markus_*.wav`) before exposing them as an "emotion" feature.
- CustomVoice was not measured live: prod serves Base only, and the shared
  3090 had about 4 GB free, which is not enough for a 1.7B second instance
  next to prod. Per the model card, 1.7B-CustomVoice is the model with real
  instruction control, but it is limited to its 9 built-in timbres and cannot
  clone `markus`/`hai`/etc. Real emotion **and** our own voices therefore
  needs one of:
  1. a second 1.7B-CustomVoice deployment for "emotional" lines in built-in
     timbres (GPU budget and big-dog sign-off required), or
  2. per-emotion reference clips for our voices (record or generate an angry,
     sad, … `ref_audio` per voice) on Base. Base reproduces the reference
     prosody well, so this is the most promising route for cloned voices, or
  3. a later evaluation of VoiceDesign → clone pipelines.

## 3. Proposed plumbing (PRs, nothing deployed)

- speaker-service: `/v1/speak` accepts `instructions`, `speaker_embedding`,
  `speaker_embedding_end`, `speaker_steering_frames` and passes them through.
  `speaker_embedding` replaces ref_audio.
- `say`: `--instruct TEXT` and `--emotion NAME` (preset instruction strings).
  These are labelled experimental because of the Base result above.

Rollout of any of this touches prod TTS on Centurion and needs Markus's OK
plus coordination with big-dog.
