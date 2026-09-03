# Adversarial review — `gems/amber/asr-request-path`

Reviewed head: `8287f7af`  
Base: `db0e4590`  
Scope: ASR ingest latency/hardening, Pascal guard, and bench kit

## CLAIMS

- **“Stereo request-path CPU time improves 26.6% / 23.9%” — UNPROVEN under the raw-data rule.** The deterministic corpus hashes match between baseline and optimized reports, `perf_counter_ns` is the clock, and the paired benchmark makes the large downmix win plausible. However, the committed JSON stores only p50/p95 summaries, not the five individual timing samples. The generated WAV bytes are not committed, the report omits the exact command/source SHA/repeat count, and the timed path reimplements stages rather than invoking the installed HTTP/production loader end to end. The exact percentages therefore lack committed raw observations.
- **“Fixed audio set” — PARTLY CONFIRMED.** The generator is committed and before/after SHA-256 values match for all three WAVs. The actual files are under an external work directory and not committed. Reproducibility depends on regenerating identical floating-point PCM.
- **“Five hostile cases reach the hardened production ingest and all reject” — REFUTED.** `fuzz_ingest(..., optimized=True)` calls `load_audio_soundfile()` directly, not production `load_audio()`. The PyAV budget test calls `_load_audio_pyav_bounded()` directly with an ordinary WAV. Neither forces libsndfile to raise an allowed fallback code and then proves `load_audio()` reaches PyAV. The new during-frame bound is present but its production branch reach is unproven.
- **“Decoded input is bounded” — CONFIRMED statically for the two helper paths.** SoundFile metadata is checked before read; PyAV checks declared duration and every emitted frame. A production-dispatch regression test is still required.
- **“Fast stereo downmix is bit-identical” — CONFIRMED for the tested two-channel float32 case.** The implementation and focused test compare with `mean(axis=0, dtype=float32)` using `array_equal`. No equivalent proof exists for more than two channels.
- **“Pascal startup guard fails loud” — PARTLY CONFIRMED.** It rejects CUDA newer than 12.6, missing sm_61 kernels, bf16, and a known bf16-checkpoint-to-fp16 cast. It does not inspect or reject a selected FP8 quantization method, despite FP8 being forbidden on sm_61.
- **“14 focused tests pass” — UNPROVEN by this reviewer.** The worker forwarded that result, but the documented `.venv-asr` currently lacks `torch`, so reviewer execution aborted during pytest plugin import. No dependency installation was attempted on the serving gem.
- **GPU latency/WER/power claims — correctly recorded as could-not-measure.** The bench procedure is present; no GPU number is presented as measured.
- **“The 2,626,388-byte wheel has a conservative 1,754,285-byte (66.79%) ASR code cut” — PARTLY CONFIRMED.** The referenced wheel exists outside Git, hashes to `7527594c0b37774beff5598a67ef8ad0e9a93732c0e5d3fda9b5e15b013bdb8e`, and an independent ZIP-member recount reproduces 672 files, 9,515,300 uncompressed bytes, and the stated compressed-member sum and percentage. The *deployable saving* is UNPROVEN: no slim wheel/image was built, its on-disk size was not measured, and no import or ASR serving test was run against it. In fact, core engine and serving modules import `vllm_omni.diffusion` unconditionally, so deleting the counted tree without a dependency-closure refactor breaks imports.
- **Charter disk ratchet — ABSENT.** This cycle records no `df -h /`, `~/work` size, venv sizes, model/cache sizes, or before/after deployed artifact/container size. A ZIP member estimate is not a substitute for the required disk inventory.

## LOST-BODIES

- Present: production monkeypatch installation, SoundFile size/truncation/empty guards, PyAV per-frame bound, stereo optimization, six ASR tests, Pascal guard/tests, deterministic corpus generator, serving WER/CER/TTFT/power harness, pinned dataset preparer, and committed summary reports.
- Absent: a test that reaches PyAV through `load_audio()`; a test that the installed monkeypatch is actually the function used by STT; raw per-repeat timings; committed corpus bytes or a source-SHA/repeat manifest; an FP8 startup rejection; reviewer-reproducible test environment; measured GPU/accuracy/power results; a built/import-tested slim artifact; and the charter-required disk inventory.

## PASCAL

**compatible-with-fixes.** The branch respects no-GPU-on-serving-gem discipline and makes fp32/INT8 the documented ASR plan, but the generic startup guard must explicitly fail on FP8 selection.

## NEW-FAILURE-MODES

- The loader monkeypatch runs at API-server import time and globally changes every STT request in that process.
- Valid audio above 512 MiB now fails; this is intentional and loud.
- `mono=False` PyAV fallback does not resample when a different `sr` is requested but returns the target rate label; uncommon callers can receive mislabeled audio.
- Summary-only benchmark output makes later audit or resampling/statistical checks impossible.
- A fallback helper can remain green while production dispatch never reaches it.
- Treating a ZIP member sum as realized disk savings can green-light an artifact that does not import because retained engine/serving modules still depend on removed diffusion modules.

## VERDICT

**mergeable-with-fixes.** Before merge: route fuzz/tests through `load_audio()` and force the permitted libsndfile-error-to-PyAV transition; test monkeypatch installation; commit individual timing samples plus command/source/repeat metadata; add FP8 rejection to the Pascal guard; either correct or explicitly reject `mono=False` resampling in the fallback; build and smoke-test the proposed slim artifact before claiming realized savings; and record `df` plus work/venv/model/cache sizes.

**UPSTREAM: needs-split.** The generic audio-ingest bounds, production-path tests, and stereo fast path are upstream candidates. The sm_61/CUDA guard, Pascal benchmark policy, HT monkeypatch integration, and fork-specific artifact plan must remain separate.
