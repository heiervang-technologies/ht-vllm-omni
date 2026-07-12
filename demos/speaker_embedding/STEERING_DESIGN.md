# Dynamic Speaker Embedding Steering — Design

## Goal
Gradually transition from one speaker embedding to another during a single
generation, producing a single audio output that morphs voice mid-speech.

## Architecture Constraints
- Speaker embedding occupies **1 position** in the prompt (`codec_input`).
- During prefill, this position is processed through all transformer layers
  and its key/value representations are cached in the KV cache.
- During decode, the KV cache is read-only — vllm's attention backends don't
  expose a "rewrite this position" API.

## Approach: KV Cache Splice per Decode Step

At each decode step:

1. Compute `t = step / total_steps` (0→1 over generation).
2. SLERP between `emb_start` and `emb_end` to get `emb_t`.
3. Build a **1-token "mini prefill"** using `emb_t`:
   - `mini_input = tts_pad_embed + emb_t` (same as original codec_prefix construction)
4. Forward this single token through the model's layers to get K/V projections
   at each layer.
5. **Overwrite** the KV cache at the speaker embedding's position index with
   these new K/V values.

### KV Cache Access
vllm stores KV cache as contiguous tensors per layer. The position index of the
speaker embedding within the sequence is deterministic (computed during prefill).
We store this index as `speaker_embed_cache_pos` in `info_dict`.

The actual overwrite requires:
- Access to `self.model.layers[i].self_attn.attn.kv_cache` (or equivalent)
- Knowledge of the block table mapping for this request
- Writing new K/V values at the correct slot

This is the part that requires understanding vllm's paged attention internals.

### Simpler Alternative: Cross-Attention Style Injection
Instead of modifying KV cache, add a **cross-attention-like bias** at each layer:

- During prefill, compute and store K/V projections of the speaker embed per layer.
- During decode, compute K/V projections of the interpolated embed per layer.
- Compute the **delta** (new - original) K/V projections.
- Hook into each layer's attention to add this delta to the cached K/V before
  the attention computation.

This requires model hooks but doesn't touch vllm's cache management.

### Simplest Prototype: Embedding-Space Bias
Add the embedding-space delta directly to the decode step's input_embeds:

```python
# In preprocess(), decode path:
if steering_active:
    t = decode_step / total_steps
    emb_t = slerp(emb_start, emb_end, t)
    delta = project(emb_t) - project(emb_start)  # in model hidden dim
    inputs_embeds_out = inputs_embeds_out + alpha * delta
```

This is an approximation — the delta propagates through the model but doesn't
retroactively change the cached K/V from the original speaker position. However,
it may produce audible steering effects since the model attends to both the
cached speaker position AND the current input.

## Implementation Plan

### Phase 1: API + Protocol (no GPU needed)
- [ ] Add `speaker_embedding_end` to `OpenAICreateSpeechRequest`
- [ ] Validation: requires `speaker_embedding` to also be set
- [ ] Pass both through `voice_clone_prompt` as `ref_spk_embedding_start/end`
- [ ] Tests for protocol validation

### Phase 2: Talker Steering Logic (needs GPU for integration test)
- [ ] In `_build_prompt_embeds`: detect start/end, store both + position index
- [ ] In `preprocess` decode path: compute interpolated embedding, apply bias
- [ ] Track decode step count in info_dict

### Phase 3: KV Cache Splice (needs deep vllm integration)
- [ ] Hook into attention layers to modify K/V at speaker position
- [ ] Benchmark perf impact of per-step splice
