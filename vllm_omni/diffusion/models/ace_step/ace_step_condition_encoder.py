# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported from ACE-Step 1.5 condition encoder for vllm-omni integration.
# These are standard PyTorch modules (no vllm-specific layers needed).
"""
ACE-Step condition encoder components for vllm-omni.

Ports the following from ACE-Step 1.5
(acestep/models/base/modeling_acestep_v15_base.py):

- AceStepEncoderLayer: bidirectional transformer encoder block
- AceStepLyricEncoder: 8-layer bidirectional transformer over lyric embeddings
- AceStepTimbreEncoder: 4-layer bidirectional transformer over reference audio
- AttentionPooler: attention-based pooling with CLS token
- AceStepAudioTokenizer: acoustic features -> attention pool -> ResidualFSQ
- AudioTokenDetokenizer: discrete tokens -> expanded continuous features
- ACEStepConditionEncoder: fuses text, lyrics, timbre into packed sequence
- pack_sequences(): concatenate + sort by mask for sequence packing
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from vector_quantize_pytorch import ResidualFSQ
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AceStepConditionConfig:
    """Configuration for ACE-Step condition encoder components.

    Default values match the ACE-Step 1.5 pretrained checkpoint.
    """

    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    use_sliding_window: bool = True
    sliding_window: int = 128

    # Text encoder
    text_hidden_dim: int = 1024

    # Lyric encoder
    num_lyric_encoder_hidden_layers: int = 8

    # Timbre encoder
    timbre_hidden_dim: int = 64
    num_timbre_encoder_hidden_layers: int = 4

    # Audio tokenizer / detokenizer
    audio_acoustic_hidden_dim: int = 64
    pool_window_size: int = 5
    num_attention_pooler_hidden_layers: int = 2
    fsq_dim: int = 2048
    fsq_input_levels: list[int] = field(
        default_factory=lambda: [8, 8, 8, 5, 5, 5]
    )
    fsq_input_num_quantizers: int = 1

    def layer_types(self, num_layers: int) -> list[str]:
        """Alternating layer types: sliding, full, sliding, full, ..."""
        return [
            "sliding_attention" if (i + 1) % 2 else "full_attention"
            for i in range(num_layers)
        ]


# ---------------------------------------------------------------------------
# Utility: RMSNorm (matches Qwen3RMSNorm)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Utility: Rotary Position Embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1_000_000.0,
    ):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1,
        ).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # [B, 1, L, D]
    sin = sin.unsqueeze(1)
    return (
        (q * cos) + (_rotate_half(q) * sin),
        (k * cos) + (_rotate_half(k) * sin),
    )


# ---------------------------------------------------------------------------
# Utility: SwiGLU MLP (matches Qwen3MLP)
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Utility: 4D attention mask
# ---------------------------------------------------------------------------

def _create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = False,
) -> torch.Tensor:
    """Create 4D additive attention mask (0.0 = keep, -inf = mask)."""
    indices = torch.arange(seq_len, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    valid = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    if is_causal:
        valid = valid & (diff >= 0)
    if is_sliding_window and sliding_window is not None:
        if is_causal:
            valid = valid & (diff <= sliding_window)
        else:
            valid = valid & (torch.abs(diff) <= sliding_window)
    valid = valid.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        pad_4d = attention_mask.view(
            attention_mask.shape[0], 1, 1, seq_len,
        ).to(torch.bool)
        valid = valid & pad_4d
    out = torch.full(valid.shape, torch.finfo(dtype).min, dtype=dtype, device=device)
    out.masked_fill_(valid, 0.0)
    return out


# ---------------------------------------------------------------------------
# Utility: pack_sequences
# ---------------------------------------------------------------------------

def pack_sequences(
    hidden1: torch.Tensor, hidden2: torch.Tensor,
    mask1: torch.Tensor, mask2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate two sequences and sort so valid tokens come first.

    Args:
        hidden1: [B, L1, D], hidden2: [B, L2, D]
        mask1:   [B, L1],    mask2:   [B, L2]

    Returns:
        (packed_hidden [B, L1+L2, D], new_mask [B, L1+L2])
    """
    hidden_cat = torch.cat([hidden1, hidden2], dim=1)
    mask_cat = torch.cat([mask1, mask2], dim=1)
    B, L, D = hidden_cat.shape
    sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)
    packed = torch.gather(
        hidden_cat, 1, sort_idx.unsqueeze(-1).expand(B, L, D),
    )
    lengths = mask_cat.sum(dim=1)
    new_mask = (
        torch.arange(L, dtype=torch.long, device=hidden_cat.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    return packed, new_mask


# ---------------------------------------------------------------------------
# Encoder Attention (bidirectional, self-attention only)
# ---------------------------------------------------------------------------

class _EncoderAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool,
        attention_dropout: float,
        layer_type: str,
        sliding_window: Optional[int],
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.attention_dropout = attention_dropout
        self.layer_type = layer_type
        self.sliding_window = (
            sliding_window if layer_type == "sliding_attention" else None
        )
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias,
        )
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------

class _EncoderLayer(nn.Module):
    """Bidirectional transformer encoder layer (self-attn + MLP)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        attention_dropout: float,
        layer_type: str,
        sliding_window: Optional[int],
    ):
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = _EncoderAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            layer_type=layer_type,
            sliding_window=sliding_window,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings, attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Helpers: build layer stack / masks / run layers
# ---------------------------------------------------------------------------

def _build_layers(
    config: AceStepConditionConfig, num_layers: int,
) -> nn.ModuleList:
    types = config.layer_types(num_layers)
    return nn.ModuleList([
        _EncoderLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            layer_type=types[i],
            sliding_window=config.sliding_window,
        )
        for i in range(num_layers)
    ])


def _build_masks(
    seq_len: int, dtype: torch.dtype, device: torch.device,
    attention_mask: Optional[torch.Tensor],
    config: AceStepConditionConfig,
) -> dict[str, Optional[torch.Tensor]]:
    full = _create_4d_mask(
        seq_len, dtype, device, attention_mask, is_causal=False,
    )
    sliding = None
    if config.use_sliding_window:
        sliding = _create_4d_mask(
            seq_len, dtype, device, attention_mask,
            sliding_window=config.sliding_window,
            is_sliding_window=True, is_causal=False,
        )
    return {"full_attention": full, "sliding_attention": sliding}


def _run_layers(
    layers: nn.ModuleList,
    hidden_states: torch.Tensor,
    pos_emb: tuple[torch.Tensor, torch.Tensor],
    masks: dict[str, Optional[torch.Tensor]],
) -> torch.Tensor:
    for layer in layers:
        hidden_states = layer(hidden_states, pos_emb, masks[layer.layer_type])
    return hidden_states


# ---------------------------------------------------------------------------
# Lyric Encoder
# ---------------------------------------------------------------------------

class _LyricEncoder(nn.Module):
    """8-layer bidirectional transformer over lyric embeddings.

    Input:  [B, L, text_hidden_dim] + [B, L] mask
    Output: [B, L, hidden_size]
    """

    def __init__(self, config: AceStepConditionConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.text_hidden_dim, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.layers = _build_layers(config, config.num_lyric_encoder_hidden_layers)

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed_tokens(inputs_embeds)
        pos_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        pos_emb = self.rotary_emb(h, pos_ids)
        masks = _build_masks(
            h.shape[1], h.dtype, h.device, attention_mask, self.config,
        )
        h = _run_layers(self.layers, h, pos_emb, masks)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Timbre Encoder
# ---------------------------------------------------------------------------

class _TimbreEncoder(nn.Module):
    """4-layer bidirectional transformer over reference audio features.

    Input:  packed [N, T, timbre_hidden_dim] + order mask [N]
    Output: (unpacked [B, max_count, D], mask [B, max_count])
    """

    def __init__(self, config: AceStepConditionConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(
            config.timbre_hidden_dim, config.hidden_size,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.special_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size),
        )
        self.layers = _build_layers(
            config, config.num_timbre_encoder_hidden_layers,
        )

    @staticmethod
    def _unpack(
        packed: torch.Tensor, order_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack [N, D] -> [B, max_count, D] + mask [B, max_count]."""
        N, d = packed.shape
        device, dtype = packed.device, packed.dtype
        B = int(order_mask.max().item() + 1)
        counts = torch.bincount(order_mask, minlength=B)
        mc = counts.max().item()
        si = torch.argsort(
            order_mask * N + torch.arange(N, device=device), stable=True,
        )
        sb = order_mask[si]
        pos = torch.arange(N, device=device)
        bs = torch.cat([
            torch.tensor([0], device=device), torch.cumsum(counts, 0)[:-1],
        ])
        pis = pos - bs[sb]
        inv = torch.empty_like(si)
        inv[si] = torch.arange(N, device=device)
        pib = pis[inv]
        idx = order_mask * mc + pib
        oh = F.one_hot(idx, num_classes=B * mc).to(dtype)
        flat = oh.t() @ packed
        unp = flat.reshape(B, mc, d)
        mf = (oh.sum(0) > 0).long().reshape(B, mc)
        return unp, mf

    def forward(
        self,
        refer_packed: torch.Tensor,
        order_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed_tokens(refer_packed)
        pos_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        pos_emb = self.rotary_emb(h, pos_ids)
        masks = _build_masks(
            h.shape[1], h.dtype, h.device, attention_mask, self.config,
        )
        h = _run_layers(self.layers, h, pos_emb, masks)
        h = self.norm(h)
        # CLS token pooling (first position)
        h = h[:, 0, :]
        return self._unpack(h, order_mask)


# ---------------------------------------------------------------------------
# Attention Pooler (for audio tokenizer)
# ---------------------------------------------------------------------------

class _AttentionPooler(nn.Module):
    """CLS-token attention pooling over patches.

    Input:  [B, T, P, D]
    Output: [B, T, D]
    """

    def __init__(self, config: AceStepConditionConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.special_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * 0.02,
        )
        self.layers = _build_layers(
            config, config.num_attention_pooler_hidden_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, D = x.shape
        x = self.embed_tokens(x)
        st = self.special_token.expand(B, T, 1, -1)
        x = torch.cat([st, x], dim=2)
        x = rearrange(x, "b t p c -> (b t) p c")
        pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = self.rotary_emb(x, pos_ids)
        masks = _build_masks(
            x.shape[1], x.dtype, x.device, None, self.config,
        )
        x = _run_layers(self.layers, x, pos_emb, masks)
        x = self.norm(x)
        cls = x[:, 0, :]
        return rearrange(cls, "(b t) c -> b t c", b=B)


# ---------------------------------------------------------------------------
# Audio Tokenizer
# ---------------------------------------------------------------------------

class _AudioTokenizer(nn.Module):
    """Acoustic features -> attention pool -> ResidualFSQ.

    Input:  [B, T_patch, pool_window_size, D]
    Output: (quantized [B, T_patch, fsq_dim], indices)
    """

    def __init__(self, config: AceStepConditionConfig):
        super().__init__()
        self.pool_window_size = config.pool_window_size
        self.audio_acoustic_proj = nn.Linear(
            config.audio_acoustic_hidden_dim, config.hidden_size,
        )
        self.attention_pooler = _AttentionPooler(config)
        self.quantizer = ResidualFSQ(
            dim=config.fsq_dim,
            levels=config.fsq_input_levels,
            num_quantizers=config.fsq_input_num_quantizers,
        )

    def forward(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.audio_acoustic_proj(hidden_states)
        h = self.attention_pooler(h)
        return self.quantizer(h)

    def tokenize(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(
            x, "n (t_patch p) d -> n t_patch p d", p=self.pool_window_size,
        )
        return self.forward(x)


# ---------------------------------------------------------------------------
# Audio Token Detokenizer
# ---------------------------------------------------------------------------

class _AudioTokenDetokenizer(nn.Module):
    """Expands quantized tokens back to 25 Hz continuous features.

    Input:  [B, T, D]
    Output: [B, T * pool_window_size, audio_acoustic_hidden_dim]
    """

    def __init__(self, config: AceStepConditionConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.special_tokens = nn.Parameter(
            torch.randn(1, config.pool_window_size, config.hidden_size) * 0.02,
        )
        self.layers = _build_layers(
            config, config.num_attention_pooler_hidden_layers,
        )
        self.proj_out = nn.Linear(
            config.hidden_size, config.audio_acoustic_hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = self.embed_tokens(x)
        x = x.unsqueeze(2).repeat(1, 1, self.config.pool_window_size, 1)
        st = self.special_tokens.expand(B, T, -1, -1)
        x = x + st
        x = rearrange(x, "b t p c -> (b t) p c")
        pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = self.rotary_emb(x, pos_ids)
        masks = _build_masks(
            x.shape[1], x.dtype, x.device, None, self.config,
        )
        h = _run_layers(self.layers, x, pos_emb, masks)
        h = self.norm(h)
        h = self.proj_out(h)
        return rearrange(
            h, "(b t) p c -> b (t p) c",
            b=B, p=self.config.pool_window_size,
        )


# ---------------------------------------------------------------------------
# Top-level: ACEStepConditionEncoder
# ---------------------------------------------------------------------------

class ACEStepConditionEncoder(nn.Module):
    """ACE-Step condition encoder for vLLM-Omni.

    Encodes multiple conditioning inputs (text, lyrics, timbre) and packs them
    into a single sequence for cross-attention in the DiT. Also prepares
    context latents (source latents + chunk masks) for the decoder.

    This wraps the following ACE-Step components:
    - AceStepConditionEncoder (text projector + lyric encoder + timbre encoder)
    - AceStepAudioTokenizer + AudioTokenDetokenizer (for cover songs)
    - prepare_condition() logic from AceStepConditionGenerationModel

    Args:
        od_config: OmniDiffusion configuration object.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config
        self.cond_config = AceStepConditionConfig()

        cfg = self.cond_config
        self.text_projector = nn.Linear(
            cfg.text_hidden_dim, cfg.hidden_size, bias=False,
        )
        self.lyric_encoder = _LyricEncoder(cfg)
        self.timbre_encoder = _TimbreEncoder(cfg)
        self.tokenizer = _AudioTokenizer(cfg)
        self.detokenizer = _AudioTokenDetokenizer(cfg)
        self.null_condition_emb = nn.Parameter(
            torch.randn(1, 1, cfg.hidden_size),
        )

    # -- tokenize / detokenize helpers for cover mode --

    def _tokenize(
        self,
        x: torch.Tensor,
        silence_latent: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pool = self.cond_config.pool_window_size
        if x.shape[1] % pool != 0:
            pad_len = pool - (x.shape[1] % pool)
            x = torch.cat(
                [x, silence_latent[:1, :pad_len].repeat(x.shape[0], 1, 1)],
                dim=1,
            )
            attention_mask = F.pad(
                attention_mask, (0, pad_len), mode="constant", value=0,
            )
        x = rearrange(x, "n (t_patch p) d -> n t_patch p d", p=pool)
        seq_len = x.shape[1]
        chunk = math.ceil(attention_mask.shape[1] / seq_len)
        attention_mask = attention_mask.to(x.dtype)
        attention_mask = F.max_pool1d(
            attention_mask.unsqueeze(1),
            kernel_size=chunk, stride=chunk, ceil_mode=True,
        ).squeeze(1)
        quantized, indices = self.tokenizer(x)
        return quantized, indices, attention_mask

    def _detokenize(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.detokenizer(quantized)

    # -- main forward --

    def forward(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_packed: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
        src_latents: torch.Tensor,
        chunk_masks: torch.Tensor,
        attention_mask: torch.Tensor,
        is_covers: torch.Tensor,
        silence_latent: Optional[torch.Tensor] = None,
        precomputed_lm_hints_25Hz: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all conditions and prepare context latents.

        Args:
            text_hidden_states: Text embeddings [B, T_text, text_dim].
            text_attention_mask: Text mask [B, T_text].
            lyric_hidden_states: Lyric embeddings [B, T_lyric, text_dim].
            lyric_attention_mask: Lyric mask [B, T_lyric].
            refer_audio_packed: Packed reference audio [N, T_ref, acoustic_dim].
            refer_audio_order_mask: Batch assignment for packed refs [N].
            src_latents: Source latents [B, T, acoustic_dim].
            chunk_masks: Chunk masks [B, T, acoustic_dim].
            attention_mask: Latent attention mask [B, T].
            is_covers: Cover indicators [B] (0=text2music, 1=cover).
            silence_latent: Silence latent for tokenizer padding.
            precomputed_lm_hints_25Hz: Optional precomputed LM hints.
            audio_codes: Optional precomputed audio codes.

        Returns:
            (encoder_hidden_states [B, S, D],
             encoder_attention_mask [B, S],
             context_latents [B, T, 2*acoustic_dim])
        """
        dtype = src_latents.dtype

        # --- Encode conditions ---
        text_proj = self.text_projector(text_hidden_states)
        lyric_enc = self.lyric_encoder(lyric_hidden_states, lyric_attention_mask)
        timbre_unp, timbre_mask = self.timbre_encoder(
            refer_audio_packed, refer_audio_order_mask,
        )

        # Pack: lyrics + timbre, then + text
        enc_h, enc_m = pack_sequences(
            lyric_enc, timbre_unp, lyric_attention_mask, timbre_mask,
        )
        enc_h, enc_m = pack_sequences(enc_h, text_proj, enc_m, text_attention_mask)

        # --- Prepare context latents (cover mode) ---
        if precomputed_lm_hints_25Hz is not None:
            lm_hints = precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :]
        else:
            if audio_codes is not None:
                lm_5hz = self.tokenizer.quantizer.get_output_from_indices(
                    audio_codes,
                )
            else:
                lm_5hz, _, _ = self._tokenize(
                    src_latents, silence_latent, attention_mask,
                )
            lm_hints = self._detokenize(lm_5hz)
            lm_hints = lm_hints[:, :src_latents.shape[1], :]

        src = torch.where(
            is_covers.unsqueeze(-1).unsqueeze(-1) > 0, lm_hints, src_latents,
        )
        context_latents = torch.cat([src, chunk_masks.to(dtype)], dim=-1)
        return enc_h, enc_m, context_latents

    # -- weight loading --

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load weights from ACE-Step pretrained checkpoint.

        Expects weight names as they appear in the original
        ``AceStepConditionGenerationModel`` state dict, e.g.::

            encoder.text_projector.weight
            encoder.lyric_encoder.layers.0.self_attn.q_proj.weight
            encoder.timbre_encoder.embed_tokens.weight
            tokenizer.audio_acoustic_proj.weight
            detokenizer.embed_tokens.weight
            null_condition_emb

        The caller (pipeline) is responsible for stripping the outer model
        prefix (``model.``) before passing weights here.
        """
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                wl = getattr(param, "weight_loader", default_weight_loader)
                wl(param, loaded_weight)
                loaded.add(name)
            else:
                logger.debug(
                    "Skipping weight %s — not in ACEStepConditionEncoder", name,
                )
        return loaded
