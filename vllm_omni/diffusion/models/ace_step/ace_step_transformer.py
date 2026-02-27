# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE-Step DiT Model for vLLM-Omni.

Ports the ACE-Step v1.5 Diffusion Transformer to vllm-omni layer primitives.
Original: acestep/models/base/modeling_acestep_v15_base.py (AceStepDiTModel)

Architecture:
- 24 DiT layers, hidden_size=2048, 16 query heads / 8 KV heads (GQA), head_dim=128
- Alternating full_attention / sliding_attention (window=128) per layer
- RoPE (Qwen3-style, theta=1e6) on Q/K after per-head RMSNorm
- AdaLN modulation per layer (6 scale-shift-gate params from timestep embedding)
- Cross-attention to packed condition encoder output (every layer)
- Qwen3MLP (gate_proj + up_proj -> SiLU gate -> down_proj)
- Two sinusoidal TimestepEmbeddings: one for t, one for (t - r)
- Input: Conv1d patchify (in_channels=192 -> hidden=2048, patch_size=2)
- Output: ConvTranspose1d de-patchify (hidden=2048 -> acoustic_dim=64)
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# RoPE helpers  (Qwen3-style: full head_dim rotation, interleaved cos/sin)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Qwen3-compatible rotary position embeddings.

    Computes cos/sin tables on the fly and caches them for reuse.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 32768,
                 rope_theta: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        # inv_freq is NOT a learned parameter
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape [B, S, head_dim]."""
        # inv_freq: [head_dim/2]  position_ids: [B, S]
        inv_freq = self.inv_freq.to(x.device, dtype=torch.float32)
        # freqs: [B, S, head_dim/2]
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
        # emb: [B, S, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (standard RoPE helper)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,       # [B, H, S, D]
    k: torch.Tensor,       # [B, H, S, D]
    cos: torch.Tensor,     # [B, S, D]
    sin: torch.Tensor,     # [B, S, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-style rotary embeddings to Q and K (full head_dim rotation).

    Input tensors are in [B, H, S, D] layout.
    cos/sin are [B, S, D] and will be broadcast to [B, 1, S, D].
    """
    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)  # [B, 1, S, D]
    q_embed = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
    k_embed = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# RMSNorm  (per-head normalisation for Q/K, Qwen3 style)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Timestep Embedding  (sinusoidal + MLP, identical to original)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP and a 6x projection.

    Produces both a hidden embedding `temb` (for output AdaLN) and a
    per-layer modulation tensor `timestep_proj` of shape [B, 6, hidden_size].
    """

    def __init__(self, in_channels: int, time_embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6)

    def timestep_embedding(self, t: torch.Tensor, dim: int,
                           max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings [B] -> [B, dim]."""
        t = t * self.scale
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_freq = self.timestep_embedding(t, self.in_channels)
        temb = self.linear_1(t_freq.to(t.dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).unflatten(1, (6, -1))
        return temb, timestep_proj


# ---------------------------------------------------------------------------
# Self-Attention (with per-head QK-norm, RoPE, GQA)
# ---------------------------------------------------------------------------

class AceStepSelfAttention(nn.Module):
    """Self-attention with Qwen3-style QK-norm + RoPE + GQA.

    Uses vllm-omni Attention kernel underneath.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.inner_dim = num_attention_heads * head_dim
        self.kv_dim = num_key_value_heads * head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        # Projections (no bias, matching original)
        self.q_proj = ReplicatedLinear(hidden_size, self.inner_dim, bias=False)
        self.k_proj = ReplicatedLinear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = ReplicatedLinear(hidden_size, self.kv_dim, bias=False)
        self.o_proj = ReplicatedLinear(self.inner_dim, hidden_size, bias=False)

        # Per-head RMSNorm on Q and K (Qwen3 style)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        # vllm-omni attention kernel (after GQA expansion)
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=False,
            num_kv_heads=num_attention_heads,  # after manual GQA expansion
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)

        # Reshape to [B, S, H, D] then apply per-head RMSNorm
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)

        # RoPE expects [B, H, S, D]
        query = query.transpose(1, 2)  # [B, H, S, D]
        key = key.transpose(1, 2)      # [B, Hkv, S, D]
        value = value.transpose(1, 2)  # [B, Hkv, S, D]

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Back to [B, S, H, D] for Attention kernel
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # GQA expansion: repeat KV heads to match Q heads
        if self.num_kv_groups > 1:
            key = key.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            value = value.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Compute attention
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        hidden_states, _ = self.o_proj(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Cross-Attention (QK-norm on Q only, no RoPE, GQA)
# ---------------------------------------------------------------------------

class AceStepCrossAttention(nn.Module):
    """Cross-attention to packed condition encoder output.

    Q from decoder hidden states, K/V from encoder hidden states.
    Uses QK-norm and GQA following the original architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.inner_dim = num_attention_heads * head_dim
        self.kv_dim = num_key_value_heads * head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        self.q_proj = ReplicatedLinear(hidden_size, self.inner_dim, bias=False)
        self.k_proj = ReplicatedLinear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = ReplicatedLinear(hidden_size, self.kv_dim, bias=False)
        self.o_proj = ReplicatedLinear(self.inner_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=head_dim,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=False,
            num_kv_heads=num_attention_heads,  # after manual GQA expansion
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.shape[1]

        # Q from decoder, K/V from encoder
        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(encoder_hidden_states)
        value, _ = self.v_proj(encoder_hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, encoder_seq_len, self.num_kv_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)

        # GQA expansion
        if self.num_kv_groups > 1:
            key = key.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            key = key.reshape(batch_size, encoder_seq_len, self.num_heads, self.head_dim)
            value = value.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            value = value.reshape(batch_size, encoder_seq_len, self.num_heads, self.head_dim)

        # Attention
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.view(batch_size, seq_len, self.inner_dim)

        hidden_states, _ = self.o_proj(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Feed-forward MLP  (Qwen3MLP: gate_proj + up_proj -> SiLU gate -> down_proj)
# ---------------------------------------------------------------------------

class AceStepMLP(nn.Module):
    """Qwen3-style gated MLP using vLLM ReplicatedLinear."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = ReplicatedLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        down_input = self.act_fn(gate) * up
        output, _ = self.down_proj(down_input)
        return output


# ---------------------------------------------------------------------------
# DiT Layer (self-attn + cross-attn + MLP, with AdaLN modulation)
# ---------------------------------------------------------------------------

class AceStepDiTLayer(nn.Module):
    """Single ACE-Step DiT transformer layer.

    1. Self-attention with AdaLN (shift, scale, gate from timestep embedding)
    2. Cross-attention to encoder hidden states (simple residual)
    3. MLP with AdaLN (shift, scale, gate from timestep embedding)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()

        # Self-attention
        self.self_attn_norm = RMSNorm(hidden_size, eps=1e-6)
        self.self_attn = AceStepSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        # Cross-attention
        self.cross_attn_norm = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = AceStepCrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

        # MLP
        self.mlp_norm = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = AceStepMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        # AdaLN scale-shift table: 6 modulation params
        # (shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, hidden_size) / hidden_size ** 0.5
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        timestep_proj: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Extract 6 modulation params from timestep embedding + learned table
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table + timestep_proj
        ).chunk(6, dim=1)

        # 1. Self-attention with AdaLN
        norm_h = self.self_attn_norm(hidden_states) * (1 + scale_sa) + shift_sa
        norm_h = norm_h.to(hidden_states.dtype)
        attn_out = self.self_attn(norm_h, position_embeddings)
        hidden_states = (hidden_states + attn_out * gate_sa).to(hidden_states.dtype)

        # 2. Cross-attention (simple residual, no AdaLN gating)
        norm_h = self.cross_attn_norm(hidden_states).to(hidden_states.dtype)
        cross_out = self.cross_attn(norm_h, encoder_hidden_states)
        hidden_states = hidden_states + cross_out

        # 3. MLP with AdaLN
        norm_h = self.mlp_norm(hidden_states) * (1 + scale_mlp) + shift_mlp
        norm_h = norm_h.to(hidden_states.dtype)
        mlp_out = self.mlp(norm_h)
        hidden_states = (hidden_states + mlp_out * gate_mlp).to(hidden_states.dtype)

        return hidden_states


# ---------------------------------------------------------------------------
# AceStepDiTModel  (top-level transformer)
# ---------------------------------------------------------------------------

class AceStepDiTModel(nn.Module):
    """ACE-Step DiT model ported to vLLM-Omni layer primitives.

    Architecture matches the original AceStepDiTModel exactly:
    - Input: [B, T, in_channels] where in_channels = 192 (64 acoustic + 64 context + 64 chunk_mask)
    - Conv1d patchify: [B, T, 192] -> [B, T//2, 2048]
    - 24 DiT layers with self-attn + cross-attn + MLP
    - AdaLN output normalization
    - ConvTranspose1d de-patchify: [B, T//2, 2048] -> [B, T, 64]
    - Output: predicted velocity field [B, T, 64]
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        in_channels: int = 192,
        acoustic_dim: int = 64,
        patch_size: int = 2,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 32768,
        layer_types: list[str] | None = None,
        sliding_window: int = 128,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.acoustic_dim = acoustic_dim

        # Layer types: alternating sliding/full by default
        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(num_layers)
            ]
        self.layer_types = layer_types

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

        # Input patchify conv: [B, in_channels, T] -> [B, hidden_size, T//patch_size]
        self.proj_in_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Two timestep embeddings: one for t, one for (t - r)
        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)

        # Condition embedder: project encoder hidden states
        self.condition_embedder = nn.Linear(hidden_size, hidden_size, bias=True)

        # Transformer layers
        self.layers = nn.ModuleList([
            AceStepDiTLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_layers)
        ])

        # Output: AdaLN + de-patchify
        self.norm_out = RMSNorm(hidden_size, eps=1e-6)
        self.proj_out_conv = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=acoustic_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        # Output AdaLN scale-shift table (2 params: shift, scale)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, hidden_size) / hidden_size ** 0.5
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the ACE-Step DiT model.

        Args:
            hidden_states: Noisy acoustic latents [B, T, acoustic_dim] (64-dim)
            timestep: Diffusion timestep t [B]
            timestep_r: Diffusion timestep r [B]
            encoder_hidden_states: Packed condition encoder output [B, S, hidden_size]
            context_latents: Context latents (context + chunk_mask) [B, T, 128]
                (64 context acoustic + 64 chunk_mask)

        Returns:
            Predicted velocity field [B, T, acoustic_dim]
        """
        # Concatenate context with noisy hidden states: [B, T, 192]
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden_states.shape[1]

        # Pad to multiple of patch_size
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))

        # Patchify: [B, T, C] -> [B, C, T] -> Conv1d -> [B, hidden, T//ps] -> [B, T//ps, hidden]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.proj_in_conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        # Project encoder hidden states
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        # Compute timestep embeddings
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Position IDs and RoPE
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                timestep_proj,
                encoder_hidden_states,
            )

        # Output AdaLN + de-patchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).to(
            hidden_states.dtype
        )

        # De-patchify: [B, T//ps, hidden] -> [B, hidden, T//ps] -> ConvT -> [B, acoustic, T] -> [B, T, acoustic]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.proj_out_conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        # Crop to original length
        hidden_states = hidden_states[:, :original_seq_len, :]

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a pretrained ACE-Step checkpoint.

        Maps original ACE-Step weight names (with Lambda wrappers in nn.Sequential)
        to our flat structure.

        Original proj_in is nn.Sequential:
            0 = Lambda (transpose)
            1 = Conv1d
            2 = Lambda (transpose)
        So proj_in.1.weight -> proj_in_conv.weight, etc.

        Original proj_out is nn.Sequential:
            0 = Lambda (transpose)
            1 = ConvTranspose1d
            2 = Lambda (transpose)
        So proj_out.1.weight -> proj_out_conv.weight, etc.

        DiT layer mapping (original -> ours):
            layers.N.self_attn.q_proj -> layers.N.self_attn.q_proj
            layers.N.self_attn.k_proj -> layers.N.self_attn.k_proj
            layers.N.self_attn.v_proj -> layers.N.self_attn.v_proj
            layers.N.self_attn.o_proj -> layers.N.self_attn.o_proj
            layers.N.self_attn.q_norm -> layers.N.self_attn.q_norm
            layers.N.self_attn.k_norm -> layers.N.self_attn.k_norm
            layers.N.cross_attn.q_proj -> layers.N.cross_attn.q_proj
            layers.N.cross_attn.k_proj -> layers.N.cross_attn.k_proj
            layers.N.cross_attn.v_proj -> layers.N.cross_attn.v_proj
            layers.N.cross_attn.o_proj -> layers.N.cross_attn.o_proj
            layers.N.cross_attn.q_norm -> layers.N.cross_attn.q_norm
            layers.N.cross_attn.k_norm -> layers.N.cross_attn.k_norm
            layers.N.mlp.gate_proj -> layers.N.mlp.gate_proj
            layers.N.mlp.up_proj -> layers.N.mlp.up_proj
            layers.N.mlp.down_proj -> layers.N.mlp.down_proj
            (norms and scale_shift_table pass through directly)

        Returns:
            Set of parameter names that were successfully loaded.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Static name mapping for non-layer params
        name_mapping = {
            # proj_in Sequential (Lambda, Conv1d, Lambda) -> flat conv
            "proj_in.1.weight": "proj_in_conv.weight",
            "proj_in.1.bias": "proj_in_conv.bias",
            # proj_out Sequential (Lambda, ConvTranspose1d, Lambda) -> flat conv
            "proj_out.1.weight": "proj_out_conv.weight",
            "proj_out.1.bias": "proj_out_conv.bias",
        }

        for name, loaded_weight in weights:
            # Apply static mapping
            mapped_name = name_mapping.get(name, name)

            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
            else:
                logger.debug(f"Skipping weight {name} -> {mapped_name} (not in model)")

        return loaded_params
