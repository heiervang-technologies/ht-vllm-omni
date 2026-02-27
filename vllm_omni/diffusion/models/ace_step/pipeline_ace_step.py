# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE-Step Pipeline for vLLM-Omni.

This module provides music generation using the ACE-Step model,
integrated with the vLLM-Omni diffusion framework.

ACE-Step is a two-stage music generation model:
1. Condition encoding: text + lyrics + timbre → cross-attention conditioning
2. Flow-matching diffusion: denoise latents conditioned on encoder outputs

The model uses:
- Qwen3-Embedding-0.6B for text encoding
- AutoencoderOobleck for 48kHz stereo audio VAE
- Custom condition encoder (lyric encoder + timbre encoder + packing)
- Custom DiT transformer with flow-matching denoising
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from diffusers import AutoencoderOobleck
from torch import nn
from transformers import AutoModel, AutoTokenizer
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ace_step.ace_step_condition_encoder import (
    ACEStepConditionEncoder,
)
from vllm_omni.diffusion.models.ace_step.ace_step_transformer import (
    AceStepDiTModel,
)
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

# ACE-Step constants
ACESTEP_SAMPLE_RATE = 48000
ACESTEP_LATENT_HZ = 25  # Latent frames per second
ACESTEP_ACOUSTIC_DIM = 64  # audio_acoustic_hidden_dim from config
ACESTEP_TEXT_HIDDEN_DIM = 1024  # Qwen3-Embedding-0.6B hidden dim


def get_ace_step_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create post-processing function for ACE-Step output.

    Converts raw audio tensor to numpy array for saving.
    """

    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio
        # Convert to numpy
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func


class ACEStepPipeline(nn.Module, SupportAudioOutput):
    """
    Pipeline for music generation using ACE-Step.

    This pipeline generates music from text prompts and lyrics using the ACE-Step
    model, integrated with vLLM-Omni's diffusion framework.

    The ACE-Step model uses a flow-matching diffusion process:
    - Timesteps go from 1.0 (pure noise) to 0.0 (clean signal)
    - The DiT predicts a velocity field v_t
    - Euler integration: x_{t-dt} = x_t - v_t * dt

    Args:
        od_config: OmniDiffusion configuration object
        prefix: Weight prefix for loading (default: "")
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.float16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Set up weights sources for transformer (DiT) and condition encoder
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="model",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Load tokenizer (Qwen3-Embedding-0.6B)
        text_encoder_path = os.path.join(model, "Qwen3-Embedding-0.6B")
        if not os.path.exists(text_encoder_path):
            # Fallback: try as a subfolder
            text_encoder_path = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )

        # Load text encoder (Qwen3-Embedding-0.6B)
        self.text_encoder = AutoModel.from_pretrained(
            text_encoder_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to(self.device)
        self.text_encoder.eval()

        # Load VAE (AutoencoderOobleck for 48kHz stereo audio)
        vae_path = os.path.join(model, "vae")
        self.vae = AutoencoderOobleck.from_pretrained(
            vae_path if os.path.exists(vae_path) else model,
            subfolder="vae" if not os.path.exists(vae_path) else None,
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)

        # Initialize custom transformer (DiT) — weights loaded via load_weights
        self.transformer = AceStepDiTModel()

        # Initialize condition encoder — weights loaded via load_weights
        self.condition_encoder = ACEStepConditionEncoder(od_config=od_config)

        # Silence latent for padding (loaded from checkpoint or initialized)
        self.silence_latent = None

        # Null condition embedding for classifier-free guidance
        # This will be loaded from the model weights
        self.null_condition_emb = None

        # Properties
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def encode_text(
        self,
        text: str | list[str],
        device: torch.device,
        max_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt to embeddings using Qwen3-Embedding.

        Args:
            text: Text prompt(s) to encode.
            device: Target device.
            max_length: Maximum token length.

        Returns:
            Tuple of (text_hidden_states, text_attention_mask).
        """
        if isinstance(text, str):
            text = [text]

        text_inputs = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_attention_mask = text_inputs.attention_mask.to(device)

        self.text_encoder.eval()
        with torch.no_grad():
            text_hidden_states = self.text_encoder(
                input_ids=text_input_ids,
            ).last_hidden_state

        return text_hidden_states, text_attention_mask

    def encode_lyrics(
        self,
        lyrics: str | list[str],
        device: torch.device,
        max_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode lyrics to token embeddings.

        The lyrics are tokenized and then passed through the text encoder's
        embedding table (not the full encoder) to get continuous embeddings
        that are processed by the condition encoder's lyric encoder.

        Args:
            lyrics: Lyrics text(s) to encode.
            device: Target device.
            max_length: Maximum token length for lyrics.

        Returns:
            Tuple of (lyric_hidden_states, lyric_attention_mask).
        """
        if isinstance(lyrics, str):
            lyrics = [lyrics]

        lyric_inputs = self.tokenizer(
            lyrics,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        lyric_input_ids = lyric_inputs.input_ids.to(device)
        lyric_attention_mask = lyric_inputs.attention_mask.to(device)

        # Get embeddings from the text encoder's embedding table
        self.text_encoder.eval()
        with torch.no_grad():
            lyric_hidden_states = self.text_encoder.embed_tokens(lyric_input_ids)

        return lyric_hidden_states, lyric_attention_mask

    def prepare_latents(
        self,
        batch_size: int,
        duration_s: float,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare initial noise latents for diffusion.

        Args:
            batch_size: Number of samples to generate.
            duration_s: Target audio duration in seconds.
            dtype: Data type for latents.
            device: Target device.
            generator: Optional random generator for reproducibility.

        Returns:
            Tuple of (noise_latents, attention_mask) where latents have shape
            [B, T, acoustic_dim] with T = duration_s * ACESTEP_LATENT_HZ.
        """
        latent_length = int(duration_s * ACESTEP_LATENT_HZ)
        shape = (batch_size, latent_length, ACESTEP_ACOUSTIC_DIM)

        if generator is not None:
            noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            noise = torch.randn(shape, device=device, dtype=dtype)

        attention_mask = torch.ones(
            batch_size, latent_length, device=device, dtype=dtype
        )
        return noise, attention_mask

    def prepare_silence_refer(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare silence reference audio embeddings (no timbre conditioning).

        When no reference audio is provided, we use the silence latent
        as a placeholder for the timbre encoder input.

        Args:
            batch_size: Number of samples.
            device: Target device.
            dtype: Data type.

        Returns:
            Tuple of (refer_audio_packed, refer_audio_order_mask).
        """
        timbre_fix_frame = 750  # From ACE-Step config default
        if self.silence_latent is not None:
            refer = self.silence_latent[:, :timbre_fix_frame, :].to(
                device=device, dtype=dtype
            )
            if refer.dim() == 2:
                refer = refer.unsqueeze(0)
        else:
            refer = torch.zeros(
                1, timbre_fix_frame, ACESTEP_ACOUSTIC_DIM, device=device, dtype=dtype
            )

        # Pack: one silence reference per batch item
        refer_list = [refer for _ in range(batch_size)]
        refer_packed = torch.cat(refer_list, dim=0)
        order_mask = torch.arange(batch_size, device=device, dtype=torch.long)
        return refer_packed, order_mask

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        lyrics: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        duration: float = 30.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        generator: torch.Generator | None = None,
        output_type: str = "np",
        shift: float = 1.0,
    ) -> DiffusionOutput:
        """
        Generate music from text prompt and lyrics.

        Args:
            req: OmniDiffusionRequest containing generation parameters.
            prompt: Text caption describing the desired music.
            lyrics: Lyrics for the music. Use "[Instrumental]" for no vocals.
            negative_prompt: Negative prompt for CFG (unused in flow-matching,
                uses null_condition_emb instead).
            duration: Target audio duration in seconds.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            generator: Random generator for reproducibility.
            output_type: Output format ("np", "pt", or "latent").
            shift: Timestep shift factor for the schedule.

        Returns:
            DiffusionOutput containing generated audio.
        """
        # Extract parameters from request
        prompt = [
            p if isinstance(p, str) else (p.get("prompt") or "")
            for p in req.prompts
        ] or prompt

        num_inference_steps = (
            req.sampling_params.num_inference_steps or num_inference_steps
        )
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(
                req.sampling_params.seed
            )

        # Extract ACE-Step-specific params from extra_args
        extra = req.sampling_params.extra_args
        duration = extra.get("duration", duration)
        lyrics = extra.get("lyrics", lyrics)
        shift = extra.get("shift", shift)

        if lyrics is None:
            lyrics = "[Instrumental]"

        device = self.device
        dtype = getattr(self.od_config, "dtype", torch.float16)
        do_cfg = guidance_scale > 1.0
        self._guidance_scale = guidance_scale

        # Determine batch size
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        if isinstance(lyrics, str):
            lyrics = [lyrics] * batch_size

        # 1. Encode text prompt
        text_hidden_states, text_attention_mask = self.encode_text(
            prompt, device, max_length=256
        )

        # 2. Encode lyrics
        lyric_hidden_states, lyric_attention_mask = self.encode_lyrics(
            lyrics, device, max_length=1024
        )

        # 3. Prepare reference audio (silence = no timbre conditioning)
        refer_packed, refer_order_mask = self.prepare_silence_refer(
            batch_size, device, dtype
        )

        # 4. Prepare latents (noise)
        noise, attention_mask = self.prepare_latents(
            batch_size, duration, dtype, device, generator
        )
        latent_length = noise.shape[1]

        # 5. Prepare source latents and chunk masks for condition encoder
        # For text2music (no cover), src_latents are silence
        if self.silence_latent is not None:
            silence = self.silence_latent.to(device=device, dtype=dtype)
            if silence.dim() == 2:
                silence = silence.unsqueeze(0)
            src_latents = silence[:, :latent_length, :].expand(
                batch_size, -1, -1
            )
        else:
            src_latents = torch.zeros(
                batch_size,
                latent_length,
                ACESTEP_ACOUSTIC_DIM,
                device=device,
                dtype=dtype,
            )

        chunk_masks = torch.ones(
            batch_size,
            latent_length,
            ACESTEP_ACOUSTIC_DIM,
            device=device,
            dtype=dtype,
        )
        is_covers = torch.zeros(batch_size, device=device, dtype=torch.long)

        # 6. Run condition encoder
        encoder_hidden_states, encoder_attention_mask, context_latents = (
            self.condition_encoder(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_packed=refer_packed,
                refer_audio_order_mask=refer_order_mask,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                attention_mask=attention_mask,
                is_covers=is_covers,
                silence_latent=self.silence_latent,
            )
        )

        # 7. Prepare CFG: duplicate conditions with null embedding
        if do_cfg:
            null_emb = self.null_condition_emb
            if null_emb is None:
                # Fallback: zeros
                null_emb = torch.zeros_like(encoder_hidden_states[:1])
            null_emb = null_emb.expand_as(encoder_hidden_states)
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, null_emb], dim=0
            )
            encoder_attention_mask = torch.cat(
                [encoder_attention_mask, encoder_attention_mask], dim=0
            )
            context_latents = torch.cat(
                [context_latents, context_latents], dim=0
            )
            attention_mask = torch.cat(
                [attention_mask, attention_mask], dim=0
            )

        # 8. Build timestep schedule (linear from 1.0 to 0.0)
        t_schedule = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype
        )
        if shift != 1.0:
            t_schedule = shift * t_schedule / (1 + (shift - 1) * t_schedule)

        self._num_timesteps = num_inference_steps
        xt = noise

        # 9. Denoising loop (flow matching Euler ODE integration)
        for step_idx, (t_curr, t_prev) in enumerate(
            zip(t_schedule[:-1], t_schedule[1:])
        ):
            self._current_timestep = t_curr

            # Expand for CFG
            x_input = torch.cat([xt, xt], dim=0) if do_cfg else xt
            t_tensor = t_curr * torch.ones(
                x_input.shape[0], device=device, dtype=dtype
            )

            # DiT forward pass
            vt = self.transformer(
                hidden_states=x_input,
                timestep=t_tensor,
                timestep_r=t_tensor,
                encoder_hidden_states=encoder_hidden_states,
                context_latents=context_latents,
            )

            # Apply CFG
            if do_cfg:
                vt_cond, vt_uncond = vt.chunk(2)
                vt = vt_uncond + guidance_scale * (vt_cond - vt_uncond)

            # Euler step: x_{t-dt} = x_t - v_t * dt
            dt = t_curr - t_prev
            xt = xt - vt * dt

        self._current_timestep = None

        # 10. VAE decode
        if output_type == "latent":
            audio = xt
        else:
            # ACE-Step VAE expects [B, C, T] (channels-first)
            latents_for_vae = xt.transpose(1, 2).to(dtype=self.vae.dtype)
            audio = self.vae.decode(latents_for_vae).sample

        return DiffusionOutput(output=audio)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
