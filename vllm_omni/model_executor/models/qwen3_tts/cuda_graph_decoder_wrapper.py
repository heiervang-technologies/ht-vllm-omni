# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import torch
from torch.cuda import CUDAGraph
from typing import Dict, List, Optional, Tuple
from vllm.logger import init_logger

logger = init_logger(__name__)


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder, capture_sizes=[25, 50, 100, 200, 300])
        wrapper.warmup(device)

        # During inference:
        output = wrapper.decode(codes)  # Automatically uses CUDA graph if possible
    """

    # Default capture sizes (in terms of code sequence length)
    # These should cover common chunk sizes used in chunked_decode
    DEFAULT_CAPTURE_SIZES = [25, 50, 100, 150, 200, 250, 300, 400, 500]

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: Optional[List[int]] = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        """
        Initialize the CUDA Graph wrapper.

        Args:
            decoder: The Qwen3TTSTokenizerV2Decoder module
            capture_sizes: List of code sequence lengths to capture graphs for
            num_quantizers: Number of quantizers (codebook layers)
            enabled: Whether CUDA graph is enabled
        """
        self.decoder = decoder
        self.capture_sizes = capture_sizes or self.DEFAULT_CAPTURE_SIZES
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        # CUDA graph storage
        self.graphs: Dict[int, CUDAGraph] = {}
        self.static_inputs: Dict[int, torch.Tensor] = {}
        self.static_outputs: Dict[int, torch.Tensor] = {}

        self._warmed_up = False
        self._device = None

    def _get_padded_size(self, actual_size: int) -> Optional[int]:
        """
        Get the smallest capture size that can accommodate the actual size.

        Args:
            actual_size: The actual code sequence length

        Returns:
            The padded size, or None if no suitable size exists
        """
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        """
        Warmup: capture CUDA graphs for all predefined sizes.

        This should be called once during model initialization.

        Args:
            device: The device to capture graphs on
            dtype: Data type for input codes (usually torch.long)
        """
        if not self.enabled:
            logger.info("CUDA Graph is disabled, skipping warmup")
            return

        if self._warmed_up:
            logger.warning("CUDA Graph already warmed up, skipping")
            return

        self._device = device
        self.decoder.eval()

        logger.info(f"Starting CUDA Graph warmup for {len(self.capture_sizes)} sizes: {self.capture_sizes}")

        # Warmup runs to ensure CUDA memory is allocated
        for size in self.capture_sizes:
            dummy_codes = torch.zeros(
                1, self.num_quantizers, size,
                dtype=dtype,
                device=device
            )
            with torch.no_grad():
                _ = self.decoder(dummy_codes)

        # Synchronize before capturing
        torch.cuda.synchronize(device)

        # Capture graphs for each size
        for size in self.capture_sizes:
            try:
                self._capture_graph_for_size(size, device, dtype)
                logger.info(f"  Captured CUDA Graph for size={size}")
            except Exception as e:
                logger.warning(f"  Failed to capture CUDA Graph for size={size}: {e}")
                # Continue with other sizes

        self._warmed_up = True
        logger.info(f"CUDA Graph warmup complete. Captured {len(self.graphs)} graphs.")

    def _capture_graph_for_size(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        """
        Capture a CUDA graph for a specific input size.

        Args:
            size: Code sequence length
            device: Device to capture on
            dtype: Input dtype
        """
        # Create static input buffer
        static_input = torch.zeros(
            1, self.num_quantizers, size,
            dtype=dtype,
            device=device
        )

        # Warmup run (required before capture)
        with torch.no_grad():
            _ = self.decoder(static_input)

        torch.cuda.synchronize(device)

        # Capture the graph
        graph = CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.decoder(static_input)

        # Store everything
        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_outputs[size] = static_output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode codes to audio using CUDA graph if possible.

        Args:
            codes: Input codes tensor of shape (batch, num_quantizers, seq_len)

        Returns:
            Decoded audio tensor
        """
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        # Only support batch size 1 for now
        if codes.shape[0] != 1:
            logger.debug("Batch size > 1, falling back to eager execution")
            return self.decoder(codes)

        actual_size = codes.shape[-1]
        padded_size = self._get_padded_size(actual_size)

        if padded_size is None or padded_size not in self.graphs:
            # Size too large or not captured, fall back to eager
            logger.debug(f"Size {actual_size} not captured, falling back to eager execution")
            return self.decoder(codes)

        # Copy input to static buffer (with padding if needed)
        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:, :, :actual_size] = codes

        # Replay the graph
        self.graphs[padded_size].replay()

        # Get output and trim to actual size
        output = self.static_outputs[padded_size]

        # Calculate actual output length based on upsample ratio
        total_upsample = self.decoder.total_upsample
        actual_output_len = actual_size * total_upsample

        return output[..., :actual_output_len].clone()

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25
    ) -> torch.Tensor:
        """
        Chunked decode with CUDA graph acceleration.

        This replaces the original chunked_decode method with CUDA graph support.

        Args:
            codes: Input codes tensor of shape (batch, num_quantizers, seq_len)
            chunk_size: Size of each chunk
            left_context_size: Context size for overlap

        Returns:
            Decoded audio tensor
        """
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            # Extract chunk with context
            codes_chunk = codes[..., start_index - context_size : end_index]

            # Decode using CUDA graph
            wav_chunk = self.decode(codes_chunk)

            # Trim context from output
            wavs.append(wav_chunk[..., context_size * total_upsample :])
            start_index = end_index

        return torch.cat(wavs, dim=-1)


def patch_decoder_with_cudagraph(
    decoder: torch.nn.Module,
    capture_sizes: Optional[List[int]] = None,
    enabled: bool = True,
) -> CUDAGraphDecoderWrapper:
    """
    Patch a Qwen3TTSTokenizerV2Decoder with CUDA graph support.

    This is a convenience function to easily add CUDA graph support to an existing decoder.

    Args:
        decoder: The decoder module to patch
        capture_sizes: List of sizes to capture graphs for
        enabled: Whether to enable CUDA graph

    Returns:
        CUDAGraphDecoderWrapper instance

    Example:
        decoder = model.speech_tokenizer.decoder
        wrapper = patch_decoder_with_cudagraph(decoder)
        wrapper.warmup(device)

        # Replace the original chunked_decode
        model.speech_tokenizer.decoder.chunked_decode = wrapper.chunked_decode_with_cudagraph
    """
    # Get num_quantizers from decoder config
    num_quantizers = getattr(decoder.config, 'num_quantizers', 8)

    wrapper = CUDAGraphDecoderWrapper(
        decoder=decoder,
        capture_sizes=capture_sizes,
        num_quantizers=num_quantizers,
        enabled=enabled,
    )

    return wrapper
