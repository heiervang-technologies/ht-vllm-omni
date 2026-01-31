import asyncio
import struct
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    CreateAudio,
    OpenAICreateSpeechRequest,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# TTS Configuration (currently supports Qwen3-TTS)
_TTS_MODEL_STAGES: set[str] = {"qwen3_tts"}
_TTS_SPEAKERS: set[str] = {
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
}
_TTS_LANGUAGES: set[str] = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}
_TTS_MAX_INSTRUCTIONS_LENGTH = 500
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    def _is_tts_model(self) -> bool:
        """Check if the current model is a supported TTS model."""
        stage_list = getattr(self.engine_client, "stage_list", None)
        if stage_list:
            for stage in stage_list:
                model_stage = getattr(stage, "model_stage", None)
                if model_stage in _TTS_MODEL_STAGES:
                    return True
        return False

    def _validate_tts_request(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Validate TTS request parameters. Returns error message or None."""
        task_type = request.task_type or "CustomVoice"

        # Validate input is not empty
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Validate language
        if request.language is not None and request.language not in _TTS_LANGUAGES:
            return f"Invalid language '{request.language}'. Supported: {', '.join(sorted(_TTS_LANGUAGES))}"

        # Validate speaker for CustomVoice task
        if task_type == "CustomVoice" and request.voice is not None:
            if request.voice not in _TTS_SPEAKERS:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(_TTS_SPEAKERS))}"

        # Validate Base task requirements
        if task_type == "Base":
            if request.ref_audio is None:
                return "Base task requires 'ref_audio' for voice cloning"
            # Validate ref_audio format
            if not (request.ref_audio.startswith(("http://", "https://")) or request.ref_audio.startswith("data:")):
                return "ref_audio must be a URL (http/https) or base64 data URL (data:...)"

        # Validate cross-parameter dependencies
        if task_type != "Base":
            if request.ref_text is not None:
                return "'ref_text' is only valid for Base task"
            if request.x_vector_only_mode is not None:
                return "'x_vector_only_mode' is only valid for Base task"

        # Validate VoiceDesign task requirements
        if task_type == "VoiceDesign" and not request.instructions:
            return "VoiceDesign task requires 'instructions' to describe the voice"

        # Validate instructions length
        if request.instructions and len(request.instructions) > _TTS_MAX_INSTRUCTIONS_LENGTH:
            return f"Instructions too long (max {_TTS_MAX_INSTRUCTIONS_LENGTH} characters)"

        # Validate max_new_tokens range
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"

        return None

    def _build_tts_prompt(self, text: str) -> str:
        """Build TTS prompt from input text."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_tts_params(self, request: OpenAICreateSpeechRequest) -> dict[str, Any]:
        """Build TTS parameters from request.

        Processes each parameter if present, skips if not.
        Values are wrapped in lists as required by the model.
        """
        params: dict[str, Any] = {}

        # Text content (always required)
        params["text"] = [request.input]

        # Task type
        if request.task_type is not None:
            params["task_type"] = [request.task_type]
        else:
            params["task_type"] = ["CustomVoice"]

        # Language
        if request.language is not None:
            params["language"] = [request.language]
        else:
            params["language"] = ["Auto"]

        # Speaker (voice)
        if request.voice is not None:
            params["speaker"] = [request.voice]
        elif params["task_type"][0] == "CustomVoice":
            params["speaker"] = ["Vivian"]  # Default for CustomVoice

        # Instructions for style/emotion control
        if request.instructions is not None:
            params["instruct"] = [request.instructions]
        else:
            params["instruct"] = [""]

        # Voice clone parameters (used with Base task)
        if request.ref_audio is not None:
            params["ref_audio"] = [request.ref_audio]
        if request.ref_text is not None:
            params["ref_text"] = [request.ref_text]
        if request.x_vector_only_mode is not None:
            params["x_vector_only_mode"] = [request.x_vector_only_mode]

        # Generation parameters
        if request.max_new_tokens is not None:
            params["max_new_tokens"] = [request.max_new_tokens]
        else:
            params["max_new_tokens"] = [2048]

        return params

    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.

        For Qwen3-TTS models, additional parameters are supported:
        - task_type: "CustomVoice", "VoiceDesign", or "Base"
        - language: Language code (e.g., "Chinese", "English", "Auto")
        - voice: Speaker name (e.g., "Vivian", "Ryan") for CustomVoice
        - instructions: Voice style/emotion instructions
        - ref_audio: Reference audio for voice cloning (Base task)
        - ref_text: Transcript of reference audio (Base task)
        - x_vector_only_mode: Use speaker embedding only (Base task)

        When ``stream=True``, audio chunks are streamed progressively as
        they are generated by the model (model-level streaming). Each chunk
        is sent as raw PCM data as soon as it becomes available, providing
        low latency time-to-first-audio.

        For WAV format with streaming, a WAV header with max-size placeholder
        is sent first, followed by raw PCM chunks. For PCM format, raw 16-bit
        signed PCM samples are streamed directly.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"

        try:
            if self._is_tts_model():
                # Validate TTS parameters
                validation_error = self._validate_tts_request(request)
                if validation_error:
                    return self.create_error_response(validation_error)

                # Build TTS parameters and prompt
                tts_params = self._build_tts_params(request)

                # Enable model-level streaming when client requests it
                if request.stream:
                    tts_params["streaming"] = [True]

                prompt_text = self._build_tts_prompt(request.input)
                prompt = {
                    "prompt": prompt_text,
                    "additional_information": tts_params,
                }
            else:
                # Fallback for unsupported models
                tts_params = {}
                prompt = {"prompt": request.input}

            logger.info(
                "TTS speech request %s: text=%r, task_type=%s, stream=%s",
                request_id,
                request.input[:50] + "..." if len(request.input) > 50 else request.input,
                tts_params.get("task_type", ["unknown"])[0],
                request.stream,
            )

            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            # ---- Progressive streaming: yield audio chunks as they arrive ----
            if request.stream:
                response_format = (request.response_format or "wav").lower()
                return StreamingResponse(
                    self._stream_progressive_audio(generator, response_format),
                    media_type=_STREAM_MEDIA_TYPES.get(response_format, "audio/wav"),
                )

            # ---- Non-streaming: collect full output then respond ----
            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            # Extract audio from output
            audio_output = self._extract_audio_output(final_output)
            if audio_output is None:
                return self.create_error_response("TTS model did not produce audio output.")

            audio_tensor = audio_output["audio"]
            sample_rate = audio_output.get("sr", 24000)
            if hasattr(sample_rate, "item"):
                sample_rate = sample_rate.item()
            sample_rate = int(sample_rate)

            # Convert tensor to numpy
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()

            # Squeeze batch dimension if present, but preserve channel dimension for stereo
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            # Apply speed adjustment if needed
            speed = request.speed or 1.0
            if speed != 1.0:
                audio_obj = CreateAudio(
                    audio_tensor=audio_tensor,
                    sample_rate=sample_rate,
                    response_format=request.response_format or "wav",
                    speed=speed,
                    stream_format=request.stream_format,
                    base64_encode=False,
                )
                audio_response: AudioResponse = self.create_audio(audio_obj)
                return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                response_format=request.response_format or "wav",
                speed=1.0,
                stream_format=request.stream_format,
                base64_encode=False,
            )

            audio_response = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))
        except Exception as e:
            logger.exception("Speech generation failed: %s", e)
            return self.create_error_response(f"Speech generation failed: {e}")

    async def _stream_progressive_audio(
        self,
        generator,
        response_format: str,
    ):
        """Stream audio chunks progressively as they arrive from the model.

        The output processor accumulates multimodal tensors, so each yielded
        output contains ALL audio chunks produced so far (not just the latest
        one).  We track a cursor to yield only new chunks.

        For WAV format: yields a WAV header with max-size placeholder first,
        then raw PCM chunks as they become available.
        For PCM format: yields raw 16-bit signed PCM chunks directly.
        """
        header_sent = False
        sample_rate = 24000  # Default, updated from first chunk
        chunks_yielded = 0  # Cursor: how many audio chunks we've already sent

        async for output in generator:
            audio_output = self._extract_audio_output(output)
            if audio_output is None:
                continue

            sr = audio_output.get("sr", 24000)
            if hasattr(sr, "item"):
                sr = sr.item()
            elif isinstance(sr, (list, tuple)):
                sr = sr[0] if sr else 24000
            sample_rate = int(sr)

            # The output processor accumulates audio tensors in a list.
            # First output: audio_data is a single tensor.
            # Subsequent outputs: audio_data is [tensor1, tensor2, ...].
            audio_data = audio_output["audio"]
            if isinstance(audio_data, list) and audio_data and hasattr(audio_data[0], "float"):
                # List of tensors from accumulation — get only new ones
                new_tensors = audio_data[chunks_yielded:]
                chunks_yielded = len(audio_data)
            elif isinstance(audio_data, list) and audio_data and isinstance(audio_data[0], list):
                # List of lists (deserialized tensors) — get only new ones
                new_tensors = audio_data[chunks_yielded:]
                chunks_yielded = len(audio_data)
            else:
                # Single tensor or first chunk
                new_tensors = [audio_data]
                chunks_yielded = 1

            for audio_tensor in new_tensors:
                # Convert to numpy — after ZMQ deserialization tensors
                # may arrive as lists, torch tensors, or numpy arrays.
                if isinstance(audio_tensor, list):
                    audio_tensor = np.array(audio_tensor, dtype=np.float32)
                elif hasattr(audio_tensor, "float"):
                    audio_tensor = audio_tensor.float().detach().cpu().numpy()
                if hasattr(audio_tensor, "ndim") and audio_tensor.ndim > 1:
                    audio_tensor = audio_tensor.squeeze()

                if len(audio_tensor) == 0:
                    continue

                # Convert float audio to int16 PCM
                audio_float = audio_tensor.astype(np.float32)
                audio_float = np.clip(audio_float, -1.0, 1.0)
                pcm_chunk = (audio_float * 32767).astype(np.int16).tobytes()

                if response_format == "wav" and not header_sent:
                    yield _make_wav_header(sample_rate, 0x7FFFFFFF)
                    header_sent = True

                yield pcm_chunk

    @staticmethod
    def _extract_audio_output(output: OmniRequestOutput) -> dict | None:
        """Extract audio dict from OmniRequestOutput, trying both locations."""
        audio_output = None
        if hasattr(output, "multimodal_output") and output.multimodal_output:
            audio_output = output.multimodal_output
        if (not audio_output or "audio" not in audio_output) and hasattr(output, "request_output"):
            if output.request_output and hasattr(output.request_output, "multimodal_output"):
                audio_output = output.request_output.multimodal_output
        if audio_output and "audio" in audio_output:
            return audio_output
        return None

    @staticmethod
    async def _stream_audio_bytes(data: bytes, chunk_size: int = 16384):
        """Yield pre-encoded audio bytes in fixed-size chunks."""
        offset = 0
        while offset < len(data):
            yield data[offset : offset + chunk_size]
            offset += chunk_size

    @staticmethod
    async def _stream_pcm_audio(
        audio: np.ndarray,
        sample_rate: int,
        response_format: str,
        chunk_duration_ms: int = 500,
    ):
        """Stream audio as chunked PCM data.

        For WAV format: yields a complete WAV header first, then raw PCM chunks.
        For PCM format: yields raw 16-bit signed PCM chunks directly.
        """
        # Convert float audio to int16 PCM
        audio_float = audio.astype(np.float32)
        audio_float = np.clip(audio_float, -1.0, 1.0)
        pcm_data = (audio_float * 32767).astype(np.int16).tobytes()

        samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        bytes_per_chunk = samples_per_chunk * 2  # 16-bit = 2 bytes per sample

        if response_format == "wav":
            # Write a complete WAV header with known data size
            yield _make_wav_header(sample_rate, len(pcm_data))

        # Stream PCM data in chunks
        offset = 0
        while offset < len(pcm_data):
            yield pcm_data[offset : offset + bytes_per_chunk]
            offset += bytes_per_chunk


_STREAM_MEDIA_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def _make_wav_header(sample_rate: int, data_size: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Build a standard WAV (RIFF) header for PCM data."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,         # file size - 8
        b"WAVE",
        b"fmt ",
        16,                     # fmt chunk size
        1,                      # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header
