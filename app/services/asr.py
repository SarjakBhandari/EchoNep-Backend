from __future__ import annotations

import base64
import binascii
import io
import time
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def load_asr_model():
    if settings.asr_backend == "faster_whisper":
        from faster_whisper import WhisperModel

        return WhisperModel(
            settings.whisper_model_size,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
            cpu_threads=settings.asr_cpu_threads,
        )

    if settings.asr_backend == "whisper_openai":
        import whisper

        return whisper.load_model(settings.whisper_model_size)

    return None


def preload_asr_model() -> str:
    model = load_asr_model()
    if model is None:
        return "unavailable"
    return settings.asr_backend


def _asr_language_for_user_type(user_type: str | None, direction: str) -> str:
    if user_type == "tourist":
        return "en"
    if user_type == "trader":
        return "ne"
    return "en" if direction == "en_np" else "ne"


def transcribe_audio(audio_b64: str, direction: str, user_type: str | None = None) -> dict:
    started_at = time.perf_counter()

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"Invalid base64 audio payload: {error}") from error

    if not audio_bytes:
        raise ValueError("Empty audio payload")

    model = load_asr_model()
    if model is None:
        return {
            "text": "",
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "backend": "unavailable",
        }

    language = _asr_language_for_user_type(user_type, direction)
    initial_prompt = (
        ""
        if language == "en"
        else "देवनागरी नेपाली। दाम, मूल्य, छुट, साइज, संख्या, रुपियाँ, चाहिन्छ, छैन, कति, कति पर्छ, सस्तो, महँगो"
    )

    try:
        segments, _ = model.transcribe(
            io.BytesIO(audio_bytes),
            language=language,
            beam_size=8,
            task="transcribe",
            initial_prompt=initial_prompt or None,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        raise RuntimeError(f"ASR transcription failed: {error}") from error

    return {
        "text": text,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "backend": settings.asr_backend,
    }
