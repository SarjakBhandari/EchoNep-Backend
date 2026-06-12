from __future__ import annotations

import base64
import io
import tempfile
import time
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def load_asr_model():
    if settings.asr_backend == "faster_whisper":
        from faster_whisper import WhisperModel

        return WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")

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
    audio_bytes = base64.b64decode(audio_b64)
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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    segments, _ = model.transcribe(
        temp_path,
        language=language,
        beam_size=8,
        task="transcribe",
        initial_prompt=initial_prompt or None,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {
        "text": text,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "backend": settings.asr_backend,
    }
