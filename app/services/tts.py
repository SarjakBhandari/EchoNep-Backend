from __future__ import annotations

import base64
import io
import math
import os
import tempfile
import time
import wave
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

from app.config import settings


def _silence_wav_bytes(duration_seconds: float = 0.4, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def _beep_wav_bytes(duration_seconds: float = 0.45, sample_rate: int = 16000, frequency_hz: int = 660) -> bytes:
    buffer = io.BytesIO()
    frame_count = int(duration_seconds * sample_rate)
    amplitude = 8000
    frames = bytearray()
    for i in range(frame_count):
        value = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * i / sample_rate))
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


def _offline_pyttsx3_bytes(text: str) -> bytes | None:
    try:
        import pyttsx3
    except Exception:
        return None

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        engine = pyttsx3.init()
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        with open(temp_path, "rb") as handle:
            return handle.read()
    except Exception:
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _piper_voice_base_url() -> str:
    return "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _piper_voice_paths() -> tuple[Path, Path]:
    voice_dir = Path(settings.local_model_dir) / "piper" / settings.tts_piper_voice_repo_path
    model_path = voice_dir.with_suffix(".onnx")
    config_path = voice_dir.with_suffix(".onnx.json")
    return model_path, config_path


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    with urlopen(url) as response, destination.open("wb") as handle:
        handle.write(response.read())


def _ensure_piper_voice_files() -> tuple[Path, Path]:
    model_path, config_path = _piper_voice_paths()
    repo_path = settings.tts_piper_voice_repo_path
    _download_file(f"{_piper_voice_base_url()}/{repo_path}.onnx?download=true", model_path)
    _download_file(f"{_piper_voice_base_url()}/{repo_path}.onnx.json?download=true", config_path)
    return model_path, config_path


@lru_cache(maxsize=1)
def _load_piper_voice():
    from piper import PiperVoice

    model_path, config_path = _ensure_piper_voice_files()
    return PiperVoice.load(model_path, config_path=config_path)


def _piper_nepali_bytes(text: str) -> bytes | None:
    try:
        voice = _load_piper_voice()
    except Exception:
        return None

    buffer = io.BytesIO()
    try:
        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()
    except Exception:
        return None


def text_to_speech(text: str, language: str) -> dict:
    started_at = time.perf_counter()
    selected_backend = settings.tts_backend
    if language == "ne":
        piper_bytes = _piper_nepali_bytes(text)
        if piper_bytes:
            return {
                "audio_b64": base64.b64encode(piper_bytes).decode("utf-8"),
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "backend": "piper_ne_np_chitwan_medium",
            }

    if settings.tts_backend == "pyttsx3":
        pyttsx3_bytes = _offline_pyttsx3_bytes(text)
        if pyttsx3_bytes:
            audio_bytes = pyttsx3_bytes
        else:
            try:
                from gtts import gTTS

                voice_language = "en" if language == "en" else "ne"
                buffer = io.BytesIO()
                gTTS(text=text, lang=voice_language, slow=False).write_to_fp(buffer)
                audio_bytes = buffer.getvalue()
                selected_backend = "gtts_fallback"
            except Exception:
                audio_bytes = _beep_wav_bytes()
                selected_backend = "beep_fallback"
    elif settings.tts_backend == "gtts":
        try:
            from gtts import gTTS

            voice_language = "en" if language == "en" else "ne"
            buffer = io.BytesIO()
            gTTS(text=text, lang=voice_language, slow=False).write_to_fp(buffer)
            audio_bytes = buffer.getvalue()
        except Exception:
            pyttsx3_bytes = _offline_pyttsx3_bytes(text)
            if pyttsx3_bytes:
                audio_bytes = pyttsx3_bytes
                selected_backend = "pyttsx3_fallback"
            else:
                audio_bytes = _beep_wav_bytes()
                selected_backend = "beep_fallback"
    else:
        audio_bytes = _beep_wav_bytes()
        selected_backend = "beep_fallback"

    return {
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "backend": selected_backend,
    }
