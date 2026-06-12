from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import AsrRequest, AsrResponse, PipelineRequest, TranslateRequest, TranslationResponse
from app.services.asr import transcribe_audio
from app.services.romanizer import romanize
from app.services.translator import translate_text
from app.services.tts import text_to_speech


router = APIRouter()


def _tts_language_for_text(text: str) -> str:
    return "ne" if any("\u0900" <= character <= "\u097F" for character in text) else "en"


@router.post("/translate", response_model=TranslationResponse)
def translate_endpoint(request: TranslateRequest) -> TranslationResponse:
    translation = translate_text(request.text, request.direction)
    translated_text = translation["translated_text"]
    romanized_text = romanize(translated_text) if request.direction == "en_np" else ""
    tts_text = translated_text
    tts = text_to_speech(tts_text, _tts_language_for_text(tts_text))
    return TranslationResponse(
        source_text=request.text,
        translated_text=translated_text,
        romanized_text=romanized_text,
        audio_b64=tts["audio_b64"],
        latency_ms={"translate": translation["latency_ms"], "tts": tts["latency_ms"]},
        backend={"translate": translation["backend"], "tts": tts["backend"]},
    )


@router.post("/asr", response_model=AsrResponse)
def asr_endpoint(request: AsrRequest) -> AsrResponse:
    asr_result = transcribe_audio(request.audio_b64, request.direction, request.user_type)
    return AsrResponse(
        text=asr_result["text"],
        latency_ms=asr_result["latency_ms"],
        backend=asr_result["backend"],
    )


@router.post("/pipeline", response_model=TranslationResponse)
def pipeline_endpoint(request: PipelineRequest) -> TranslationResponse:
    if not request.text and not request.audio_b64:
        raise HTTPException(status_code=422, detail="Provide either text or audio_b64")

    source_text = request.text or ""
    asr_result = {"latency_ms": 0.0, "backend": "skipped"}

    if request.audio_b64:
        asr_result = transcribe_audio(request.audio_b64, request.direction, request.user_type)
        source_text = asr_result["text"]

    if not source_text.strip():
        raise HTTPException(status_code=422, detail="No source text detected")

    translation = translate_text(source_text, request.direction)
    translated_text = translation["translated_text"]
    romanized_text = romanize(translated_text) if request.direction == "en_np" else ""
    tts_text = translated_text
    tts = text_to_speech(tts_text, _tts_language_for_text(tts_text))

    return TranslationResponse(
        source_text=source_text,
        translated_text=translated_text,
        romanized_text=romanized_text,
        audio_b64=tts["audio_b64"],
        latency_ms={"asr": asr_result["latency_ms"], "translate": translation["latency_ms"], "tts": tts["latency_ms"]},
        backend={"asr": asr_result["backend"], "translate": translation["backend"], "tts": tts["backend"]},
    )
