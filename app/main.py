from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers.pipeline import router
from app.schemas import HealthResponse
from app.services.asr import preload_asr_model
from app.services.translator import preload_translation_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr_status = preload_asr_model()
    translation_status = preload_translation_models()

    if asr_status == "unavailable":
        raise RuntimeError(f"ASR model could not be loaded at startup: {settings.asr_backend}")

    primary_translation_model = settings.translation_model_name
    if translation_status.get(primary_translation_model) != "loaded":
        detail = translation_status.get(primary_translation_model, "unavailable")
        raise RuntimeError(
            f"Primary translation model could not be loaded at startup: {primary_translation_model} ({detail})"
        )
    yield


app = FastAPI(title="ECHONEP API", version="1.0.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        backend={
            "asr_backend": settings.asr_backend,
            "translation_backend": settings.translation_backend,
            "tts_backend": settings.tts_backend,
            "trained_model_dir": settings.trained_model_dir,
        },
    )
