from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_XET", "1")


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


@dataclass(frozen=True, slots=True)
class Settings:
    asr_backend: str = "faster_whisper"
    whisper_model_size: str = "large-v3"
    translation_backend: str = "echo_nep"
    translation_model_name: str = "SarjakBhandari-230383/EchoNep"
    hf_token: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    trained_model_dir: str = str(PROJECT_DIR.parent / "Training" / "checkpoints" / "translation_model")
    trained_model_base_name: str = "facebook/nllb-200-distilled-600M"
    local_model_dir: str = str(PROJECT_DIR / "models")
    tts_backend: str = "pyttsx3"
    tts_piper_voice_repo_path: str = "ne/ne_NP/chitwan/medium/ne_NP-chitwan-medium"
    host: str = "0.0.0.0"
    port: int = 8000
    phrasebook_path: str = str(BASE_DIR / "data" / "phrasebook.json")


settings = Settings()
