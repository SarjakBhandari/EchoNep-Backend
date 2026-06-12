from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    direction: str = Field(pattern=r"^(en_np|np_en)$")


class AsrRequest(BaseModel):
    audio_b64: str = Field(min_length=1)
    direction: str = Field(pattern=r"^(en_np|np_en)$")
    user_type: str = Field(default="tourist", pattern=r"^(tourist|trader)$")


class PipelineRequest(BaseModel):
    direction: str = Field(pattern=r"^(en_np|np_en)$")
    user_type: str = Field(default="tourist", pattern=r"^(tourist|trader)$")
    text: Optional[str] = None
    audio_b64: Optional[str] = None


class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str
    romanized_text: str = ""
    audio_b64: str = ""
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    backend: Dict[str, Any] = Field(default_factory=dict)


class AsrResponse(BaseModel):
    text: str
    latency_ms: float
    backend: str


class HealthResponse(BaseModel):
    status: str
    backend: Dict[str, Any]
