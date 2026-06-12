from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import settings
from app.services.phrasebook import phrasebook_translate

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_XET", "1")

_MODEL_LOAD_ERRORS: dict[str, str] = {}


@lru_cache(maxsize=1)
def load_tokenizer_and_model(direction: str):
    if settings.translation_backend not in {"trained_model", "model", "hf_model", "echo_nep"}:
        return None, None

    for model_source in _candidate_model_sources(direction):
        tokenizer, model = _load_model_cached(model_source)
        if tokenizer is not None and model is not None:
            return tokenizer, model

    return None, None


@lru_cache(maxsize=None)
def _load_model_cached(model_source: str):
    if settings.translation_backend not in {"trained_model", "model", "hf_model", "echo_nep"}:
        return None, None

    try:
        cache_dir = settings.local_model_dir if not Path(model_source).exists() else None
        load_kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            use_fast=False,
            token=settings.hf_token if settings.hf_token else None,
            **load_kwargs,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_source,
            token=settings.hf_token if settings.hf_token else None,
            **load_kwargs,
        )
        model.eval()
        _MODEL_LOAD_ERRORS.pop(model_source, None)
        return tokenizer, model
    except Exception as error:
        _MODEL_LOAD_ERRORS[model_source] = str(error)
        return None, None


def _candidate_model_sources(direction: str) -> tuple[str, ...]:
    if settings.translation_model_name:
        return (settings.translation_model_name,)
    return ()


def _resolve_model_source(direction: str) -> str:
    for model_source in _candidate_model_sources(direction):
        return model_source
    return settings.translation_model_name or settings.trained_model_dir


def _normalize_direction(direction: str) -> str:
    return direction if direction in {"en_np", "np_en"} else "en_np"


def translate_text(text: str, direction: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    direction = _normalize_direction(direction)

    phrasebook_result = phrasebook_translate(text, direction)
    if phrasebook_result:
        return {
            "translated_text": phrasebook_result,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "backend": "phrasebook",
        }

    source_lang = "eng_Latn" if direction == "en_np" else "npi_Deva"
    target_lang = "npi_Deva" if direction == "en_np" else "eng_Latn"

    for model_source in _candidate_model_sources(direction):
        tokenizer, model = _load_model_cached(model_source)
        if tokenizer is None or model is None:
            continue

        tokenizer.src_lang = source_lang
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        # Safe forced BOS token handling for NLLB-200M
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        if forced_bos_token_id == tokenizer.unk_token_id:
            raise ValueError(f"Target language {target_lang} not recognized by tokenizer")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=128,
            )

        candidate_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {
            "translated_text": candidate_text,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "backend": model_source,
        }

    raise RuntimeError("Translation model unavailable or failed to produce output")


def preload_translation_models() -> dict[str, str]:
    if settings.translation_backend not in {"trained_model", "model", "hf_model", "echo_nep"}:
        return {"status": "disabled"}

    model_source = settings.translation_model_name
    tokenizer, model = _load_model_cached(model_source)
    if tokenizer is not None and model is not None:
        return {model_source: "loaded"}
    return {model_source: _MODEL_LOAD_ERRORS.get(model_source, "unavailable")}
