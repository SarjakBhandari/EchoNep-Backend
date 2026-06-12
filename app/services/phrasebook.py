from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.config import settings


@lru_cache(maxsize=1)
def load_phrasebook() -> dict[str, dict[str, str]]:
    path = Path(settings.phrasebook_path)
    if not path.exists():
        return {"en_np": {}, "np_en": {}}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "en_np": {key.lower(): value for key, value in data.get("en_np", {}).items()},
        "np_en": {key.lower(): value for key, value in data.get("np_en", {}).items()},
    }


def phrasebook_translate(text: str, direction: str) -> str | None:
    normalized = " ".join(text.strip().split()).lower()
    phrasebook = load_phrasebook()
    return phrasebook.get(direction, {}).get(normalized)
