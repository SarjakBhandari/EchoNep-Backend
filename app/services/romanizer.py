from __future__ import annotations

import re

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except Exception:  # pragma: no cover - optional dependency
    transliterate = None
    sanscript = None


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")


def _romanize_chunk(chunk: str) -> str:
    if transliterate is None or sanscript is None:
        return chunk
    try:
        return transliterate(chunk, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        return chunk


def romanize(text: str) -> str:
    if transliterate is None or sanscript is None:
        return text
    return _DEVANAGARI_RE.sub(lambda match: _romanize_chunk(match.group(0)), text)
