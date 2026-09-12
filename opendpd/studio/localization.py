"""Translate application prose at presentation time, without changing stored scientific records.

The JSON catalogues are shared with the frontend. Unknown source text, code,
identifiers and numeric captures are preserved, never guessed or recomputed.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from opendpd.schemas.settings import UI_LANGUAGES


@lru_cache(maxsize=9)
def catalogue(language: str) -> dict[str, str]:
    code = language if language in UI_LANGUAGES else "en"
    return json.loads((Path(__file__).parent / "locales" / f"{code}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _templates():
    result = []
    for source in sorted(catalogue("en"), key=len, reverse=True):
        names = re.findall(r"\{(\w+)\}", source)
        if names:
            pattern = "([\\s\\S]+?)".join(re.escape(part) for part in re.split(r"\{\w+\}", source))
            result.append((source, names, re.compile("^" + pattern + "$")))
    return result


def localize(value: Optional[str], language: str = "en") -> str:
    if not value:
        return ""
    source = value.strip()
    table = catalogue(language)
    if source in table:
        return table[source]
    for template, names, pattern in _templates():
        match = pattern.fullmatch(source)
        if match:
            values = dict(zip(names, match.groups()))
            return re.sub(r"\{(\w+)\}", lambda m: values.get(m[1], m[0]), table.get(template, template))
    return value


def language_tag(language: str) -> str:
    return "zh-CN" if language == "zh" else language if language in UI_LANGUAGES else "en"
