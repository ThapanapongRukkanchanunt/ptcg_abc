from __future__ import annotations

import re
import unicodedata
from hashlib import sha256

from ptcg_abc.models import CardLine


_SPACE_RE = re.compile(r"\s+")
_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â€™", "â€œ", "â€\x9d")
_BASIC_ENERGY_ALIASES = {
    "darkness energy": "basic {d} energy",
    "fighting energy": "basic {f} energy",
    "fire energy": "basic {r} energy",
    "grass energy": "basic {g} energy",
    "lightning energy": "basic {l} energy",
    "metal energy": "basic {m} energy",
    "psychic energy": "basic {p} energy",
    "water energy": "basic {w} energy",
}
_EXACT_ALIASES = {
    "growing grass energy": "grow grass energy",
    "rocky fighting energy": "rock fighting energy",
    "telepathic psychic energy": "telepath psychic energy",
}


def clean_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def repair_mojibake(value: str) -> str:
    if not any(marker in value for marker in _MOJIBAKE_MARKERS):
        return value
    try:
        repaired = value.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return value
    return repaired


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_card_name(name: str) -> str:
    cleaned = repair_mojibake(clean_text(name))
    cleaned = cleaned.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "`": "'",
                "´": "'",
                "“": '"',
                "”": '"',
                "–": "-",
                "—": "-",
                "♢": "prism star",
            }
        )
    )
    normalized = clean_text(_strip_accents(cleaned)).casefold()
    return _EXACT_ALIASES.get(normalized, _BASIC_ENERGY_ALIASES.get(normalized, normalized))


def slugify(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold())
    slug = slug.strip("-")
    return (slug[:max_length].strip("-") or "item")


def deck_fingerprint(cards: list[CardLine]) -> str:
    normalized = sorted((normalize_card_name(card.name), card.count) for card in cards)
    body = "\n".join(f"{count} {name}" for name, count in normalized)
    return sha256(body.encode("utf-8")).hexdigest()
