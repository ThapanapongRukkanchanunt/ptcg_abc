from __future__ import annotations

import re
from hashlib import sha256

from ptcg_abc.models import CardLine


_SPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def normalize_card_name(name: str) -> str:
    return clean_text(name).casefold()


def slugify(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold())
    slug = slug.strip("-")
    return (slug[:max_length].strip("-") or "item")


def deck_fingerprint(cards: list[CardLine]) -> str:
    normalized = sorted((normalize_card_name(card.name), card.count) for card in cards)
    body = "\n".join(f"{count} {name}" for name, count in normalized)
    return sha256(body.encode("utf-8")).hexdigest()
