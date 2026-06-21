from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


CARD_IMAGE_PAGE_OFFSET = 39


class CardArtCache:
    """Extract and cache card art from the Kaggle Card ID PDF on demand."""

    def __init__(self, *, pdf_path: Path, cache_dir: Path) -> None:
        self.pdf_path = pdf_path
        self.cache_dir = cache_dir
        self._reader: Any | None = None

    def get(self, card_id: int | None) -> Path | None:
        if card_id is None or card_id <= 0:
            return None
        output_path = self.cache_dir / f"{card_id:04d}.png"
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        if not self.pdf_path.exists():
            return None

        reader = self._pdf_reader()
        page_index = card_id + CARD_IMAGE_PAGE_OFFSET - 1
        if page_index < 0 or page_index >= len(reader.pages):
            return None
        images = list(reader.pages[page_index].images)
        if not images:
            return None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        image = images[0].image.convert("RGBA")
        image.save(output_path)
        return output_path

    def _pdf_reader(self) -> Any:
        if self._reader is not None:
            return self._reader
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required to extract card art from the Card ID PDF. "
                "Install the rl extra with `pip install -e .[rl]`."
            ) from exc
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        self._reader = PdfReader(str(self.pdf_path))
        return self._reader
