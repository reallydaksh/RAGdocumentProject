"""Split pages into overlapping chunks while keeping page numbers for citations."""

import re
from dataclasses import dataclass

from .loader import Page


@dataclass
class Chunk:
    id: int
    source: str
    page: int
    text: str


def _split_units(text: str) -> list[str]:
    """Break text into paragraph/sentence-sized units so chunks end on natural boundaries."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
    return units


def _hard_split(unit: str, size: int) -> list[str]:
    """Fallback for a single sentence longer than the chunk size."""
    return [unit[i : i + size] for i in range(0, len(unit), size)]


def chunk_pages(pages: list[Page], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        units: list[str] = []
        for u in _split_units(page.text):
            units.extend(_hard_split(u, chunk_size) if len(u) > chunk_size else [u])

        current: list[str] = []
        length = 0
        for unit in units:
            if current and length + len(unit) + 1 > chunk_size:
                chunks.append(Chunk(len(chunks), page.source, page.page, " ".join(current)))
                # carry trailing units forward as overlap without exceeding chunk_size
                carried: list[str] = []
                carried_len = 0
                for prev in reversed(current):
                    if carried_len + len(prev) + 1 > overlap or carried_len + len(prev) + 1 + len(unit) + 1 > chunk_size:
                        break
                    carried.insert(0, prev)
                    carried_len += len(prev) + 1
                current, length = carried, carried_len
            current.append(unit)
            length += len(unit) + 1
        if current:
            chunks.append(Chunk(len(chunks), page.source, page.page, " ".join(current)))
    return chunks
