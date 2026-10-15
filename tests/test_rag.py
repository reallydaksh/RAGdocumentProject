"""Offline tests — use a fake embedder/chat model so no API key is needed.

Run:  pytest -q
"""

import hashlib
import re

import numpy as np
import pytest

from rag.chunker import chunk_pages
from rag.config import Settings
from rag.loader import Page, load_file
from rag.pipeline import RAGPipeline, build_prompt


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-size vector — deterministic and dependency-free."""

    DIM = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype="float32")
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1
        return v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeChat:
    def stream(self, system, prompt):
        yield "stub answer [1]"


DOC = """Leave Policy

Employees receive 24 days of paid annual leave per calendar year. Unused leave can be carried forward up to 10 days.

Notice Period

The notice period for resignation is 60 days for permanent staff and 30 days for interns.

Remote Work

Employees may work remotely up to two days per week with manager approval. The VPN must be used at all times."""


@pytest.fixture
def pipeline():
    p = RAGPipeline(Settings(chunk_size=200, chunk_overlap=40), embedder=FakeEmbedder(), chat=FakeChat())
    p.index_files([("handbook.txt", DOC.encode())])
    return p


def test_chunks_respect_size_and_keep_pages():
    pages = [Page("a.pdf", 3, DOC)]
    chunks = chunk_pages(pages, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.page == 3 and c.source == "a.pdf" for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([Page("a", 1, DOC)], chunk_size=100, overlap=100)


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        load_file("image.png", b"...")


def test_retrieval_finds_relevant_chunk(pipeline):
    results = pipeline.retrieve("What is the notice period for resignation?", k=2)
    assert "60 days" in results[0].chunk.text


def test_keyword_search_catches_exact_terms(pipeline):
    results = pipeline.retrieve("VPN", k=1)
    assert "VPN" in results[0].chunk.text
    assert results[0].keyword_rank == 1


def test_prompt_numbers_and_cites_sources(pipeline):
    results = pipeline.retrieve("annual leave", k=2)
    prompt = build_prompt("annual leave?", results)
    assert "[1] (source: handbook.txt, page 1)" in prompt
    assert "[2]" in prompt


def test_answer_stream(pipeline):
    results = pipeline.retrieve("remote work", k=2)
    assert "".join(pipeline.answer_stream("remote work?", results)) == "stub answer [1]"
