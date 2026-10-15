"""Gemini embedding and chat model wrappers."""

from collections.abc import Iterator
from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class ChatModel(Protocol):
    def stream(self, system: str, prompt: str) -> Iterator[str]: ...


# ---------------------------------------------------------------- Gemini
class GeminiEmbedder:
    BATCH = 100

    def __init__(self, api_key: str, model: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts[i : i + self.BATCH],
                config=types.EmbedContentConfig(task_type=task),
            )
            vectors.extend(e.values for e in resp.embeddings)
        return np.asarray(vectors, dtype="float32")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiChat:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        from google.genai import types

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
        ):
            if chunk.text:
                yield chunk.text


def build_providers(settings: Settings) -> tuple[Embedder, ChatModel]:
    settings.validate()
    return (
        GeminiEmbedder(settings.gemini_api_key, settings.gemini_embed_model),
        GeminiChat(settings.gemini_api_key, settings.gemini_chat_model),
    )
