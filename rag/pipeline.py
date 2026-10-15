"""Ties loading, chunking, retrieval and generation together."""

from collections.abc import Iterator

from .chunker import chunk_pages
from .config import Settings
from .loader import load_file
from .providers import ChatModel, Embedder, build_providers
from .retriever import HybridRetriever, SearchResult

SYSTEM_PROMPT = """You are AskMyDocs, an assistant that answers questions strictly from the provided document excerpts.

Rules:
- Use ONLY the numbered context excerpts. Do not use outside knowledge.
- Cite every factual claim with the excerpt number in square brackets, e.g. [1] or [2][4].
- If the excerpts do not contain the answer, say: "I couldn't find that in the uploaded documents." and stop.
- Be concise. Use bullet points when listing several items."""


def build_prompt(question: str, results: list[SearchResult], history: list[dict] | None = None) -> str:
    context = "\n\n".join(
        f"[{n}] (source: {r.chunk.source}, page {r.chunk.page})\n{r.chunk.text}"
        for n, r in enumerate(results, start=1)
    )
    convo = ""
    if history:
        recent = history[-6:]  # last 3 exchanges, enough for follow-up questions
        convo = "Conversation so far:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in recent) + "\n\n"
    return f"{convo}Context excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None, chat: ChatModel | None = None):
        self.settings = settings or Settings()
        if embedder is None or chat is None:
            embedder, chat = build_providers(self.settings)
        self.embedder = embedder
        self.chat = chat
        self.retriever: HybridRetriever | None = None
        self.stats: dict = {}

    def index_files(self, files: list[tuple[str, bytes]]) -> dict:
        pages = [p for name, data in files for p in load_file(name, data)]
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        self.retriever = HybridRetriever(chunks, self.embedder)
        self.stats = {"files": len(files), "pages": len(pages), "chunks": len(chunks)}
        return self.stats

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        if self.retriever is None:
            raise RuntimeError("No documents indexed yet.")
        return self.retriever.search(question, k or self.settings.top_k)

    def answer_stream(self, question: str, results: list[SearchResult], history: list[dict] | None = None) -> Iterator[str]:
        return self.chat.stream(SYSTEM_PROMPT, build_prompt(question, results, history))
