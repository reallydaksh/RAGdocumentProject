"""Turn uploaded files (PDF, TXT, MD) into page-level text records."""

import io
from dataclasses import dataclass

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


@dataclass
class Page:
    source: str  # file name
    page: int  # 1-based page number (always 1 for text files)
    text: str


def load_file(name: str, data: bytes) -> list[Page]:
    """Load a single file from raw bytes. Returns one Page per PDF page."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, pdf_page in enumerate(reader.pages, start=1):
            text = (pdf_page.extract_text() or "").strip()
            if text:
                pages.append(Page(source=name, page=i, text=text))
        return pages
    if lower.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore").strip()
        return [Page(source=name, page=1, text=text)] if text else []
    raise ValueError(f"Unsupported file type: {name}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
