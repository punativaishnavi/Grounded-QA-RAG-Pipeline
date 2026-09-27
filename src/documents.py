"""Document loading and chunking."""

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger("rag")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(docs_dir: str, supported_exts: list[str]) -> list[tuple[str, str]]:
    """Return [(filename, text)] for every supported file in docs_dir."""
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported_exts:
            continue
        path = os.path.join(docs_dir, fname)
        try:
            text = _read_pdf(path) if ext == ".pdf" else _read_txt(path)
        except Exception as exc:  # noqa: BLE001 - skip unreadable files loudly
            log.warning("Skipping %s: %s", fname, exc)
            continue
        text = " ".join(text.split())  # collapse whitespace
        if text:
            docs.append((fname, text))
    log.info("Loaded %d document(s) from %s", len(docs), docs_dir)
    return docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks, start = [], 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def chunk_documents(docs: list[tuple[str, str]], chunk_size: int,
                    chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for fname, text in docs:
        for i, piece in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(text=piece, source=fname, chunk_id=i))
    log.info("Created %d chunk(s) from %d document(s)", len(chunks), len(docs))
    return chunks
