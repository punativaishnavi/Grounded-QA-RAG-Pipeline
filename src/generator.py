"""Answer generation with pluggable backends.

- extractive: no LLM needed; returns the top chunks as a grounded answer.
- ollama: local LLM via the Ollama HTTP API.
- openai: OpenAI chat completions (needs OPENAI_API_KEY).
"""

import logging
import os

import requests

from .documents import Chunk

log = logging.getLogger("rag")


class Generator:
    def __init__(self, cfg: dict):
        self.backend = cfg.get("backend", "extractive")
        self.model = cfg.get("model", "")
        self.ollama_url = cfg.get("ollama_url", "http://localhost:11434")
        self.max_chunks = cfg.get("max_chunks_in_prompt", 4)
        self.template = cfg.get("prompt_template", "Context:\n{context}\n\nQ: {question}\nA:")

    def generate(self, question: str, chunks: list[tuple[Chunk, float]]) -> str:
        chunks = chunks[: self.max_chunks]
        if self.backend == "extractive":
            return self._extractive(question, chunks)
        context = self._format_context(chunks)
        prompt = self.template.format(context=context, question=question)
        if self.backend == "ollama":
            return self._ollama(prompt)
        if self.backend == "openai":
            return self._openai(prompt)
        raise ValueError(f"Unknown generate backend: {self.backend!r}")

    @staticmethod
    def _format_context(chunks: list[tuple[Chunk, float]]) -> str:
        return "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c, _ in chunks)

    def _extractive(self, question: str, chunks: list[tuple[Chunk, float]]) -> str:
        if not chunks:
            return "I don't know based on the provided documents."
        lines = []
        for c, score in chunks:
            snippet = c.text if len(c.text) <= 400 else c.text[:400].rsplit(" ", 1)[0] + "…"
            lines.append(f"[chunk {c.chunk_id} from {c.source} | score {score:.2f}]\n{snippet}")
        return "\n\n".join(lines)

    def _ollama(self, prompt: str) -> str:
        resp = requests.post(f"{self.ollama_url}/api/generate",
                             json={"model": self.model, "prompt": prompt,
                                   "stream": False},
                             timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _openai(self, prompt: str) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                             headers={"Authorization": f"Bearer {key}"},
                             json={"model": self.model or "gpt-4o-mini",
                                   "messages": [{"role": "user", "content": prompt}],
                                   "temperature": 0},
                             timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
