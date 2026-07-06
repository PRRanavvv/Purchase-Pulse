from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    score: float
    text: str


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_markdown(text: str, chunk_size: int = 900) -> list[str]:
    sections = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        while len(section) > chunk_size:
            cut = section.rfind(". ", 0, chunk_size)
            if cut < 250:
                cut = chunk_size
            chunks.append(section[: cut + 1].strip())
            section = section[cut + 1 :].strip()
        if section:
            chunks.append(section)
    return chunks


class LocalRagAssistant:
    """Tiny local retriever for v2.

    This is intentionally dependency-light. It retrieves grounding text from
    markdown notes and composes a concise answer. Later, these retrieved chunks
    can be passed into an LLM for a fuller RAG answer.
    """

    def __init__(self, knowledge_dir: str | Path):
        self.knowledge_dir = Path(knowledge_dir)
        self.sources: list[str] = []
        self.chunks: list[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None
        self._load()

    def _load(self) -> None:
        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for chunk in _split_markdown(text):
                self.sources.append(path.name)
                self.chunks.append(_clean_text(chunk))

        if self.chunks:
            self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not query.strip() or self.matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            results.append(
                RetrievedChunk(
                    source=self.sources[idx],
                    score=float(scores[idx]),
                    text=self.chunks[idx],
                )
            )
        return results

    def answer(self, query: str, top_k: int = 4) -> str:
        query = query.strip()
        if not query:
            return "Ask about E20 risk, dataset strategy, buyer checks, or pricing logic."

        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return "I could not find a strong match in the local knowledge base yet. Add a note to `knowledge/` and ask again."

        answer_lines = ["### Grounded answer", ""]
        for chunk in retrieved[:3]:
            preview = chunk.text
            sentences = re.split(r"(?<=[.!?])\s+", preview)
            compact = " ".join(sentences[:3]).strip()
            answer_lines.append(f"- {compact}")

        answer_lines.extend(["", "### Suggested next action", _next_action(query), "", "### Retrieved sources"])
        for chunk in retrieved:
            answer_lines.append(f"- `{chunk.source}` (score {chunk.score:.2f})")
        return "\n".join(answer_lines)


def _next_action(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["dataset", "data", "scrape", "collect"]):
        return "Start with a clean listing schema, collect city-specific comps, and keep source URL plus collection date for every row."
    if any(word in q for word in ["e20", "fuel", "mileage", "ethanol"]):
        return "Treat E20 as a rule-and-evidence layer first. Use model year, fuel type, BS norm, and service history before making a buyer recommendation."
    if any(word in q for word in ["price", "negotiate", "overpriced", "fair"]):
        return "Compare against local comps before judging price. The model should show a fair band, not one overconfident number."
    return "Turn this into a checklist item or data field so the product gets better every time a buyer uses it."
