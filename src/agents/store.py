from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self) -> None:
        self._texts: list[str] = []
        self._embeddings: list[list[float]] = []
        self._metadata: list[dict] = []

    def add(self, text: str, embedding: list[float], metadata: dict | None = None) -> None:
        self._texts.append(text)
        self._embeddings.append(embedding)
        self._metadata.append(metadata or {})

    def add_documents(self, texts: list[str], embed_fn, metadata: dict | None = None) -> None:
        embeddings = [embed_fn(text) for text in texts]
        for text, embedding in zip(texts, embeddings):
            self.add(text, embedding, metadata)

    def query(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        if not self._embeddings:
            return []
        scores = self._cosine_sim(embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in top_indices
        ]

    def search(self, query: str, embed_fn, top_k: int = 3) -> list[dict]:
        query_embedding = embed_fn(query)
        return self.query(query_embedding, top_k)

    def _cosine_sim(self, query: list[float]) -> np.ndarray:
        matrix = np.array(self._embeddings)
        q = np.array(query)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1, norms)
        return matrix @ q / norms
