from abc import abstractmethod, ABC
from typing import List

import torch
from transformers import AutoModel

from common.data_classes.rag_system import Chunk
from common.logging.run_logger import RunLogger


class RerankingStrategy(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        ...


class NoRerank(RerankingStrategy):
    """Identity strategy – returns the list unchanged."""

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:  # noqa: D401
        return chunks


class JinaReranker(RerankingStrategy):
    """Rerank using Jina Reranker v3 (600M).

    Uses the model's built-in .rerank() method which handles chunking
    and list-wise scoring internally.
    """

    def __init__(
        self,
        *,
        log: RunLogger,
        model_name: str = "jinaai/jina-reranker-v3",
        device: str | None = None,
    ) -> None:
        self.log = log
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info(f"🚀 Initializing JinaReranker ({model_name}) on {self.device}")

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 25) -> List[Chunk]:
        if not chunks:
            return []

        # Only rerank the first top_k items
        to_rerank = chunks[:top_k]
        remaining = chunks[top_k:]

        if not to_rerank:
            return chunks

        doc_texts = [c.text for c in to_rerank]

        results = self.model.rerank(query, doc_texts)

        # Map index within to_rerank to its relevance score
        score_map = {res["index"]: res["relevance_score"] for res in results}

        for i, chunk in enumerate(to_rerank):
            chunk.score = float(score_map.get(i, 0.0))

        # Sort only the subset that was reranked
        reranked = sorted(to_rerank, key=lambda c: c.score or 0.0, reverse=True)

        return reranked + remaining