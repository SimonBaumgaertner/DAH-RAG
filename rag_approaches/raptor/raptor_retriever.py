from __future__ import annotations

from typing import Dict, List, Optional

from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk, Retriever
from common.logging.run_logger import RunLogger

from .raptor_src.tree_retriever import TreeRetriever, TreeRetrieverConfig
from .raptor_src.tree_structures import Tree


class RaptorRetriever(Retriever):
    def __init__(
        self,
        *,
        log: RunLogger,
        config: TreeRetrieverConfig,
        default_top_k: int = 8,
    ) -> None:
        self.log = log
        self._config = config
        self._default_top_k = max(1, default_top_k)
        self._tree: Optional[Tree] = None
        self._tree_retriever: Optional[TreeRetriever] = None
        self._leaf_mapping: Dict[int, Chunk] = {}

    def update_tree(self, tree: Tree, leaf_mapping: Dict[int, Chunk]) -> None:
        self._tree = tree
        self._leaf_mapping = leaf_mapping
        self._tree_retriever = TreeRetriever(self._config, tree)
        self.log.info("🔄 RAPTOR retriever updated with %d leaves", len(leaf_mapping))

    def retrieve(self, question: str, k: int = 5, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        if self._tree_retriever is None or self._tree is None:
            self.log.warning("⚠️ RAPTOR tree not initialised; returning no chunks.")
            return []

        top_k = max(1, k if k is not None else self._default_top_k)
        context, layer_info = self._tree_retriever.retrieve(
            question,
            start_layer=0,
            num_layers=1,
            top_k=top_k,
            collapse_tree=True,
            return_layer_information=True,
        )

        seen_indices: set[int] = set()
        results: List[Chunk] = []
        for entry in layer_info:
            node_index = entry.get("node_index")
            if node_index is None or node_index in seen_indices:
                continue
            chunk = self._leaf_mapping.get(node_index)
            if not chunk:
                continue
            seen_indices.add(node_index)
            metadata = dict(chunk.metadata) if chunk.metadata else {}
            results.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=None,
                    doc_id=chunk.doc_id,
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break

        if not results and context:
            synthetic_id = f"raptor-context-{abs(hash(question)) & 0xFFFFFFFF:x}"
            synthetic_chunk = Chunk(
                chunk_id=synthetic_id,
                text=context,
                score=None,
                doc_id=None,
                metadata={"note": "Generated RAPTOR context"},
            )
            results.append(synthetic_chunk)

        self.log.info("🔍 RAPTOR retrieved %d chunk(s) for query", len(results))
        return results

