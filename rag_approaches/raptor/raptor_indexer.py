from __future__ import annotations

import copy
from typing import Callable, Dict, List

from common.data_classes.documents import Document
from common.data_classes.evaluation import EntryType
from common.data_classes.rag_system import Chunk, Indexer
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ChunkingStrategy

from .raptor_src.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .raptor_src.tree_structures import Tree


class PreChunkedClusterTreeBuilder(ClusterTreeBuilder):
    """Cluster tree builder that operates on pre-computed chunks."""

    def __init__(self, config: ClusterTreeConfig) -> None:
        super().__init__(config)

    def build_from_chunks(self, chunk_texts: List[str]) -> Tree:
        if not chunk_texts:
            raise ValueError("chunk_texts must not be empty")

        leaf_nodes = {}
        for index, text in enumerate(chunk_texts):
            _, node = self.create_node(index, text)
            leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}
        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(
            current_level_nodes=leaf_nodes,
            all_tree_nodes=all_nodes,
            layer_to_nodes=layer_to_nodes,
            use_multithreading=True,
        )

        return Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)


class RaptorIndexer(Indexer):
    def __init__(
        self,
        *,
        builder: PreChunkedClusterTreeBuilder,
        chunker: ChunkingStrategy,
        log: RunLogger,
        chunk_store: Dict[str, Chunk],
        on_tree_update: Callable[[Tree, Dict[int, Chunk]], None] | None = None,
    ) -> None:
        self._builder = builder
        self._chunker = chunker
        self.log = log
        self._chunk_store = chunk_store
        self._on_tree_update = on_tree_update
        self._chunks: List[Chunk] = []

    def index(self, document: Document) -> None:
        self.log.info("🔪 Chunking doc %s …", document.id)
        chunks = self._chunker.chunk(document)
        if not chunks:
            self.log.warning("📭 No chunks produced for doc %s; skipping RAPTOR indexing.", document.id)
            return

        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata.setdefault("source_document_id", document.id)
            self._chunk_store[chunk.chunk_id] = chunk

        self._chunks.extend(chunks)

        self.log.track(
            entry_type=EntryType.CHUNK_COUNT_TRACK.value,
            identifier=document.id,
            value=str(len(chunks)),
        )

        # Tree building is deferred to finalize_indexing() to avoid rebuilding
        # the tree from all chunks after every document, which causes performance degradation

    def finalize_indexing(self) -> None:
        """Build the RAPTOR tree from all accumulated chunks. Called after all documents are indexed."""
        if not self._chunks:
            self.log.warning("⚠️ No chunks available to build RAPTOR tree")
            return

        self.log.info("🌳 Building RAPTOR tree from %d accumulated chunks...", len(self._chunks))
        tree, leaf_mapping = self._build_tree()

        if self._on_tree_update:
            self._on_tree_update(tree, leaf_mapping)

    def _build_tree(self) -> tuple[Tree, Dict[int, Chunk]]:
        chunk_texts = [chunk.text for chunk in self._chunks]
        tree = self._builder.build_from_chunks(chunk_texts)
        leaf_mapping = {idx: chunk for idx, chunk in enumerate(self._chunks)}
        self.log.info("🌳 RAPTOR tree built with %d leaf nodes", len(leaf_mapping))
        return tree, leaf_mapping

