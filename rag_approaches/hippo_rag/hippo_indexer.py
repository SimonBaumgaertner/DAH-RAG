from __future__ import annotations

from datetime import date
from typing import Callable, Dict

from common.data_classes.documents import Document
from common.data_classes.evaluation import EntryType, LLMCallContext
from common.data_classes.rag_system import Chunk, Indexer
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ChunkingStrategy

import sys
from pathlib import Path

# Add the local HippoRAG src directory to the path
hippo_src_path = Path(__file__).parent / "HippoRAG" / "src"
sys.path.insert(0, str(hippo_src_path))

from hipporag.HippoRAG import HippoRAG as HippoRAGEngine
from hipporag.utils.misc_utils import compute_mdhash_id


def _safe_date(value: date | None) -> str | None:
    return value.isoformat() if value else None


class HippoRAGIndexer(Indexer):
    """Chunk documents with our strategy and feed them into HippoRAG."""

    def __init__(
        self,
        *,
        rag: HippoRAGEngine,
        chunker: ChunkingStrategy,
        log: RunLogger,
        chunk_store: Dict[str, Chunk],
        on_update: Callable[[], None] | None = None,
        batch_size: int = 3000,
    ) -> None:
        self.rag = rag
        self.chunker = chunker
        self.log = log
        self._chunk_store = chunk_store
        self._on_update = on_update
        self.batch_size = batch_size
        self._accumulated_chunks: list[str] = []

    def index(self, document: Document) -> None:
        self.log.info("🔪 Chunking doc %s …", document.id)
        chunks = self.chunker.chunk(document)
        if not chunks:
            self.log.warning("📭 No chunks produced for doc %s; skipping HippoRAG indexing.", document.id)
            return

        chunk_texts = []
        for chunk in chunks:
            hash_id = compute_mdhash_id(chunk.text, prefix="chunk-")
            metadata: Dict[str, str] = {
                "original_chunk_id": chunk.chunk_id
            }
            metadata.setdefault("source_document_id", document.id)
            if document.title:
                metadata.setdefault("document_title", document.title)
            if document.author:
                metadata.setdefault("document_author", document.author)
            pub_date = _safe_date(document.publication_date)
            if pub_date:
                metadata.setdefault("document_publication_date", pub_date)

            stored_chunk = Chunk(
                chunk_id=hash_id,
                text=chunk.text,
                doc_id=document.id,
                metadata=metadata,
            )
            self._chunk_store[hash_id] = stored_chunk
            chunk_texts.append(chunk.text)

        # Track chunk count for this document
        self.log.track(
            entry_type=EntryType.CHUNK_COUNT_TRACK.value,
            identifier=document.id,
            value=str(len(chunk_texts))
        )

        # Accumulate chunks for batch indexing
        self._accumulated_chunks.extend(chunk_texts)
        self.log.info("📦 Accumulated %d chunk(s) from doc %s (total: %d chunks)", 
                     len(chunk_texts), document.id, len(self._accumulated_chunks))
        
        if self._on_update:
            self._on_update()

    def finalize_indexing(self) -> None:
        """Build HippoRAG index from all accumulated chunks in batches."""
        if not self._accumulated_chunks:
            self.log.warning("⚠️ No chunks available to index in HippoRAG")
            return

        total_chunks = len(self._accumulated_chunks)
        self.log.info("🌳 Building HippoRAG index from %d accumulated chunks in batches of %d...", 
                     total_chunks, self.batch_size)
        
        # Process chunks in batches
        num_batches = (total_chunks + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_chunks)
            batch = self._accumulated_chunks[start_idx:end_idx]
            
            self.log.info("🚀 Processing batch %d/%d (%d chunks: %d-%d)", 
                         batch_idx + 1, num_batches, len(batch), start_idx, end_idx - 1)
            
            try:
                self.log.info("🔧 Starting HippoRAG indexing for batch %d/%d", batch_idx + 1, num_batches)
                
                # Set LLM context for indexing
                if hasattr(self.rag.llm_model, 'set_context'):
                    batch_id = f"hippo_batch_{batch_idx + 1}_of_{num_batches}"
                    self.rag.llm_model.set_context(LLMCallContext.INDEXING, identifier=batch_id)
                    
                self.rag.index(batch)
                self.log.info("✅ Completed HippoRAG indexing for batch %d/%d", batch_idx + 1, num_batches)
            except Exception as e:
                self.log.error("❌ Failed to index batch %d/%d: %s", batch_idx + 1, num_batches, e)
                raise
        
        self.log.info("🏁 HippoRAG indexing completed for all %d chunks in %d batches", 
                     total_chunks, num_batches)
