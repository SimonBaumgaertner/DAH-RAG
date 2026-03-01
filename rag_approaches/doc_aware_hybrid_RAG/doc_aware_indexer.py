from pathlib import Path
from typing import Optional
import time

from common.data_classes.documents import Document
from common.data_classes.evaluation import EntryType
from common.data_classes.knowledge_triplets import StructuredDocument
from common.data_classes.rag_system import Indexer
from common.logging.run_logger import RunLogger
from common.neo4j.db_installer import DbInstaller
from common.neo4j.neo4j_environment import Neo4JEnvironment
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.chunking import ChunkingStrategy
from common.strategies.encoding import EncodingStrategy
from common.strategies.knowledge_triplet_extraction import KnowledgeTripletExtractionStrategy


class DocumentAwareIndexer(Indexer):
    def __init__(
        self,
        *,
        encoder: EncodingStrategy,
        chunker: ChunkingStrategy,
        log: RunLogger,
        triplet_extraction_strategy: KnowledgeTripletExtractionStrategy,
        env: Neo4JEnvironment,
        executor: StandardExecutor,
        clean_before_indexing: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        self.triplet_extraction_strategy = triplet_extraction_strategy
        self.encoder, self.chunker, self.log = encoder, chunker, log
        self.env = env
        self.executor = executor
        self.clean_before_indexing = clean_before_indexing
        # Use provided cache_dir or fall back to default location
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cached_structured_documents"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    def index(self, document: Document) -> None:
        # Initialize timing dictionary
        timings = {}
        total_start = time.time()
        
        # Database cleaning
        if self.clean_before_indexing:
            clean_start = time.time()
            self.executor.cleandb()
            timings['db_cleaning'] = time.time() - clean_start
            self.log.info("🧹 Clean database (look at config if you did not want this) - took %.3fs", timings['db_cleaning'])
        else:
            timings['db_cleaning'] = 0.0
        
        # Chunking
        self.log.info("🔪 Chunking doc %s …", document.id)
        chunk_start = time.time()
        chunks = self.chunker.chunk(document)
        timings['chunking'] = time.time() - chunk_start
        self.log.info("📑 → %d chunks - took %.3fs", len(chunks), timings['chunking'])
        
        # Cache check and triplet extraction
        cached_path = self.cache_dir / f"{document.title}.json"
        if cached_path.exists():
            cache_start = time.time()
            self.log.info("📁 Using cached document")
            structured_document = StructuredDocument.load(cached_path)
            
            structured_document.document.text = document.text
            
            timings['cache_loading'] = time.time() - cache_start
            timings['triplet_extraction'] = 0.0
            self.log.info("📁 Cache loading took %.3fs", timings['cache_loading'])
        else:
            timings['cache_loading'] = 0.0
            extraction_start = time.time()
            structured_document = self.triplet_extraction_strategy.extract_and_build_structured_doc(document, chunks)
            timings['triplet_extraction'] = time.time() - extraction_start
            self.log.info("🔗 Triplet extraction took %.3fs", timings['triplet_extraction'])
            
            save_start = time.time()
            structured_document.save(cached_path)
            timings['cache_saving'] = time.time() - save_start
            self.log.info("💾 Cache saving took %.3fs", timings['cache_saving'])

        # Database persistence
        persist_start = time.time()
        self.executor.persist(structured=structured_document, chunks=chunks)
        timings['db_persistence'] = time.time() - persist_start
        self.log.info("💾 Database persistence took %.3fs", timings['db_persistence'])

        # Track chunk count for this document
        self.log.track(
            entry_type=EntryType.CHUNK_COUNT_TRACK.value,
            identifier=document.id,
            value=str(len(chunks))
        )

        timings['total'] = time.time() - total_start
        
        # Print timing summary in one line
        self.log.info("📥 Stored %d chunks in GraphDB for doc %s", len(chunks), document.id)
        self.log.info(
            "⏱️  INDEXING TIMING for %s: TOTAL=%.3fs | DB_Clean=%.3fs (%.1f%%) | Chunk=%.3fs (%.1f%%) | "
            "Cache_Load=%.3fs (%.1f%%) | Triplet_Extract=%.3fs (%.1f%%) | %sDB_Persist=%.3fs (%.1f%%)",
            document.id,
            timings['total'],
            timings['db_cleaning'], 100 * timings['db_cleaning'] / timings['total'] if timings['total'] > 0 else 0,
            timings['chunking'], 100 * timings['chunking'] / timings['total'] if timings['total'] > 0 else 0,
            timings['cache_loading'], 100 * timings['cache_loading'] / timings['total'] if timings['total'] > 0 else 0,
            timings['triplet_extraction'], 100 * timings['triplet_extraction'] / timings['total'] if timings['total'] > 0 else 0,
            f"Cache_Save={timings.get('cache_saving', 0.0):.3f}s ({100 * timings.get('cache_saving', 0.0) / timings['total'] if timings['total'] > 0 else 0:.1f}%) | " if 'cache_saving' in timings else "",
            timings['db_persistence'], 100 * timings['db_persistence'] / timings['total'] if timings['total'] > 0 else 0
        )
