from pathlib import Path
from enum import Enum

from common.data_classes.rag_system import RAGSystem, Generator
from common.llm.base_llm_runner import BaseLLMRunner
from typing import Optional
from common.llm.message_template import MessageTemplate
from common.logging.run_logger import RunLogger
from common.neo4j.db_installer import DbInstaller
from common.neo4j.neo4j_environment import Neo4JEnvironment
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.chunking import ChunkingStrategy, FixedSizeWordChunker, ContextualizedSentenceChunker
from common.strategies.encoding import EncodingStrategy, MiniLMMeanPoolingEncoder, QwenEncoder
from common.strategies.generator import StandardMCAnswerGenerator
from common.strategies.graph_search.abstract_graph_search import GraphSearch
from common.strategies.graph_search.document_aware_advanced_graph_search import DocumentAwareAdvancedGraphSearch
from common.strategies.graph_search.document_aware_experimental_graph_search import DocumentAwareExperimentalGraphSearch
from common.strategies.graph_search.document_aware_dense_graph_search import DocumentAwareDenseGraphSearch
from common.strategies.graph_search.hipporag_graph_search import HippoRAGGraphSearch
from common.strategies.graph_search.vector_graph_search import VectorGraphSearch
from common.strategies.knowledge_triplet_extraction import KnowledgeTripletExtractionStrategy, StandardTripletExtraction
from common.strategies.named_entity_recognition import NERStrategy, DistilBertNER
from common.templates.knowledge_triplet_extraction_template import KnowledgeTripletExtractionTemplate
from rag_approaches.doc_aware_hybrid_RAG.doc_aware_indexer import DocumentAwareIndexer
from rag_approaches.doc_aware_hybrid_RAG.doc_aware_retriever import DocumentAwareRetriever


class GraphSearchMode(Enum):
    """Enum to easily switch between different graph search strategies."""
    EXPERIMENTAL = "experimental"
    HIPPORAG = "hipporag"
    ADVANCED = "advanced"


class DocumentAwareHybridRAG(RAGSystem):
    def __init__(
            self,
            *,
            llm: BaseLLMRunner,
            name: str = "document‑aware‑hybrid-rag",
            log: RunLogger | None = None,
            chunker: ChunkingStrategy | None = None,
            encoder: EncodingStrategy | None = None,
            triplet_extraction_strategy: KnowledgeTripletExtractionStrategy | None = None,
            extraction_template: Optional[MessageTemplate] = None,
            ner: NERStrategy | None = None,
            database_data_root: Path = None,
            chunk_size: int = 1200,
            generation_max_chunks: int = 5,
            graph_search: GraphSearch | None = None,
            generator: Generator | None = None,
            clean_before_indexing: bool = False, # if we should clean the db before indexing a new document
            checkpoint_name: Optional[str] = None,
            skip_db_reset: bool = False, # if True, don't clear the database on initialization
    ):
        log = log or RunLogger(run_id=name)

        # init strategies
        encoder = encoder or QwenEncoder(task="query", max_length=chunk_size, model_name="Qwen/Qwen3-Embedding-0.6B", log=log)
        chunker = chunker or ContextualizedSentenceChunker(tokenizer=encoder.tokenizer)
        ner = ner or DistilBertNER()
        extraction_template = extraction_template or KnowledgeTripletExtractionTemplate()
        triplet_extraction_strategy = triplet_extraction_strategy or StandardTripletExtraction(llm=llm, log=log, chunking=chunker, encoding=encoder, ner=ner, extraction_template=extraction_template)

        # Determine cache directory based on checkpoint_name
        project_root = Path(__file__).resolve().parents[2]
        cache_storage_dir = project_root / "experiments" / "cache_and_storage"
        if checkpoint_name and checkpoint_name.strip():
            cache_dir = cache_storage_dir / checkpoint_name.strip() / "cached_structured_documents"
            log.info(f"🔄 Using checkpoint directory: {checkpoint_name.strip()}")
        else:
            cache_dir = cache_storage_dir / log.run_id / "cached_structured_documents"
            log.info(f"🆕 Using new experiment directory: {log.run_id}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # init the Neo4J environment
        # If skip_db_reset is True, don't nuke the database on start (preserve existing data)
        self.env = Neo4JEnvironment(log=log, data_root=database_data_root, nuke_on_start=not skip_db_reset)
        executor = StandardExecutor(env=self.env, encoder=encoder)
        installer = DbInstaller()
        installer.installDB(env=self.env, executor=executor, log=log)
        
        # Easy switching between graph search strategies
        # Change the mode here to switch between different approaches:
        GRAPH_SEARCH_MODE = GraphSearchMode.ADVANCED  # Options: EXPERIMENTAL, HIPPORAG, ADVANCED
        
        if graph_search is None:
            if GRAPH_SEARCH_MODE == GraphSearchMode.EXPERIMENTAL:
                graph_search = DocumentAwareExperimentalGraphSearch(executor=executor, log=log)
            elif GRAPH_SEARCH_MODE == GraphSearchMode.HIPPORAG:
                graph_search = HippoRAGGraphSearch(executor=executor, log=log)
            elif GRAPH_SEARCH_MODE == GraphSearchMode.ADVANCED:
                graph_search = DocumentAwareAdvancedGraphSearch(executor=executor, log=log)
            else:
                raise ValueError(f"Unknown graph search mode: {GRAPH_SEARCH_MODE}")
        # init RAG components
        indexer = DocumentAwareIndexer(
            encoder=encoder,
            chunker=chunker,
            log=log,
            triplet_extraction_strategy=triplet_extraction_strategy,
            env=self.env,
            executor=executor,
            clean_before_indexing=clean_before_indexing,
            cache_dir=cache_dir,
        )
        retriever = DocumentAwareRetriever(
            encoder=encoder,
            log=log,
            env=self.env,
            executor=executor,
            search=graph_search,
        )
        generator = generator or StandardMCAnswerGenerator(llm=llm, log=log, max_chunks=generation_max_chunks)

        super().__init__(
            indexer=indexer,
            retriever=retriever,
            generator=generator,
            name=name,
            log=log
        )

    def export_db_dump(self, dump_file: Path):
        self.env.export_to_file(dump_file)
        self.env.check_connection(overwrite_nuke=True) # restart without nuking the db

    def import_db_dump(self, dump_file: Path):
        self.env.import_from_file(dump_file)