from pathlib import Path
from typing import List, Optional

from common.data_classes.documents import Document
from common.data_classes.rag_system import Chunk
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
from common.data_classes.enums import RAG, GenerationConfiguration, Encoder, ChunkingStrategy as ChunkingStrategyEnum
from experiments.base_experiment import prepare_rag_system
from dah_rag_app.database_interface import RAGDatabase

class DAHRAGDatabase(RAGDatabase):
    """
    Implementation of the RAGDatabase interface for the Document-Aware Hybrid RAG system.
    Wraps the existing DocumentAwareHybridRAG to provide a standard database-like interface.
    """

    def __init__(
        self, 
        llm: BaseLLMRunner, 
        log: RunLogger, 
        encoder: Encoder = Encoder.Qwen3_4B,
        chunking_strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.ContextualizedChunker,
        checkpoint_name: Optional[str] = None
    ):
        self.llm = llm
        self.log = log
        self.encoder = encoder
        self.chunking_strategy = chunking_strategy
        self.checkpoint_name = checkpoint_name
        self._rag_system = None 

    def initialize_database(self, wipe_at_start: bool) -> bool:
        try:
            self._rag_system = prepare_rag_system(
                rag_system=RAG.DocAwareHybridRAG,
                generation=GenerationConfiguration.NoGen,
                log=self.log,
                llm=self.llm,
                encoding_strategy=self.encoder,
                chunking_strategy=self.chunking_strategy,
                checkpoint_name=self.checkpoint_name,
                skip_db_reset=not wipe_at_start
            )
            return True
        except Exception as e:
            self.log.error(f"Failed to initialize database: {e}")
            return False

    def get_all_documents(self) -> List[Document]:
        if not self._rag_system:
            raise RuntimeError("Database not initialized. Call initialize_database first.")
            
        driver = self._rag_system.retriever.env.get_driver()
        docs = []
        with driver.session() as session:
            # Match document and optionally all its chunks to reconstruct text if present
            cypher = """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d.id AS id, d.title AS title, d.author AS author, 
                   collect(c) AS chunks
            """
            result = session.run(cypher)
            for record in result:
                # Sort chunks chronologically based on their ID structure 
                # (Assuming they have an order in their name or just concatenating as is)
                chunks_info = record["chunks"]
                chunks_info.sort(key=lambda x: x["chunk_name"] if x and "chunk_name" in x else "")
                reconstructed_text = " ".join([c["text"] for c in chunks_info if c and "text" in c])
                
                docs.append(Document(
                    id=record["id"],
                    title=record["title"],
                    author=record["author"],
                    publication_date=None,
                    references=[],
                    text=reconstructed_text
                ))
        return docs

    def query(self, query_text: str, top_k: int) -> List[Chunk]:
        if not self._rag_system:
            raise RuntimeError("Database not initialized. Call initialize_database first.")
            
        return self._rag_system.retriever.retrieve(question=query_text, return_chunk_amount=top_k)

    def add_document(self, document: Document) -> bool:
        if not self._rag_system:
            raise RuntimeError("Database not initialized. Call initialize_database first.")
            
        try:
            self._rag_system.indexer.index(document)
            return True
        except Exception as e:
            self.log.error(f"Failed to add document {document.id}: {e}")
            return False

    def remove_document(self, document_id: str) -> bool:
        if not self._rag_system:
            raise RuntimeError("Database not initialized. Call initialize_database first.")

        driver = self._rag_system.retriever.env.get_driver()
        try:
            with driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (c)<-[:DERIVED_FROM]-(a:Assertion)
                    DETACH DELETE a, c, d
                    """,
                    doc_id=document_id
                )
            return True
        except Exception as e:
            self.log.error(f"Failed to remove document {document_id}: {e}")
            return False

    def get_document_by_id(self, document_id: str) -> Document:
        if not self._rag_system:
            raise RuntimeError("Database not initialized. Call initialize_database first.")
            
        driver = self._rag_system.retriever.env.get_driver()
        with driver.session() as session:
            cypher = """
            MATCH (d:Document {id: $doc_id})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d.id AS id, d.title AS title, d.author AS author,
                   collect(c) AS chunks
            """
            result = session.run(cypher, doc_id=document_id)
            record = result.single()
            if record:
                chunks_info = record["chunks"]
                chunks_info.sort(key=lambda x: x["chunk_name"] if x and "chunk_name" in x else "")
                reconstructed_text = " ".join([c["text"] for c in chunks_info if c and "text" in c])
                
                return Document(
                    id=record["id"],
                    title=record["title"],
                    author=record["author"],
                    publication_date=None,
                    references=[],
                    text=reconstructed_text
                )
        return None

    def get_all_documents_count(self) -> int:
        if not self._rag_system:
            raise RuntimeError("Database not initialized. Call initialize_database first.")
            
        driver = self._rag_system.retriever.env.get_driver()
        with driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN count(d) AS count")
            record = result.single()
            if record:
                return record["count"]
        return 0
