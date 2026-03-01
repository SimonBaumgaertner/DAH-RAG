# knowledge_triplet_extraction.py
import asyncio
import operator
import re
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from common.data_classes.documents import Document
from common.data_classes.knowledge_triplets import (
    ExtractedKnowledgeTriplet,
    StructuredDocument,
)
from common.data_classes.ner_chunk import NERChunk
from common.data_classes.rag_system import Chunk
from common.llm.base_llm_runner import BaseLLMRunner
from common.llm.message_template import MessageTemplate
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ChunkingStrategy, ContextualizedSentenceChunker
from common.strategies.encoding import EncodingStrategy, MiniLMMeanPoolingEncoder
from common.strategies.entity_processing import (
    build_aliases,
    resolve_aliases,
    deduplicate_triplets,
)
from common.strategies.named_entity_recognition import NERStrategy, DistilBertNER
from common.templates.knowledge_triplet_extraction_template import (
    KnowledgeTripletExtractionTemplate,
)


class KnowledgeTripletExtractionStrategy(ABC):
    @abstractmethod
    def extract_and_build_structured_doc(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
    ) -> StructuredDocument:
        """
        Extract knowledge triplets from a document (and optional precomputed chunks)
        and return a StructuredDocument. If `chunks` is not provided, the
        implementation should perform its own chunking.
        """
        ...


def parse_to_triplets(answer: str, chunk_id: str) -> List[ExtractedKnowledgeTriplet]:
    triplets: List[ExtractedKnowledgeTriplet] = []
    # Find all matches of the pattern {'subject','relationship','object'}
    matches = re.findall(r"\{'\s*(.*?)\s*','\s*(.*?)\s*','\s*(.*?)\s*'\}", answer)

    for subject, relationship, obj in matches:
        triplet = ExtractedKnowledgeTriplet(
            subject=subject,
            relationship=relationship,
            object=obj,
            chunk_id=chunk_id,
        )
        triplets.append(triplet)

    return triplets


class StandardTripletExtraction(KnowledgeTripletExtractionStrategy):
    def __init__(
        self,
        *,
        llm: BaseLLMRunner,
        log: RunLogger,
        chunking: ChunkingStrategy | None = None,
        encoding: EncodingStrategy | None = None,
        ner: NERStrategy = DistilBertNER(),
        extraction_template: MessageTemplate = KnowledgeTripletExtractionTemplate(),
    ):
        encoding = encoding or MiniLMMeanPoolingEncoder()
        chunking = chunking or ContextualizedSentenceChunker(tokenizer=encoding.tokenizer)
        self.chunking = chunking
        self.encoding = encoding
        self.ner = ner
        self.llm = llm
        self.log = log
        self.extraction_template = extraction_template

    def extract_and_build_structured_doc(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
    ) -> StructuredDocument:
        """
        Extract knowledge triplets and build a StructuredDocument.

        Args:
            document: The source document.
            chunks: Optional precomputed chunks. If provided, these will be used
                    directly; otherwise, the strategy will chunk `document` itself.

        Returns:
            StructuredDocument containing entities (with aliases) and deduplicated triplets.
        """
        # Use provided chunks if available; otherwise, perform chunking.
        if chunks is None:
            chunks = self.chunking.chunk(document)

        # Run async extraction
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._extract_async(document, chunks))

    async def _extract_async(
        self,
        document: Document,
        chunks: List[Chunk],
    ) -> StructuredDocument:
        """Async implementation of triplet extraction with parallel LLM calls."""
        import time
        
        all_triplets: List[ExtractedKnowledgeTriplet] = []
        entities: List[Tuple[str, str]] = []

        # Process all chunks in parallel
        async def process_chunk(chunk: Chunk) -> Tuple[List[ExtractedKnowledgeTriplet], List[Tuple[str, str]], float]:
            """Process a single chunk and return its triplets and entities."""
            chunk_start = time.time()
            
            # Extract entities once
            extracted_entities = self.ner.extract_entities(chunk.text)
            
            # Create NERChunk using the already extracted entities
            ner_chunk = NERChunk.from_chunk_and_entities(chunk, extracted_entities)

            # Build prompt using the new NERChunk method
            prompt = self.extraction_template.build_from_ner_chunk(ner_chunk)

            context_was_set = False
            if self.log.indexing_document_id != document.id:
                self.log.set_indexing_context(document.id)
                context_was_set = True

            try:
                # Use async LLM call for parallel processing
                from common.data_classes.evaluation import LLMCallContext
                identifier = document.id
                answer = await self.llm.generate_text_async(prompt, context=LLMCallContext.INDEXING, identifier=identifier)
            finally:
                if context_was_set:
                    self.log.set_indexing_context(None)

            # Parse triplets
            triplets = parse_to_triplets(answer, chunk.chunk_id)
            
            chunk_time = time.time() - chunk_start
            return triplets, extracted_entities, chunk_time

        # Process all chunks concurrently
        parallel_start = time.time()
        results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
        parallel_time = time.time() - parallel_start
        
        # Collect results
        chunk_times = []
        for triplets, extracted_entities, chunk_time in results:
            all_triplets.extend(triplets)
            entities.extend(extracted_entities)
            chunk_times.append(chunk_time)
        
        # Calculate timing stats
        sequential_time = sum(chunk_times)
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        

        # Resolve duplications and aliases
        entities_with_aliases = build_aliases(all_triplets, entities)
        alias_resolved_triplets = resolve_aliases(
            triplets=all_triplets, entities=entities_with_aliases
        )
        deduplicated_triplets = deduplicate_triplets(alias_resolved_triplets)

        # Build StructuredDocument
        doc = StructuredDocument(
            document=document,
            entities=entities_with_aliases,
            triplets=deduplicated_triplets,
        )
        return doc

