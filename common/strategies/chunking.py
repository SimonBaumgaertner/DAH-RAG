import re
from abc import ABC, abstractmethod
from typing import List, Sequence

from transformers import PreTrainedTokenizerBase, AutoTokenizer

from common.data_classes.documents import Document
from common.data_classes.rag_system import Chunk


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        ...

class FixedSizeWordChunker(ChunkingStrategy):
    def __init__(self, *, chunk_size: int = 1200):
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> List[Chunk]:
        words = document.text.split()
        chunks: List[Chunk] = []
        for i in range(0, len(words), self.chunk_size):
            piece = words[i : i + self.chunk_size]
            chunk_id = f"{document.id}_{i//self.chunk_size:04d}"
            chunks.append(Chunk(chunk_id=chunk_id,
                                text=" ".join(piece),
                                doc_id=document.id))
        return chunks


class ContextualizedSentenceChunker(ChunkingStrategy):
    """Sentence‑aware deterministic chunker for ~1200 token windows."""
    _sentence_split_re = re.compile(r"(?<=[.!?])[\s]+", flags=re.MULTILINE)

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        target_tokens: int = 1200,
        min_tokens: int = 1100,
        max_tokens: int = 1300,
        overlap_tokens: int = 100,
    ) -> None:
        if not (0 < min_tokens <= target_tokens <= max_tokens):  # sanity
            raise ValueError("Token thresholds must satisfy 0 < min ≤ target ≤ max")
        if overlap_tokens >= min_tokens:
            raise ValueError("overlap_tokens should be < min_tokens to guarantee progress")

        self.tokenizer: PreTrainedTokenizerBase = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True
        )
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _sentences(self, text: str) -> Sequence[str]:
        """Split text into sentences deterministically using regex."""
        # Also strip any lingering whitespace around each sentence.
        return [s.strip() for s in self._sentence_split_re.split(text) if s.strip()]

    def _token_len(self, text: str) -> int:
        """Return *number of tokens* without adding special tokens."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))


    def chunk(self, document: Document) -> List[Chunk]:  # noqa: D401, N802
        """Deterministically split *document* into contextual chunks.

        Parameters
        ----------
        document:
            The document whose raw_text will be chunked.

        Returns
        -------
        list[Chunk]
            A list of `Chunk` objects in original order; each chunk meets the
            350‑425 token constraint, never breaking sentences when avoidable.
        """
        sentences = self._sentences(document.text)

        chunks: List[Chunk] = []
        current_chunk: List[str] = []
        current_len = 0  # token length of current_chunk

        for sent in sentences:
            sent_len = self._token_len(sent)

            # Case 1: the sentence alone is too long. Rare but must abort.
            if sent_len > self.max_tokens:
                truncated_tokens = self.tokenizer.encode(sent, add_special_tokens=False)[: self.max_tokens]
                truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                self._flush_chunk(chunks, truncated_text, document.id, len(chunks))
                current_chunk = []
                current_len = 0
                continue

            # If adding this sentence would exceed max_tokens, flush first.
            if current_len + sent_len > self.max_tokens:
                if current_len < self.target_tokens:
                    self._apply_overlap(current_chunk, self.overlap_tokens)

                self._flush_chunk(chunks, " ".join(current_chunk), document.id, len(chunks))
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            if current_len < self.min_tokens and chunks:
                # Attempt to pull tokens from previous chunk tail deterministically.
                prev_chunk_text = chunks[-1].text
                tail_tokens = self.tokenizer.encode(prev_chunk_text, add_special_tokens=False)[
                    -self.overlap_tokens :
                ]
                tail_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)
                # Prepend tail only if it keeps us ≤ max_tokens
                if current_len + len(tail_tokens) <= self.max_tokens:
                    current_chunk.insert(0, tail_text)
                    current_len += len(tail_tokens)
            self._flush_chunk(chunks, " ".join(current_chunk), document.id, len(chunks))

        return chunks

    def _apply_overlap(self, chunk_sents: List[str], overlap_tokens: int) -> None:
        """Append final *overlap_tokens* from chunk_sents[-1] to itself.

        Operates in‑place on *chunk_sents* by duplicating a tail token span
        to be reused in the next chunk. Ensures we stay ≤ max_tokens overall.
        """
        if not chunk_sents:
            return
        last_sent = chunk_sents[-1]
        tokens = self.tokenizer.encode(last_sent, add_special_tokens=False)
        if len(tokens) <= overlap_tokens:
            # Duplicate entire sentence.
            chunk_sents.append(last_sent)
        else:
            tail_tokens = tokens[-overlap_tokens:]
            tail_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)
            chunk_sents.append(tail_text)

    def _flush_chunk(self, chunks: List[Chunk], text: str, doc_id: str, idx: int) -> None:
        text = text.strip()
        if not text:
            return
        chunk_id = f"{doc_id}_{idx:04d}"
        chunks.append(Chunk(chunk_id=chunk_id, text=text, doc_id=doc_id))


class SemanticChunker(ChunkingStrategy):
    def __init__(
        self,
        *,
        encoding_strategy,  # EncodingStrategy instance
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,
        number_of_chunks: int | None = None,
        max_chunk_tokens: int = 1000,  # Target chunk size in tokens
    ) -> None:
        from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
        
        self.encoding_strategy = encoding_strategy
        self.max_chunk_tokens = max_chunk_tokens
        
        # Wrap our encoding strategy to be compatible with LangChain's Embeddings interface
        embeddings_adapter = _EncodingStrategyAdapter(encoding_strategy)
        
        estimated_buffer_size = max(1, max_chunk_tokens // 20)
        
        self.lc_chunker = LCSemanticChunker(
            embeddings=embeddings_adapter,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
            buffer_size=estimated_buffer_size,
        )
    
    def chunk(self, document: Document) -> List[Chunk]:
        lc_documents = self.lc_chunker.create_documents([document.text])
        
        chunks: List[Chunk] = []
        for idx, lc_doc in enumerate(lc_documents):
            chunk_id = f"{document.id}_{idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=lc_doc.page_content,
                    doc_id=document.id
                )
            )
        
        return chunks


class _EncodingStrategyAdapter:
    def __init__(self, encoding_strategy):
        self.encoding_strategy = encoding_strategy
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (passages)."""
        embeddings = []
        for text in texts:
            # Use query=False for document embeddings
            embedding = self.encoding_strategy.encode(text, query=False)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        # Use query=True for query embeddings
        embedding = self.encoding_strategy.encode(text, query=True)
        return embedding.tolist()
