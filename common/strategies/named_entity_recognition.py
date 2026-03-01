from abc import ABC, abstractmethod
from chunk import Chunk
from typing import List, Tuple
import re
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline, pipeline,
)

from common.data_classes.ner_chunk import NERChunk


class NERStrategy(ABC):
    @abstractmethod
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract a list of (entity, type) pairs from *text*."""

    def extract_NERChunk_from_Chunk(self, chunk: Chunk) -> NERChunk:
        """Extract a NERChunk from a Chunk."""
        return NERChunk(chunk=chunk, extracted_entities=self.extract_entities(chunk.text))


class NoNER(NERStrategy):
    """Identity strategy – returns no entities."""

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        return []




class DistilBertNER(NERStrategy):
    """Named‑entity recognition using a 66M‑parameter DistilBERT model."""
    QUOTE_CHARS = "\"'`´“”„‚«»‹›‘’"
    def __init__(
        self,
        model_name: str = "dslim/distilbert-NER",
        device: str | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        device_id = 0 if self.device == "cuda" else -1
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer,  aggregation_strategy="simple", device=device_id)

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Return a list of (entity_text, entity_type) pairs from *text*."""
        outs = self.pipeline(text)

        merged = self._merge_tokens_and_deduplicate(outs, text)
        return self._resolve_types(merged)

    def _resolve_types(self, merged: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        mapping = {
            "PER": "Person",
            "LOC": "Location",
            "MISC": "Miscellaneous",
            "ORG": "Organization",
        }

        return [(word, mapping.get(tag, tag)) for word, tag in merged]

    def _merge_tokens_and_deduplicate(self, tokens, text):
        # ---- merge contiguous subword/B-I pieces ----
        merged = []
        cur = None
        for t in tokens:
            label = t["entity_group"].split("-", 1)[-1]  # drop B-/I-
            start, end = t["start"], t["end"]

            if cur and label == cur["label"]:
                gap = text[cur["end"]:start]
                # merge if only whitespace or common joiners between pieces
                if gap == "" or gap.isspace() or gap in {"-", "’", "'", "."}:
                    cur["end"] = end
                    continue

            if cur:
                merged.append(cur)
            cur = {"start": start, "end": end, "label": label}
        if cur:
            merged.append(cur)

        def _norm(s: str) -> str:
            # ignore leading whitespace
            s = s.lstrip()
            # remove only LEADING quote-like characters (none from the end)
            s = re.sub(rf'^[{re.escape(self.QUOTE_CHARS)}]+', '', s)
            return s.lower()

        seen = set()
        out = []
        for m in merged:
            ent_text = text[m["start"]:m["end"]]
            key = (m["label"], _norm(ent_text))  # dedup by normalized text + label
            if key in seen:
                continue
            seen.add(key)
            out.append((ent_text, m["label"]))

        return out