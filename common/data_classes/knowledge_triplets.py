from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Dict

from common.data_classes.documents import Document


# ----------------------------
# Data classes for knowledge triplets
# ----------------------------

@dataclass
class Entity:
    """Entity with a canonical primary name, a simple string type, and optional aliases."""
    name: str
    type: str  # e.g., "Person", "Location", "Organization", ...
    aliases: List[str]

    def to_dict(self) -> Dict:
        return {"name": self.name, "type": self.type, "aliases": list(self.aliases)}

    @staticmethod
    def from_dict(d: Dict) -> Entity:
        # Strict: require "type" to be present (no legacy compatibility).
        if "type" not in d:
            raise ValueError("Entity JSON missing required field 'type'.")
        return Entity(name=d["name"], type=d["type"], aliases=list(d.get("aliases", [])))


@dataclass
class ExtractedKnowledgeTriplet:
    """Knowledge triplet referencing entities by their primary names."""
    subject: str
    relationship: str
    object: str
    chunk_id: str
    rank: int

    # explicit constructor to normalize inputs
    def __init__(
        self,
        *,
        subject: str,
        relationship: str,
        object: str,
        chunk_id: str,
        rank: int = 1,
    ) -> None:
        self.subject = subject.strip()
        self.relationship = relationship.strip()
        self.object = object.strip()
        self.chunk_id = chunk_id
        self.rank = rank

    def to_string(self) -> str:
        return f"{self.subject} {self.relationship} {self.object}"

    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "relationship": self.relationship,
            "object": self.object,
            "chunk_id": self.chunk_id,
            "rank": self.rank
        }

    @staticmethod
    def from_dict(d: Dict) -> ExtractedKnowledgeTriplet:
        return ExtractedKnowledgeTriplet(
            subject=d["subject"],
            relationship=d["relationship"],
            object=d["object"],
            chunk_id=d["chunk_id"],
            rank=d["rank"]
        )


# ----------------------------
# Structured document
# ----------------------------

@dataclass
class StructuredDocument:
    """
    Whole-document container with DB-aligned document metadata, an entity catalog,
    and knowledge triplets.

    JSON shape:
    {
      "document": {
        "id": "...",
        "title": "...",
        "author": "...",
        "publication_date": "YYYY-MM-DD",
        "references": ["doc_id_1", "doc_id_2", ...]
      },
      "entities": [
        {"name": "...", "type": "Person", "aliases": ["...", ...]},
        ...
      ],
      "triplets": [
        {"subject": "...", "relationship": "...", "object": "...", "chunk_id": "..."},
        ...
      ]
    }
    """
    document: Document
    entities: List[Entity]
    triplets: List[ExtractedKnowledgeTriplet]

    # -------- Import / Export --------
    def to_dict(self) -> Dict:
        return {
            "document": {
                "id": self.document.id,
                "title": self.document.title,
                "author": self.document.author,
                "publication_date": (
                    self.document.publication_date.isoformat()
                    if self.document.publication_date else None
                ),
                "references": list(self.document.references),
            },
            "entities": [e.to_dict() for e in self.entities],
            "triplets": [t.to_dict() for t in self.triplets],
        }

    @staticmethod
    def from_dict(d: Dict) -> StructuredDocument:
        doc_d = d["document"]

        pub_date_raw = doc_d.get("publication_date")
        pub_date = None
        if pub_date_raw:  # only try parsing if not None/empty
            try:
                pub_date = date.fromisoformat(pub_date_raw)
            except ValueError:
                pub_date = None

        document = Document(
            id=doc_d["id"],
            title=doc_d.get("title"),
            author=doc_d.get("author"),
            publication_date=pub_date,
            references=list(doc_d.get("references", [])),
        )
        entities = [Entity.from_dict(e) for e in d.get("entities", [])]
        triplets = [ExtractedKnowledgeTriplet.from_dict(t) for t in d.get("triplets", [])]
        return StructuredDocument(document=document, entities=entities, triplets=triplets)

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @staticmethod
    def load(input_path: Path) -> StructuredDocument:
        with input_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or "document" not in raw:
            raise ValueError("Invalid StructuredDocument JSON format: missing 'document'.")
        return StructuredDocument.from_dict(raw)
