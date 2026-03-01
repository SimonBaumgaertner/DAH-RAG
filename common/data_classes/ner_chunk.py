

from dataclasses import dataclass
from typing import Dict, List, Tuple

from common.data_classes.rag_system import Chunk


@dataclass
class NERChunk:
    chunk: Chunk
    extracted_entities: List[Tuple[str, str]]
    extracted_relationships: List[Tuple[str, str, str]]

    def __init__(self, chunk: Chunk, extracted_entities: List[Tuple[str, str]]):
        self.chunk = chunk
        self.extracted_entities = extracted_entities
        self.extracted_relationships = []
    
    @classmethod
    def from_chunk_and_entities(cls, chunk: Chunk, extracted_entities: List[Tuple[str, str]]) -> 'NERChunk':
        """Create NERChunk from chunk and pre-extracted entities."""
        return cls(chunk=chunk, extracted_entities=extracted_entities)

    def to_json(self) -> Dict:
        # Group entities by type
        entities_by_type = {}
        for entity, entity_type in self.extracted_entities:
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            if entity not in entities_by_type[entity_type]:  # Avoid duplicates
                entities_by_type[entity_type].append(entity)
        
        return {
            "chunk": self.chunk.to_json(),
            "entities": entities_by_type,
            "extracted_relationships": self.extracted_relationships
        }