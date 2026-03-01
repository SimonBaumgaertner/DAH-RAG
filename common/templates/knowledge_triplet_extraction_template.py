from typing import List, Dict

from common.data_classes.qa import QuestionAnswerPair  # noqa: F401
from common.data_classes.rag_system import Chunk       # noqa: F401
from common.data_classes.ner_chunk import NERChunk
from common.llm.message_template import MessageTemplate


class KnowledgeTripletExtractionTemplate(MessageTemplate):
    SYSTEM_PROMPT = (
        "You are a smart relationship extractor. "
        "Using ONLY the given Entities and the Chunk extract sophisticated knowledge triplets in the form "
        "{'Subject','Relationship','Object'}. Here are some examples of well done triplets:\n"
        "{'Penguins','live in','Antarctica'}\n"
        "{'Catherine','alias','Cat'}\n"
        "{'Monotremes','are a type of','Mammal'}\n"
        "{'Ralph Murphy','is','American'}\n"
        "{'Dursleys','are blood relatives to','Harry'}\n"
        "{'London','is located in','England'}\n"
        "{'Michael Jr','was scarred by','Father'}\n"
        "You should NOT extract triplets like this:\n"
        "{'Harry','asked','Ron'}, <- this is BAD, as it does not convey meaningful information \n"
        "{'Whistler Blackcomb','is','the largest ski resort in North America'}, <- this is BAD, as it is not a relationship between two entities\n"
        "{'Whistler Blackcomb','has','over 8,100 acres of skiable terrain'}, <- this is BAD, as its a property of the subject, not a relationship between two entities\n"
        "Only return the triplets. Note that Subject and Object should be in the entity list.\n"
    )

    @staticmethod
    def _format_entities(entities: List[str]) -> str:
        entities = entities or []
        return "Entities:\n" + ",".join(f"'{e}'" for e in entities)

    @staticmethod
    def _format_entities_by_type(entities_by_type: Dict[str, List[str]]) -> str:
        """Format entities grouped by type for better context."""
        if not entities_by_type:
            return "Entities: None"
        
        formatted_entities = []
        for entity_type, entities in entities_by_type.items():
            entity_list = ", ".join(f"'{e}'" for e in entities)
            formatted_entities.append(f"{entity_type}: {entity_list}")
        
        return "Entities by type:\n" + "\n".join(formatted_entities)

    def build_from_template(
        self,
        entities: List[str],
        chunk: Chunk
    ) -> List[Dict[str, str]]:

        entity_block = self._format_entities(entities)
        chunk_text = getattr(chunk, "text", str(chunk))

        user_content = f"{entity_block}\nChunk:\n{chunk_text}\nGenerate up to two triples."

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def build_from_ner_chunk(
        self,
        ner_chunk: NERChunk
    ) -> List[Dict[str, str]]:
        """Build template using NERChunk with structured entity information."""
        
        # Get entities grouped by type from the NERChunk
        ner_data = ner_chunk.to_json()
        entities_by_type = ner_data.get("entities", {})
        
        # Format entities by type for better context
        entity_block = self._format_entities_by_type(entities_by_type)
        chunk_text = ner_chunk.chunk.text

        user_content = (
            f"{entity_block}\nChunk:\n{chunk_text}\n\n End Chunk.\n\n"
            "Generate the three most relevant relationships between the entities in the chunk in the form\n"
            "{'Subject1','Relationship1','Object1'},\n"
            "{'Subject2','Relationship2','Object2'},\n"
            "{'Subject3','Relationship3','Object3'}"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
