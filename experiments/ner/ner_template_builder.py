import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from common.data_classes.ner_chunk import NERChunk
from common.data_classes.rag_system import Chunk
from common.templates.knowledge_triplet_extraction_template import KnowledgeTripletExtractionTemplate


def load_ner_chunk_from_json(json_path: Path) -> NERChunk:
    """Load NERChunk from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct the chunk
    chunk_data = data["chunk"]
    chunk = Chunk(
        chunk_id=chunk_data["chunk_id"],
        text=chunk_data["text"]
    )
    
    # Reconstruct entities from the grouped format
    entities = []
    entities_by_type = data.get("entities", {})
    for entity_type, entity_list in entities_by_type.items():
        for entity in entity_list:
            entities.append((entity, entity_type))
    
    return NERChunk(chunk=chunk, extracted_entities=entities)


def build_templates_from_outputs(outputs_dir: Path) -> List[Dict]:
    """Build templates from all JSON files in outputs directory."""
    template_builder = KnowledgeTripletExtractionTemplate()
    templates = []
    
    # Get all JSON files in outputs directory
    json_files = list(outputs_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {outputs_dir}")
    print("=" * 80)
    
    for json_file in sorted(json_files):
        print(f"\n📄 Processing: {json_file.name}")
        print("-" * 60)
        
        try:
            # Load NERChunk from JSON
            ner_chunk = load_ner_chunk_from_json(json_file)
            
            # Build template
            template = template_builder.build_from_ner_chunk(ner_chunk)
            
            # Store template info
            template_info = {
                "file": json_file.name,
                "chunk_id": ner_chunk.chunk.chunk_id,
                "template": template,
                "entities_count": len(ner_chunk.extracted_entities),
                "entities_by_type": ner_chunk.to_json()["entities"]
            }
            templates.append(template_info)
            
            # Print template for chat use
            print("🤖 SYSTEM PROMPT:")
            print(template[0]["content"])
            print("\n👤 USER PROMPT:")
            print(template[1]["content"])
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            continue
    
    return templates


def main():
    """Main function to build templates from NER outputs."""
    # Get outputs directory
    outputs_dir = Path(__file__).parent / "outputs"
    
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return
    
    # Build templates
    templates = build_templates_from_outputs(outputs_dir)
    
    print(f"\n✅ Successfully processed {len(templates)} files")
    print(f"📊 Total entities found: {sum(t['entities_count'] for t in templates)}")
    
    # Print summary
    print("\n📋 SUMMARY:")
    for template_info in templates:
        entities_by_type = template_info["entities_by_type"]
        type_summary = ", ".join(f"{t}: {len(e)}" for t, e in entities_by_type.items())
        print(f"  • {template_info['file']}: {template_info['entities_count']} entities ({type_summary})")


if __name__ == "__main__":
    main()
