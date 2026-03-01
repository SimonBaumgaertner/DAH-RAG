import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from common.data_classes.documents import Document
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ContextualizedSentenceChunker
from common.strategies.named_entity_recognition import DistilBertNER

EXPERIMENT_MAIN = "NER"


timestamp = datetime.now().strftime("%d-%m_%H-%M")
log_name = f"{EXPERIMENT_MAIN}_{timestamp}"
log_dir = Path(__file__).parent / "logs"
output_dir = Path(__file__).parent / "outputs"

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

log = RunLogger(run_id=log_name, log_dir=log_dir)
ner_strategy = DistilBertNER()
chunking_strategy = ContextualizedSentenceChunker()

root = Path(__file__).resolve().parents[2]
harry_potter_3 = Document.from_folder(root / "data/NovelQA/harry_potter_and_the_prisoner_of_azkaban")
wiki_james_dean = Document.from_folder(root / "data/HotpotQA_1k/James_Dean")
multihop_rag = Document.from_folder(root / "data/MultiHopRAG/7 of the best ski holidays in Canada")

chunks = chunking_strategy.chunk(multihop_rag)
for chunk in chunks:
    print("--------------------------------")
    ner_chunk = ner_strategy.extract_NERChunk_from_Chunk(chunk)
    print(ner_chunk.to_json())
    # save to json
    with open(f"{output_dir}/{chunk.chunk_id}.json", "w") as f:
        json.dump(ner_chunk.to_json(), f, indent=2)