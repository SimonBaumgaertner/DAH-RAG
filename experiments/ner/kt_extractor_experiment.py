from datetime import datetime
from pathlib import Path

from common.data_classes.documents import Document
from common.llm.llm_factory import get_llm_runner
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ContextualizedSentenceChunker
from common.strategies.knowledge_triplet_extraction import StandardTripletExtraction
from common.strategies.named_entity_recognition import DistilBertNER
from common.templates.knowledge_triplet_extraction_template import KnowledgeTripletExtractionTemplate

EXPERIMENT_MAIN = "KT_Extraction"
def main() -> None:
    # log
    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    log_name = f"{EXPERIMENT_MAIN}_{timestamp}"
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log = RunLogger(run_id=log_name, log_dir=log_dir)

    # strategies
    ner_strategy = DistilBertNER()
    chunking_strategy = ContextualizedSentenceChunker()
    template_builder = KnowledgeTripletExtractionTemplate()
    # llm
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "qwen3-4b"
    llm = get_llm_runner(backend="local", model=model_path, log=log)




    # choices of doc
    root = Path(__file__).resolve().parents[2]
    harry_potter_3 = Document.from_folder(root / "data 🗃️/NovelQA/harry_potter_and_the_prisoner_of_azkaban")
    wiki_Ralph_Murphy = Document.from_folder(root / "data 🗃️/HotpotQA_1k/Ralph Murphy")
    wiki_2014_FIFA_World_Cup = Document.from_folder(root / "data 🗃️/HotpotQA_1k/2014 FIFA World Cup")
    # used
    used_doc = harry_potter_3
    triplet_extraction_strategy = StandardTripletExtraction(llm=llm, log=log)

    structured_document = triplet_extraction_strategy.extract_and_build_structured_doc(used_doc)

    output_path = f"{log_dir}/{used_doc.title}.json"
    structured_document.save(Path(output_path))


if __name__ == "__main__":
    main()