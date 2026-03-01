
from datetime import datetime
from pathlib import Path
from datetime import datetime

from common.data_classes.documents import Document
from common.data_classes.knowledge_triplets import StructuredDocument
from common.logging.run_logger import RunLogger
from common.neo4j.db_installer import DbInstaller
from common.neo4j.neo4j_environment import Neo4JEnvironment
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.chunking import ContextualizedSentenceChunker
from common.strategies.encoding import MiniLMMeanPoolingEncoder

EXPERIMENT_NAME = "Document_Persist"



def main() -> None:
    backup_path = Path(__file__).resolve().parent / "backup" / "harry_potter_graph.dump"
    # logging
    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    log_name = f"{EXPERIMENT_NAME}_{timestamp}"
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log = RunLogger(run_id=log_name, log_dir=log_dir)

    # prepare neo4J
    env = Neo4JEnvironment(log=log)
    executor = StandardExecutor(env=env, encoder=MiniLMMeanPoolingEncoder())
    installer = DbInstaller()
    installer.installDB(env=env, executor=executor, log=log)

    # prepare document
    chunking_strategy = ContextualizedSentenceChunker()
    project_root = Path(__file__).resolve().parents[2]
    document = Document.from_folder(project_root / "data 🗃️/NovelQA/harry_potter_and_the_prisoner_of_azkaban")
    chunks = chunking_strategy.chunk(document)
    structured_document = StructuredDocument.load(project_root / "logs/Harry Potter and the Prisoner of Azkaban.json")

    # persist document
    log.info("📥 Begin persisting document")
    executor.persist(structured=structured_document, chunks=chunks)
    log.info("✅ Finished persisting document")

    env.open_browser()
    # Wait for user to press Enter
    input("Press Enter to exit the program...")
    env.export_to_file(backup_path)

if __name__ == "__main__":
    main()

main
