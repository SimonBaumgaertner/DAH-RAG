from pathlib import Path

from common.data_classes.documents import Document
from common.data_classes.knowledge_triplets import StructuredDocument
from common.neo4j.db_installer import DbInstaller
from common.neo4j.neo4j_environment import Neo4JEnvironment
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.chunking import ContextualizedSentenceChunker
from common.strategies.encoding import MiniLMMeanPoolingEncoder
from experiments.base_experiment import prepare_llm, prepare_log

EXPERIMENT_NAME = "Document_Persist"
RUN_ON_CLUSTER = False


def main() -> None:
    log = prepare_log(EXPERIMENT_NAME)
    prepare_llm(RUN_ON_CLUSTER, log)

    env = Neo4JEnvironment(log=log)
    executor = StandardExecutor(env=env, encoder=MiniLMMeanPoolingEncoder())
    DbInstaller().installDB(env=env, executor=executor, log=log)

    chunking_strategy = ContextualizedSentenceChunker()
    document = Document.from_folder(
        "/home/simon/PycharmProjects/MastersThesis/data 🗃️/NovelQA/harry_potter_and_the_sorcerers_stone/"
    )
    chunks = chunking_strategy.chunk(document)
    structured_document = StructuredDocument.load(
        Path(
            "/home/simon/PycharmProjects/MastersThesis/logs_and_tracks 🗒️/archived_experiment_data/Harry Potter and the Sorcerer's Stone.json"
        )
    )

    log.info("Begin persisting document")
    executor.persist(structured=structured_document, chunks=chunks)
    log.info("Finished persisting document")

    env.open_browser()
    input("Press Enter to exit the program...")


if __name__ == "__main__":
    main()
