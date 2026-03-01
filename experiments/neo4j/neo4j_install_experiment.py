from datetime import datetime
from pathlib import Path

from common.logging.run_logger import RunLogger
from common.neo4j.db_installer import DbInstaller
from common.neo4j.neo4j_environment import Neo4JEnvironment
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.encoding import MiniLMMeanPoolingEncoder

EXPERIMENT_NAME = "Neo4J Installation"

def main() -> None:
    # logging
    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    log_name = f"{EXPERIMENT_NAME}_{timestamp}"
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log = RunLogger(run_id=log_name, log_dir=log_dir)

    # neo4j
    env = Neo4JEnvironment(log=log)
    executor = StandardExecutor(env=env, encoder=MiniLMMeanPoolingEncoder())
    installer = DbInstaller()

    installer.installDB(env=env, executor=executor, log=log)

if __name__ == "__main__":
    main()
