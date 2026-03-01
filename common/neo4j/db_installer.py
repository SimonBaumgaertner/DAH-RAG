from neo4j import Query

from common.logging.run_logger import RunLogger
from common.neo4j.data_classes import Executor
from common.neo4j.neo4j_environment import Neo4JEnvironment


class DbInstaller:
    """
    Applies the schema with managed write transactions and logs progress.
    Each string is a single Cypher statement; do not include multiple statements per string.
    """
    def installDB(self, env: Neo4JEnvironment, executor: Executor, log: RunLogger):
        schema = executor.get_installation_schema()  # List[str]
        driver = env.get_driver()  # neo4j.Driver

        def _apply_stmt(tx, stmt: str):
            res = tx.run(stmt)
            # fully consume to surface errors and get summary/notifications
            summary = res.consume()
            return summary

        # Apply schema statements (write or read depending on command)
        with driver.session() as session:
            for stmt in schema:
                try:
                    # CALL db.awaitIndexes() is READ; everything else is DDL/WRITE.
                    is_await = stmt.strip().upper().startswith("CALL DB.AWAITINDEXES")
                    if is_await:
                        summary = session.execute_read(_apply_stmt, stmt)
                    else:
                        summary = session.execute_write(_apply_stmt, stmt)

                    # Useful logging
                    counters = summary.counters
                    note_count = len(summary.notifications) if summary.notifications else 0
                    log.info(
                        f"🧾 Applied: {stmt.splitlines()[0][:120]} "
                        f"(updates: {counters}, notifications: {note_count})"
                    )
                except Exception as e:
                    # Constraints that already exist (without IF NOT EXISTS) or Enterprise-only
                    # type constraints on Community can raise errors. We log and proceed.
                    log.warning(f"⚠️ Schema statement failed (continuing): {stmt}\n  Error: {e}")

        # Optionally show what we have now
        try:
            with driver.session() as session:
                idx = session.run(Query("""
                    SHOW INDEXES YIELD name, type, entityType, state, createStatement
                    RETURN name, type, entityType, state, createStatement
                """)).data()

                con = session.run(Query("""
                    SHOW CONSTRAINTS YIELD name, type, entityType, properties, ownedIndex
                    RETURN name, type, entityType, properties, ownedIndex
                """)).data()
                log.info(f"📚 Online indexes: {len(idx)}")
                log.info(f"📚 Constraints: {len(con)}")
        except Exception as e:
            log.warning(f"⚠️ Introspection (SHOW INDEXES/CONSTRAINTS) failed: {e}")
