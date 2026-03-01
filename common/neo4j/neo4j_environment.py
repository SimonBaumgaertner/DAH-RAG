import atexit
import os
import platform
import shutil
import subprocess
import tarfile
import time
import webbrowser
from pathlib import Path
from typing import Final, Optional

from neo4j import GraphDatabase, basic_auth, Driver
from common.logging.run_logger import RunLogger


class Neo4JEnvironment:
    """Self-contained Neo4j Community + portable Temurin JRE installer/launcher."""

    # ---------- Pin your artifacts here ----------
    NEO4J_COMMUNITY_2025_08_TAR = f"{Path(__file__).parent.resolve()}/tar/neo4j-community-2025.08.0-unix.tar.gz"
    TEMURIN_JRE21              = f"{Path(__file__).parent.resolve()}/tar/OpenJDK21U-jre_x64_linux_hotspot_21.0.8_9.tar.gz"
    GDS_JAR                    = f"{Path(__file__).parent.resolve()}/tar/neo4j-graph-data-science-2.21.0.jar"

    _NEO4J_DIRNAME: Final[str] = "neo4j"
    _JAVA_DIRNAME: Final[str]  = "jre"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def __init__(self, *, log: RunLogger, data_root: Optional[Path] = None, nuke_on_start: bool = True):
        """
        :param log: logger
        :param data_root: optional external data root; defaults to ~/.neo4j-data/masters-thesis
        :param nuke_on_start: if True, removes databases/neo4j and transactions/neo4j on every start
        """
        self.driver: Optional[Driver] = None
        self.log = log
        self._neo4j_proc: subprocess.Popen | None = None
        self._neo4j_home: Path | None = None
        self._java_home: Path | None = None
        self._data_root: Path = Path(data_root).expanduser().resolve() if data_root else (
            Path.home() / ".neo4j-data" / "masters-thesis"
        )
        self._logs_root: Path = self._data_root / "logs"
        self._nuke_on_start: bool = bool(nuke_on_start)

        log.info("📦 Installing self-contained Neo4j environment (binaries in repo, data external)")
        self._install_environment_from_tar(
            self.NEO4J_COMMUNITY_2025_08_TAR,
            self.TEMURIN_JRE21,
        )
        log.info("✅ Done installing local Neo4j")

    # ---------------- Runtime helpers ----------------
    def check_connection(self, overwrite_nuke=False) -> Driver:
        """Ensure database is running and return a verified Bolt driver."""
        if self._neo4j_proc is None or self._neo4j_proc.poll() is not None:
            # Ensure external data & logs directories exist
            (self._data_root / "databases").mkdir(parents=True, exist_ok=True)
            (self._data_root / "transactions").mkdir(parents=True, exist_ok=True)
            self._logs_root.mkdir(parents=True, exist_ok=True)

            # Nuke the store each start (preserve dbms/auth)
            if self._nuke_on_start and not overwrite_nuke:
                db_dir = self._data_root / "databases" / "neo4j"
                tx_dir = self._data_root / "transactions" / "neo4j"
                for path in (db_dir, tx_dir):
                    if path.exists():
                        shutil.rmtree(path)
                self.log.info("🧹 Fresh start: removed 'databases/neo4j' and 'transactions/neo4j'")

            # Launch Neo4j
            launch = self._neo4j_home / "run-neo4j.sh"
            log_file = open(self._logs_root / "neo4j-console.out", "ab", buffering=0)
            self._neo4j_proc = subprocess.Popen(
                [launch, "console"],
                stdout=log_file,
                stderr=log_file,
                env=self._neo4j_env(),
            )
            atexit.register(self._stop_db)

            self.log.info("⏳ Waiting up to 180 seconds to ensure Neo4J is up and running!")
            deadline = time.time() + 180  # up to 180s
            last_err = None
            while time.time() < deadline:
                try:
                    driver = GraphDatabase.driver("bolt://127.0.0.1:7687",
                                                  auth=basic_auth("neo4j", "superSecret!"))
                    driver.verify_connectivity()
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(2)
            else:
                raise RuntimeError(f"Neo4j did not become ready: {last_err}")

        driver: Driver = GraphDatabase.driver(
            "bolt://127.0.0.1:7687",
            auth=basic_auth("neo4j", "superSecret!"),
        )

        driver.verify_connectivity()
        info = driver.get_server_info()
        self.log.info(f"✅ Connected to {info.address} running Neo4j")
        self.driver = driver

        # --- NEW: verify GDS loaded (helpful early signal) ---
        try:
            with driver.session() as s:
                v = s.run("RETURN gds.version() AS v").single()
                if v and v["v"]:
                    self.log.info(f"🧠 GDS available: {v['v']}")
        except Exception as e:
            self.log.warning(f"⚠️ GDS not available yet (that's ok if you skipped the plugin). Detail: {e}")

        return driver

    def clean_db(self) -> None:
        """Remove the default *neo4j* database store so the next start is empty (external data root)."""
        self._stop_db()
        db_dir = self._data_root / "databases" / "neo4j"
        tx_dir = self._data_root / "transactions" / "neo4j"
        for path in (db_dir, tx_dir):
            if path.exists():
                shutil.rmtree(path)
        self.log.info("🗑️ Neo4j store removed – clean start ensured")

    def export_to_file(self, dump_file: Path) -> None:
        """
        Export the current database to a dump file at the given path.
        Example: export_to_file(Path("backups/NovelQA.dump"))
        """
        self._stop_db()
        dump_file = Path(dump_file)
        dump_file.parent.mkdir(parents=True, exist_ok=True)

        # Neo4j will always write neo4j.dump in the target folder
        subprocess.run([
            self._neo4j_home / "bin" / "neo4j-admin", "database", "dump", "neo4j",
            f"--to-path={dump_file.parent}", "--overwrite-destination=true"
        ], check=True, env=self._neo4j_env())

        # Rename/move the produced neo4j.dump to the user-specified filename
        produced = dump_file.parent / "neo4j.dump"
        produced.replace(dump_file)

        self.log.info(f"💾 Database dumped to {dump_file}")

    def import_from_file(self, dump_file: Path, *, restart: bool = True) -> None:
        """
        Import a database dump from a file at the given path.
        Example: import_from_file(Path("backups/NovelQA.dump"))
        """
        self._stop_db()
        dump_file = Path(dump_file)
        if not dump_file.is_file():
            raise FileNotFoundError(dump_file)

        target_dir = dump_file.parent
        temp = target_dir / "neo4j.dump"

        # Copy (not rename) to neo4j.dump so the original stays intact
        if temp.exists():
            temp.unlink()
        shutil.copy(dump_file, temp)

        subprocess.run([
            self._neo4j_home / "bin" / "neo4j-admin", "database", "load", "neo4j",
            f"--from-path={target_dir}", "--overwrite-destination=true"
        ], check=True, env=self._neo4j_env())

        self.log.info(f"📥 Database loaded from {dump_file}")

        if restart:
            self.check_connection(overwrite_nuke=True)

    def _stop_db(self) -> None:
        if self._neo4j_proc and self._neo4j_proc.poll() is None:
            try:
                self._neo4j_proc.terminate()
                self._neo4j_proc.wait(timeout=120)
            except Exception:
                self._neo4j_proc.kill()
            finally:
                self._neo4j_proc = None
                self.log.info("🛑 Neo4j stopped")

    def open_browser(self) -> None:
        url = "http://localhost:7474"
        self.log.info(f"🌐 Opening Neo4j Browser at {url}")
        try:
            webbrowser.open(url)
        except Exception as exc:
            self.log.warning(f"⚠️ Could not open browser automatically: {exc}. Please navigate to {url} manually.")

    # ---------------- Installation helpers ----------------
    def _neo4j_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["JAVA_HOME"] = str(self._java_home)
        env["PATH"] = f"{self._java_home}/bin:" + env.get("PATH", "")
        env["NEO4J_HOME"] = str(self._neo4j_home)
        return env

    def _install_environment_from_tar(
        self,
        neo4j_community_tar_path: str,
        temurin_jre_tar_path: str,
        target_root: Path | None = None,
    ) -> None:
        """Unpack Neo4j + JRE; write conf to point DATA/LOGS to external dirs; set password; then install GDS."""
        target_root = Path(target_root or Path(__file__).parent / "neo4j_env")
        neo4j_home = target_root / self._NEO4J_DIRNAME
        java_home = neo4j_home / self._JAVA_DIRNAME

        def _safe_extract(tar_path: Path, dest: Path) -> Path:
            dest.mkdir(parents=True, exist_ok=True)
            pre_existing = {p.name for p in dest.iterdir() if p.is_dir()}
            with tarfile.open(tar_path) as tf:
                for m in tf.getmembers():
                    target = dest / m.name
                    if not target.resolve().is_relative_to(dest):
                        raise RuntimeError(f"Unsafe path in tar: {m.name}")
                tf.extractall(dest, filter="data")
            post_dirs = [p for p in dest.iterdir() if p.is_dir() and p.name not in pre_existing]
            if len(post_dirs) != 1:
                raise RuntimeError(f"{tar_path} should create exactly one dir, found {post_dirs}")
            return post_dirs[0]

        # Unpack Neo4j
        if neo4j_home.exists():
            self.log.info("📦 Neo4j already present – skipping unpack")
        else:
            root = _safe_extract(Path(neo4j_community_tar_path), target_root)
            root.rename(neo4j_home)
            self.log.info(f"📦 Neo4j unpacked to {neo4j_home}")

        # Unpack JRE
        if java_home.exists():
            self.log.info("📦 JRE already present – skipping unpack")
        else:
            root = _safe_extract(Path(temurin_jre_tar_path), neo4j_home)
            root.rename(java_home)
            self.log.info(f"📦 Temurin JRE unpacked to {java_home}")

        # Remember locations
        self._neo4j_home = neo4j_home
        self._java_home = java_home

        # Arch + Java sanity (non-fatal warnings)
        try:
            arch = platform.machine()
            if "x86_64" not in arch and "amd64" not in arch:
                self.log.warning(f"⚠️ Host arch is {arch}. Ensure your JRE tar matches the host (you provided x64).")
            java_ver = self._read_java_major(java_home)
            if java_ver and java_ver < 21:
                self.log.warning(f"⚠️ Detected Java {java_ver}; Neo4j 2025.x requires Java 21.")
        except Exception as e:
            self.log.debug(f"Java/arch check skipped ({e})")

        # Ensure external dirs exist
        (self._data_root / "databases").mkdir(parents=True, exist_ok=True)
        (self._data_root / "transactions").mkdir(parents=True, exist_ok=True)
        self._logs_root.mkdir(parents=True, exist_ok=True)

        # --- Write conf BEFORE any plugin install ---
        conf = neo4j_home / "conf" / "neo4j.conf"
        self._write_or_update_conf(conf, {
            # ✅ correct keys for Neo4j 5/2025.x
            "server.directories.data": str(self._data_root),
            "server.directories.logs": str(self._logs_root),

            # network
            "server.default_listen_address": "127.0.0.1",
            "server.bolt.listen_address": "127.0.0.1:7687",
            "server.bolt.advertised_address": "127.0.0.1:7687",

            # GDS permissions (names unchanged in 5.x)
            "dbms.security.procedures.unrestricted": "gds.*",
            "dbms.security.procedures.allowlist": "gds.*",
        })

        # --- IMPORTANT: set initial password BEFORE copying plugins ---
        auth_file = self._data_root / "dbms" / "auth"
        if not auth_file.exists():
            subprocess.run(
                [
                    neo4j_home / "bin" / "neo4j-admin",
                    "dbms",
                    "set-initial-password",
                    "superSecret!",
                ],
                check=True,
                env={**os.environ, "JAVA_HOME": str(java_home), "NEO4J_HOME": str(neo4j_home)},
            )
            self.log.info("🔑 Initial password set for user 'neo4j'")
        else:
            self.log.debug("🔒 Password already initialised – skipping")

        # --- NOW install the GDS plugin jar (so a bad jar can't brick admin) ---
        plugins_dir = neo4j_home / "plugins"
        plugins_dir.mkdir(exist_ok=True)
        gds_src = Path(self.GDS_JAR)
        if gds_src.exists():
            gds_dst = plugins_dir / gds_src.name
            if not gds_dst.exists():
                shutil.copy2(gds_src, gds_dst)
                self.log.info(f"🔌 GDS plugin installed: {gds_dst}")
        else:
            self.log.warning(f"⚠️ GDS jar not found at {gds_src}; install it to enable gds.*")

        # Helper launcher script
        launcher = neo4j_home / "run-neo4j.sh"
        if not launcher.exists():
            launcher.write_text(
                f"""#!/usr/bin/env bash
export JAVA_HOME="{java_home}"
export NEO4J_HOME="{neo4j_home}"
exec "{neo4j_home}/bin/neo4j" "$@"
"""
            )
            launcher.chmod(0o755)

    def get_driver(self) -> Driver:
        return self.check_connection() if self.driver is None else self.driver

    def is_database_populated(self) -> bool:
        """Check if the database contains any nodes (i.e., has been indexed)."""
        try:
            driver = self.get_driver()
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) AS node_count LIMIT 1")
                record = result.single()
                if record:
                    node_count = record["node_count"]
                    return node_count > 0
                return False
        except Exception as e:
            self.log.warning(f"⚠️ Could not check database population status: {e}")
            return False

    # ---------------- Utils ----------------
    def _write_or_update_conf(self, conf_path: Path, kv: dict[str, str]) -> None:
        existing = {}
        if conf_path.exists():
            for line in conf_path.read_text().splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                key, value = s.split("=", 1)
                existing[key.strip()] = value.strip()

        self.log.debug(f"🔍 Existing config keys: {list(existing.keys())}")
        self.log.debug(f"🔍 Updating with: {kv}")
        
        changed = False
        for k, v in kv.items():
            old_val = existing.get(k)
            if old_val != v:
                self.log.debug(f"🔍 Changing {k}: '{old_val}' -> '{v}'")
                existing[k] = v
                changed = True
            else:
                self.log.debug(f"🔍 Keeping {k}: '{v}' (unchanged)")

        if changed or not conf_path.exists():
            lines = [f"{k}={v}" for k, v in existing.items()]
            conf_path.parent.mkdir(parents=True, exist_ok=True)
            conf_path.write_text("\n".join(lines) + "\n")
            self.log.info(f"📝 Updated {conf_path} with external data/logs configuration")

    @staticmethod
    def _read_java_major(java_home: Path) -> Optional[int]:
        """Return detected Java major version (e.g., 21) or None if unknown."""
        try:
            out = subprocess.check_output(
                [str(java_home / "bin" / "java"), "-version"],
                stderr=subprocess.STDOUT
            ).decode("utf-8", "ignore")
            # lines like: 'openjdk version "21.0.8" 2025-07-15'
            for tok in out.replace("(", " ").replace(")", " ").replace("\n", " ").split():
                if tok.startswith('"') and tok.endswith('"') and tok.count(".") >= 1:
                    major = tok.strip('"').split(".")[0]
                    return int(major)
        except Exception:
            return None
        return None
