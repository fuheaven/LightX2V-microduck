"""Transactional queue with fenced leases. Expired jobs require explicit resume."""

import contextlib
import json
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .schema import SkillDraft, RunRequest, canonical, digest, write_json

TERMINAL = {"complete", "failed", "blocked", "cancelled", "interrupted"}


class Store:
    def __init__(self, root):
        if (3, 51, 0) <= sqlite3.sqlite_version_info < (3, 51, 2):
            raise RuntimeError("SQLite 3.51.0/3.51.1 has a Unix locking deadlock; use a patched SQLite runtime (recommended 3.51.3 or newer) or the Studio Web container.")
        self.root = Path(root).resolve()
        # One Web process owns SQLite. Serialize short transactions, including
        # connection close/checkpoint, across its request threads.
        self._db_lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        # Keep the WAL connection alive instead of forcing a last-connection
        # checkpoint/close on every heartbeat. All access is fenced by the lock.
        self._db = sqlite3.connect(self.root / "studio.sqlite3", timeout=20, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY, body TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, snapshot TEXT NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, worker TEXT,
                    lease TEXT, heartbeat REAL, cancel INTEGER DEFAULT 0, progress TEXT DEFAULT '{}',
                    result TEXT DEFAULT '{}', idempotency TEXT UNIQUE);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, at REAL, body TEXT);
                CREATE TABLE IF NOT EXISTS workers(id TEXT PRIMARY KEY, body TEXT NOT NULL, heartbeat REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, body TEXT NOT NULL, created REAL);
                CREATE TABLE IF NOT EXISTS gpu_allocations(uuid TEXT PRIMARY KEY, run_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS video_reviews(seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, at REAL, body TEXT);
            """)

    @contextlib.contextmanager
    def connect(self):
        with self._db_lock:
            db = self._db
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise

    def close(self):
        with self._db_lock:
            self._db.close()

    def save_skill(self, body, skill_id=None):
        body = SkillDraft.model_validate(body).model_dump()
        from .scene import validate_scene

        validate_scene(body["scene"])
        key = skill_id or uuid.uuid4().hex[:12]
        with self.connect() as db:
            db.execute("INSERT INTO skills VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body,updated=excluded.updated", (key, canonical(body), time.time()))
        return {"id": key, **body}

    def skills(self):
        with self.connect() as db:
            return [{"id": r["id"], **json.loads(r["body"])} for r in db.execute("SELECT * FROM skills ORDER BY updated DESC")]

    def apply_scene_draft(self, skill_id, scene, base_sha):
        from .scene import validate_scene
        scene = validate_scene(scene)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM skills WHERE id=?", (skill_id,)).fetchone()
            if not row:
                raise KeyError("unknown skill")
            body = json.loads(row["body"])
            if digest(body["scene"]) != base_sha:
                raise ValueError("原场景已被编辑，请基于新版本重新生成草案")
            body["scene"] = scene
            db.execute("UPDATE skills SET body=?,updated=? WHERE id=?", (canonical(body), time.time(), skill_id))
        return {"id": skill_id, **body}

    def skill(self, key):
        return next((s for s in self.skills() if s["id"] == key), None)

    def submit(self, skill_id, request, provenance):
        skill = self.skill(skill_id)
        if skill is None:
            raise KeyError("unknown skill")
        req = RunRequest.model_validate(request).model_dump()
        if req["kind"] in ("scene_draft", "video_enhance") and not req["prompt"].strip():
            raise ValueError("请输入描述")
        if req["kind"] in ("train",) and not req["data_run"]:
            raise ValueError("training requires a frozen dataset run")
        if req["kind"] == "collect" and not req["teacher_run"]:
            raise ValueError("collection requires a qualified teacher run")
        if req["kind"] in ("evaluate", "optimize") and req["policy"] in ("fastwam", "lightweight") and not req["version"]:
            raise ValueError("evaluation requires a frozen model version")
        dependencies = {}
        for field in ("teacher_run", "data_run", "resume_run"):
            if req[field]:
                run = self.run(req[field])
                if not run or run["skill_id"] != skill_id or run["status"] not in TERMINAL:
                    raise ValueError(f"{field} must be a finished run of this skill")
                if field != "resume_run" and run["status"] != "complete":
                    raise ValueError(f"{field} must have completed successfully")
                if any(run["snapshot"]["skill"][k] != skill[k] for k in ("task", "scene")):
                    raise ValueError(f"{field} belongs to a different frozen task or scene")
                if field == "teacher_run" and not run["result"].get("teacher_qualified"):
                    raise ValueError("teacher gate has not passed")
                dependencies[field] = {"id": run["id"], "snapshot_sha256": digest(run["snapshot"]), "result": run["result"]}
        version = self.version(req["version"]) if req["version"] else None
        if req["version"] and (not version or version["skill_id"] != skill_id):
            raise ValueError("model version does not belong to this skill")
        if version and version["backend"] != req["policy"]:
            raise ValueError("model version and policy backend differ")
        if version and version.get("task_contract") != skill["task"]:
            raise ValueError("task contract changed since publication; create and validate a new model version")
        snap = {"schema": "studio_run_v1", "skill": skill, "request": req, "provenance": provenance, "dependencies": dependencies, "model_version": version}
        from .assets import asset_identity

        snap["environment"] = {"scene_sha256": digest(skill["scene"]), "assets_sha256": asset_identity(skill["scene"])}
        if req["kind"] == "video_enhance":
            from .video import freeze_source

            snap["video_source"] = freeze_source(self, skill_id, req["video"])
        key = uuid.uuid4().hex[:16]
        idem = f"{skill_id}:{req['idempotency_key']}" if req["idempotency_key"] else None
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idem:
                row = db.execute("SELECT * FROM runs WHERE idempotency=?", (idem,)).fetchone()
                if row:
                    if json.loads(row["snapshot"]) != snap:
                        raise ValueError("idempotency key reused for different inputs")
                    return self.decode(row)
            db.execute("INSERT INTO runs(id,skill_id,snapshot,status,created,updated,idempotency) VALUES(?,?,?,?,?,?,?)", (key, skill_id, canonical(snap), "queued", now, now, idem))
            write_json(self.root / "runs" / key / "snapshot.json", snap)
        return self.run(key)

    @staticmethod
    def decode(row):
        if row is None:
            return None
        out = dict(row)
        for k in ("snapshot", "progress", "result"):
            out[k] = json.loads(out[k])
        out.pop("lease", None)
        return out

    def run(self, key):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM runs WHERE id=?", (key,)).fetchone())

    def runs(self, skill_id=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs WHERE (? IS NULL OR skill_id=?) ORDER BY created DESC LIMIT 500", (skill_id, skill_id))
            return [self.decode(row) for row in rows]

    def cancel(self, key):
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated=? WHERE id=? AND status NOT IN ('complete','failed','blocked','cancelled','interrupted')",
                (time.time(), key),
            )
        return self.run(key)

    def claim(self, worker, capabilities, provenance):
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO workers VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body,heartbeat=excluded.heartbeat",
                (worker, canonical({"capabilities": capabilities, **provenance}), now),
            )
            # Do not requeue after a lease expires: the old GPU process may still exist.
            db.execute("UPDATE runs SET status='interrupted',lease=NULL,updated=? WHERE status='running' AND heartbeat<?", (now, now - 90))
            if db.execute("SELECT 1 FROM runs WHERE worker=? AND status='running'", (worker,)).fetchone():
                return None
            if provenance.get("device_busy"):
                return None
            for row in db.execute("SELECT * FROM runs WHERE status='queued' ORDER BY created"):
                snap = json.loads(row["snapshot"])
                req = snap["request"]
                video = req["kind"] == "video_enhance"
                if video != ("video" in capabilities):
                    continue
                gpu = video or req["kind"] in ("train", "pipeline", "optimize", "historical") or (req["kind"] in ("evaluate", "live", "correction") and req["policy"] in ("fastwam", "vlm_local"))
                if gpu and "gpu" not in capabilities:
                    continue
                if "gpu" in capabilities and not gpu:
                    continue
                if snap["provenance"]["source_sha256"] != provenance.get("source_sha256"):
                    continue
                expected_image = snap["provenance"].get("video_image") if video else snap["provenance"]["worker_image"]
                if expected_image != provenance.get("image"):
                    continue
                if gpu:
                    device = provenance.get("gpu_uuid")
                    if not device or not provenance.get("exclusive_device_lock"):
                        continue
                    allocation = db.execute("SELECT runs.status FROM gpu_allocations JOIN runs ON runs.id=run_id WHERE uuid=?", (device,)).fetchone()
                    if allocation and allocation["status"] == "running":
                        continue
                    # The inherited OS lock proves any expired previous child has exited.
                    db.execute("INSERT INTO gpu_allocations VALUES(?,?) ON CONFLICT(uuid) DO UPDATE SET run_id=excluded.run_id", (device, row["id"]))
                lease = secrets.token_hex(24)
                db.execute("UPDATE runs SET status='running',worker=?,lease=?,heartbeat=?,updated=? WHERE id=?", (worker, lease, now, now, row["id"]))
                return {**self.decode(row), "lease": lease, "status": "running"}
        return None

    def update(self, key, lease, *, progress=None, status=None, result=None):
        if status is not None and status not in TERMINAL:
            raise ValueError("invalid terminal status")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=? AND lease=? AND status='running'", (key, lease)).fetchone()
            if not row:
                raise ValueError("stale worker lease; stop local execution")
            now = time.time()
            db.execute("UPDATE workers SET heartbeat=? WHERE id=?", (now, row["worker"]))
            db.execute(
                "UPDATE runs SET heartbeat=?,updated=?,progress=?,status=?,result=? WHERE id=?",
                (now, now, canonical(progress if progress is not None else json.loads(row["progress"])), status or "running", canonical(result or json.loads(row["result"])), key),
            )
            if progress is not None and canonical(progress) != row["progress"]:
                db.execute("INSERT INTO events(run_id,at,body) VALUES(?,?,?)", (key, now, canonical(progress)))
            if status in TERMINAL:
                db.execute("DELETE FROM gpu_allocations WHERE run_id=?", (key,))
            return {"cancel": bool(row["cancel"])}

    def video_reviews(self, run_id):
        with self.connect() as db:
            return [{"id": r["seq"], "at": r["at"], **json.loads(r["body"])} for r in db.execute("SELECT * FROM video_reviews WHERE run_id=? ORDER BY seq", (run_id,))]

    def add_video_review(self, run_id, body):
        with self.connect() as db:
            db.execute("INSERT INTO video_reviews(run_id,at,body) VALUES(?,?,?)", (run_id, time.time(), canonical(body)))
        return self.video_reviews(run_id)[-1]

    def workers(self):
        with self.connect() as db:
            return [{"id": r["id"], "heartbeat": r["heartbeat"], "online": time.time() - r["heartbeat"] < 90, **json.loads(r["body"])} for r in db.execute("SELECT * FROM workers")]

    def versions(self, skill_id=None):
        with self.connect() as db:
            return [
                {"id": r["id"], "skill_id": r["skill_id"], **json.loads(r["body"])}
                for r in db.execute("SELECT * FROM versions WHERE (? IS NULL OR skill_id=?) ORDER BY created DESC", (skill_id, skill_id))
            ]

    def version(self, key):
        return next((v for v in self.versions() if v["id"] == key), None)

    def add_version(self, skill_id, body):
        key = digest({"skill_id": skill_id, **body})[:20]
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?,?)", (key, skill_id, canonical(body), time.time()))
        return self.version(key)
