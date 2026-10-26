"""Pinned upstream acquisition. Importing this module never downloads anything."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RESOURCE_DIR = Path(__file__).parent / "resources"
LOCK_PATH = RESOURCE_DIR / "upstream.lock.json"


def resolve_asset_root(root=None) -> Path:
    return Path(root or os.environ.get("MICRODUCK_ASSET_ROOT", "~/.cache/lightx2v-microduck/upstream")).expanduser().resolve()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_lock(path=None):
    return json.loads(Path(path or LOCK_PATH).read_text())


def receipt_tree_digest(receipt):
    payload = {"files": receipt["files"], "symlinks": receipt.get("symlinks", {})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _download(url: str, target: Path, expected: str | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": "lightx2v-microduck/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    actual = sha256_file(target)
    if expected and actual != expected:
        raise ValueError(f"checksum mismatch for {target.name}: expected {expected}, got {actual}")
    return actual


def fetch_source(source_id: str, root=None, *, lock_path=None) -> dict:
    lock = load_lock(lock_path)
    source = lock["sources"].get(source_id)
    if source is None:
        raise ValueError(f"unknown source {source_id}; available: {', '.join(lock['sources'])}")
    base = resolve_asset_root(root)
    base.mkdir(parents=True, exist_ok=True)
    dest = base / source_id
    receipt = dest / ".asset-receipt.json"
    if receipt.exists():
        checked = verify_source(source_id, base, lock_path=lock_path)
        if not checked["ok"]:
            raise ValueError(f"existing asset is modified or stale: {checked}; move it aside before fetching")
        return checked
    if dest.exists():
        raise ValueError(f"refusing to overwrite unverified asset directory {dest}")
    with tempfile.TemporaryDirectory(prefix=f".{source_id}-", dir=base) as temp:
        temp = Path(temp)
        staged = temp / "staged"
        staged.mkdir()
        archive_hash = None
        if source["kind"] == "github_archive":
            archive = temp / "source.tar.gz"
            archive_hash = _download(source["url"], archive, source.get("archive_sha256"))
            unpacked = temp / "unpacked"
            unpacked.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(unpacked, filter="data")
            children = list(unpacked.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise ValueError("upstream archive must contain one root directory")
            shutil.copytree(children[0], staged, dirs_exist_ok=True, symlinks=True)
        elif source["kind"] == "files":
            for spec in source["files"]:
                relative = Path(spec["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe asset file path")
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                _download(spec["url"], target, spec.get("sha256"))
        else:
            raise ValueError(f"unsupported asset source kind {source['kind']}")
        hashes = {str(p.relative_to(staged)): sha256_file(p) for p in sorted(staged.rglob("*")) if p.is_file() and not p.is_symlink()}
        links = {str(p.relative_to(staged)): os.readlink(p) for p in staged.rglob("*") if p.is_symlink()}
        result = {"source": source_id, "revision": source["revision"], "archive_sha256": archive_hash, "files": hashes, "symlinks": links, "license": source.get("license"), "schema_version": 1}
        (staged / ".asset-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        staged.rename(dest)
    return verify_source(source_id, base, lock_path=lock_path)


def verify_source(source_id: str, root=None, *, lock_path=None) -> dict:
    base = resolve_asset_root(root) / source_id
    source = load_lock(lock_path)["sources"][source_id]
    receipt_path = base / ".asset-receipt.json"
    if not receipt_path.exists():
        return {"source": source_id, "ok": False, "errors": ["missing receipt; use assets fetch"]}
    receipt = json.loads(receipt_path.read_text())
    errors = []
    if receipt["revision"] != source["revision"]:
        errors.append("revision mismatch")
    if source.get("archive_sha256") and receipt.get("archive_sha256") != source["archive_sha256"]:
        errors.append("archive digest mismatch")
    expected_files = {spec["path"]: spec.get("sha256") for spec in source.get("files", [])}
    if not set(expected_files).issubset(receipt["files"]):
        errors.append("receipt omits required files")
    if source.get("file_tree_sha256") and receipt_tree_digest(receipt) != source["file_tree_sha256"]:
        errors.append("file manifest differs from locked upstream tree")
    for name, expected in receipt["files"].items():
        path = base / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"missing or modified: {name}")
        if expected_files.get(name) and expected != expected_files[name]:
            errors.append(f"file digest differs from lock: {name}")
    for name, target in receipt.get("symlinks", {}).items():
        path = base / name
        if not path.is_symlink() or os.readlink(path) != target:
            errors.append(f"modified symlink: {name}")
    return {"source": source_id, "revision": receipt["revision"], "ok": not errors, "files": len(receipt["files"]), "errors": errors, "root": str(base)}


def verify_assets(root=None, sources=None, *, lock_path=None) -> dict:
    lock = load_lock(lock_path)
    names = sources or lock["default_sources"]
    results = [verify_source(name, root, lock_path=lock_path) for name in names]
    return {"ok": all(r["ok"] for r in results), "sources": results, "hardware_validated": False}
