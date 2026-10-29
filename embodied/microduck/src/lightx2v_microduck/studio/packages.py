"""Local skill bundles. No caches/data/checkpoint history; strict safe extraction."""

import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from .schema import file_hash, write_json, digest


def export_bundle(store, version_id, *, offline=False, dependency_root=None):
    v = store.version(version_id)
    if not v:
        raise KeyError("unknown skill version")
    if v.get("historical"):
        raise ValueError("historical publication is downloaded from its pinned reference")
    model = (store.root / v["model_path"]).resolve()
    if store.root not in model.parents:
        raise ValueError("model path escapes workspace")
    if file_hash(model / "model.json") != v["manifest_sha256"]:
        raise ValueError("published manifest changed")
    spec = json.loads((model / "model.json").read_text())
    for name in spec["files"]:
        if model not in (model / name).resolve().parents:
            raise ValueError("model member path escapes model directory")
    files = {"model/" + n: model / n for n in ["model.json", *spec["files"]]}
    if offline and spec["backend"] == "fastwam":
        if dependency_root is None:
            raise ValueError("offline FastWAM export requires a mounted backbone")
        lock = json.loads((Path(__file__).resolve().parents[5] / "deploy/microduck/fastwam/model-files.lock.json").read_text())
        for name, meta in lock["files"].items():
            if name.startswith("models/Wan2.2-TI2V-5B/"):
                path = Path(dependency_root) / name
                if file_hash(path) != meta["sha256"]:
                    raise ValueError("offline backbone checksum mismatch")
                files["dependencies/" + name] = path
    for name, sha in spec["files"].items():
        if file_hash(model / name) != sha:
            raise ValueError("model bytes changed")
    folder = store.root / "exports"
    folder.mkdir(exist_ok=True)
    target = folder / (version_id + ("-offline" if offline else "") + ".tar")
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        from .environments import export_environment

        manifest = {
            "schema": "studio_skill_package_v1",
            "skill": v["skill_snapshot"],
            "environment": export_environment(v["skill_snapshot"]["scene"]),
            "version": {k: value for k, value in v.items() if k not in ("id", "skill_id", "model_path")},
            "files": {n: {"sha256": file_hash(p), "bytes": p.stat().st_size} for n, p in files.items()},
            "backbone": {"repo": "lightx2v/microduck", "revision": "dad4beaf998780df1f7c17aba46c63ecf2a4e72d"} if spec["backend"] == "fastwam" else None,
            "offline": offline,
            "hardware_validated": False,
        }
        write_json(Path(temp) / "skill.json", manifest)
        with tarfile.open(str(target) + ".tmp", "w") as tar:
            tar.add(Path(temp) / "skill.json", arcname="skill.json")
            for name, path in files.items():
                tar.add(path, arcname=name, recursive=False)
        Path(str(target) + ".tmp").replace(target)
    return target


def import_bundle(store, path):
    folder = store.root / "imports"
    folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        dest = Path(temp)
        with tarfile.open(path) as tar:
            members = tar.getmembers()
            if len(members) > 1000 or sum(m.size for m in members) > 40 * 1024**3:
                raise ValueError("skill package too large")
            names = set()
            for member in members:
                target = (dest / member.name).resolve()
                if not member.isfile() or dest not in target.parents or member.name in names:
                    raise ValueError("package paths, links or duplicate entries are unsafe")
                names.add(member.name)
            if "skill.json" not in names:
                raise ValueError("missing skill manifest")
            manifest = json.load(tar.extractfile("skill.json"))
            if manifest["schema"] != "studio_skill_package_v1" or names != {"skill.json", *manifest["files"]}:
                raise ValueError("unlisted package contents")
            for member in members:
                target = dest / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
        for name, meta in manifest["files"].items():
            p = dest / name
            if p.stat().st_size != meta["bytes"] or file_hash(p) != meta["sha256"]:
                raise ValueError("package checksum mismatch")
        from .schema import OBSERVATION, ACTION

        model = dest / "model"
        model_spec = json.loads((model / "model.json").read_text())
        if model_spec["observation"] != OBSERVATION or model_spec["action"] != ACTION or model_spec["horizon"] != 32:
            raise ValueError("unsupported model contracts")
        if manifest["version"]["backend"] != model_spec["backend"]:
            raise ValueError("package backend mismatch")
        for name, sha in model_spec["files"].items():
            member = (model / name).resolve()
            if model not in member.parents or file_hash(member) != sha:
                raise ValueError("model member checksum/path mismatch")
        # Imported third-party evidence remains unverified; never inherit an accepted badge.
        spec = manifest["skill"].copy()
        if "environment" in manifest:
            from .environments import import_environment

            if import_environment(manifest["environment"]) != spec["scene"]:
                raise ValueError("skill/environment mismatch")
        spec.pop("id", None)
        skill = store.save_skill(spec)
        identifier = digest(manifest)
        target = folder / identifier
        if not target.exists():
            shutil.copytree(dest, target)
        version = {
            **manifest["version"],
            "status": "experimental",
            "imported_evidence": True,
            "model_path": str((target / "model").relative_to(store.root)),
            "manifest_sha256": file_hash(target / "model/model.json"),
            "skill_snapshot": skill,
        }
        if manifest.get("offline") and manifest.get("backbone"):
            version["dependency_root"] = str((target / "dependencies").relative_to(store.root))
        return store.add_version(skill["id"], version)
