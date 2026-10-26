"""Qualified, complete teacher episodes from an additional scene distribution.

These are demonstrations from reset, not student-state DAgger corrections.
The original base release and its normalization remain immutable. A caller must
pin the additional release index; its complete qualification denominator,
physical replay and normalizer coverage are checked before exposing samples.
"""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .navigation_augmentation import contained, sha256, validate_tensors
from .navigation_dataset_metadata import read_indexed_metadata

KIND = "qualified_teacher_episodes_v1"
TASKS = {"find_red_ball", "find_blue_ball", "find_yellow_ball"}
SCENE = "room_curriculum_zero"
SPATIAL_KIND = "qualified_spatial_teacher_episodes_v1"
SPATIAL_SCENE = "room_curriculum_one"
RELEASE_SCENES = {KIND: SCENE, SPATIAL_KIND: SPATIAL_SCENE}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def _qualification_complete(report, scene):
    if scene == SCENE:
        return (report.get("status") == "complete" and report.get("scene_qualified") is True
                and report.get("audited_scenes") == [scene])
    return (scene == SPATIAL_SCENE and report.get("status") == "complete"
            and report.get("formal_teacher_qualification") is True
            and set(report.get("cohorts", {})) == {scene}
            and report["cohorts"][scene].get("formal_teacher_qualification") is True)


def validate_teacher_configuration(train, validation, audit):
    """Train/validation see the same release, filtered by immutable layout splits."""
    roots = train.get("qualified_teacher_roots", []) or []
    hashes = train.get("qualified_teacher_sha256", []) or []
    require(not isinstance(roots, (str, Path)) and not isinstance(hashes, str)
            and len(roots) == len(hashes), "teacher roots and index hashes must match")
    roots, hashes = list(roots), list(hashes)
    require(list(audit.get("qualified_teacher_index_sha256", []) or []) == hashes,
            "teacher data differs from the frozen training audit")
    require(list(validation.get("qualified_teacher_roots", []) or []) == roots
            and list(validation.get("qualified_teacher_sha256", []) or []) == hashes,
            "teacher validation must use the same releases with its held-out layout split")
    if roots:
        require(str(train["root"]) == str(validation["root"]), "teacher train/validation base artifacts differ")
    return roots, hashes


def verify_teacher_resume(receipt, hashes):
    require(receipt.get("qualified_teacher_index_sha256", []) == list(hashes),
            "resume belongs to another qualified teacher dataset; initialize a new stage")


def load_qualified_teacher_release(root, base_dataset, expected_index_sha256):
    """Return pinned full episodes; never fit statistics or alter a split."""
    from .wam_data import split_for_seed

    root, base = Path(root).resolve(), Path(base_dataset).resolve()
    require(isinstance(expected_index_sha256, str) and len(expected_index_sha256) == 64
            and sha256(root / "index.json") == expected_index_sha256,
            "additional teacher index changed or is not explicitly pinned")
    index = json.loads((root / "index.json").read_text())
    scene = index.get("scene")
    require(index.get("kind") in RELEASE_SCENES and RELEASE_SCENES[index["kind"]] == scene,
            "additional teacher release has an unsupported kind or scene")
    require(index.get("base_index_sha256") == sha256(base / "index.json")
            and index.get("base_qualification_sha256") == sha256(base / "qualification_report.json")
            and index.get("stats_sha256") == sha256(base / "dataset_stats.json")
            == sha256(root / "dataset_stats.json"), "additional teacher base or frozen normalization changed")
    base_qualification = json.loads((base / "qualification_report.json").read_text())
    require(base_qualification.get("teacher_qualified") is True
            and base_qualification.get("index_sha256") == index["base_index_sha256"]
            and base_qualification.get("stats_sha256") == index["stats_sha256"],
            "original teacher qualification is missing or belongs to another artifact")
    evidence = {}
    for name in ("qualification", "protocol", "physics_replay", "normalization"):
        ref = index["evidence"][name]; path = contained(root, ref["path"])
        require(sha256(path) == ref["sha256"], name + " evidence changed")
        evidence[name] = json.loads(path.read_text())
    qualification, protocol = evidence["qualification"], evidence["protocol"]
    require(_qualification_complete(qualification, scene)
            and qualification.get("protocol_sha256") == index["evidence"]["protocol"]["sha256"],
            "complete scene-specific teacher qualification is required")
    seeds = protocol["seeds"]
    require(len(seeds) == len(set(seeds)) == 100 and set(protocol["tasks"]) == TASKS
            and protocol.get("episodes_per_scene") == 300
            and protocol.get("minimum_eligible_success_rate_per_task") == .95
            and protocol.get("max_ticks") == 1500 and protocol.get("strict_stop_ticks") == 100
            and protocol.get("strict_stop_surface_m") == [.25, .4],
            "teacher qualification changed its cohort or acceptance criteria")
    require(not set(seeds).intersection(range(10000, 10200)), "reserved final layout entered teacher release")
    expected = {(s, t) for s in seeds for t in TASKS}
    records = qualification["records"]
    require(len(records) == 300 and {(r["seed"], r["task"]) for r in records} == expected
            and all(r["scene"] == scene for r in records)
            and len({r["layout_id"] for r in records}) == 100,
            "teacher qualification must retain every layout, color and failure")
    by_case = {(r["seed"], r["task"]): r for r in records}
    cohort = qualification["cohorts"][scene]
    for task in TASKS:
        successes = sum(r["success"] is True for r in records if r["task"] == task)
        saved = cohort["tasks"][task]
        require(successes >= 95 and saved["episodes"] == 100 and saved["strict_successes"] == successes
                and saved["success_rate"] == successes / 100 and saved["qualified"] is True,
                "a color failed the unchanged 95/100 teacher gate")
    collection_hash = cohort["evidence"][0]["sha256"]
    require(collection_hash == index["source_collection_sha256"], "teacher source collection changed")
    replay = evidence["physics_replay"]
    if scene == SPATIAL_SCENE:
        replay_cases = replay.get("cases", [])
        require(replay.get("collection_sha256") == collection_hash
                and replay.get("assessment_sha256") == index["evidence"]["qualification"]["sha256"]
                and replay.get("protocol_sha256") == index["evidence"]["protocol"]["sha256"]
                and isinstance(qualification.get("source_manifest_sha256"), str)
                and len(qualification["source_manifest_sha256"]) == 64
                and replay.get("source_manifest_sha256") == qualification["source_manifest_sha256"],
                "spatial replay belongs to another qualification or source")
        collection_replayed = True
    else:
        replay_cases = [r for r in replay.get("cases", []) if r.get("cohort") == "egl_qualification300"]
        collection_replayed = any(c["sha256"] == collection_hash and c["episodes"] == 300
                                  for c in replay["collections"])
    require(replay.get("status") == "complete" and len(replay_cases) == 300
            and {(r["seed"], r["task"]) for r in replay_cases} == expected
            and collection_replayed,
            "complete physics replay of the qualification collection is required")
    for case in replay_cases:
        original = by_case[case["seed"], case["task"]]
        require(case["metadata_sha256"] == original["metadata_sha256"]
                and case["strict_success"] == original["task_completion"]
                and case["exact_control_ticks"] == original["ticks"], "replay differs from qualified episode")
        if scene == SPATIAL_SCENE:
            ticks = case["exact_control_ticks"]
            require(case.get("action_label_verified_ticks") == ticks // 5 * 5
                    and case.get("unlabeled_tail_raw_replay_ticks") == ticks % 5
                    and case.get("exact_rgb_frames") == ticks // 5 + 1,
                    "spatial replay omitted action labels, control tail or camera frames")
        if original["success"]:
            require(100 <= case["goal_hold_ticks"] <= case["exact_control_ticks"] <= 1500
                    and not case["collision"] and not case["ever_fallen"]
                    and case["recovery_events"] == 0, "replay disqualifies a claimed teacher success")
    normal = evidence["normalization"]
    require(normal.get("status") == "complete" and normal.get("episodes") == 300
            and normal.get("collection_sha256") == collection_hash
            and normal.get("stats_sha256") == index["stats_sha256"]
            and normal.get("no_clipping") is True and normal.get("normalization_changed") is False
            and all(normal["fields"][k]["clipped_values"] == 0
                    and normal["fields"][k]["roundtrip_max_abs"] < 1e-5 for k in ("state", "action")),
            "complete frozen-normalizer coverage audit is required")
    normal_files = normal.get("files", [])
    require(len(normal_files) == 300 and {(r["seed"], r["task"]) for r in normal_files} == expected
            and all(r["metadata_sha256"] == by_case[r["seed"], r["task"]]["metadata_sha256"]
                    for r in normal_files), "normalizer audit omitted or substituted source episodes")
    base_index = json.loads((base / "index.json").read_text())
    require(not set(seeds).intersection(e["seed"] for e in base_index["episodes"]),
            "additional teacher layouts overlap the immutable base dataset")
    entries = index["episodes"]
    accepted = {key for key, r in by_case.items() if r["success"] is True}
    require(len(entries) == len(accepted) and {(e["seed"], e["task"]) for e in entries} == accepted,
            "release must contain exactly all eligible teacher episodes")
    for entry in entries:
        original = by_case[entry["seed"], entry["task"]]
        require(entry.get("metadata_sha256") == original["metadata_sha256"],
                "released language/state metadata differs from qualified source")
        meta = read_indexed_metadata(root, entry, require_digest=True)
        require(entry["split"] == split_for_seed(entry["seed"])
                and entry["eligible"] is True and entry["search_visibility_gated"] is True
                and meta.get("eligible_training_demonstration") is True
                and meta.get("search_visibility_gated") is True
                and meta["scene"]["name"] == scene
                and meta["student_state"] == "obs61[:48]; command13 excluded"
                and meta["warmup_ticks"] == 50 and meta["control_hz"] == 50 and meta["camera_hz"] == 10,
                "released episode changed its split or public observation contract")
        path = contained(root, entry["path"]) / "navigation.npz"
        require(sha256(path) == entry["sha256"] == meta["files"]["navigation.npz"],
                "qualified teacher tensors changed")
        with np.load(path, allow_pickle=False) as z:
            validate_tensors(z, entry["frames"] - 1, 0, public_only=True)
            require(0 <= original["ticks"] - (entry["frames"] - 1) * 5 < 5,
                    "released teacher sequence does not cover its recorded episode")
    return entries


def export_qualified_teacher_release(source, base_dataset, evidence, output, *, scene=SCENE):
    """Atomically publish a new immutable data artifact after its full audit.

``evidence`` maps each of qualification/protocol/physics_replay/normalization
to an explicit path and expected SHA256. Failed preparation remains in a
separate staging directory; the requested release path is never exposed early.
This function never schedules model training.
"""
    source, base, output = map(lambda p: Path(p).resolve(), (source, base_dataset, output))
    require(scene in RELEASE_SCENES.values(), "unsupported teacher release scene")
    kind = next(k for k, v in RELEASE_SCENES.items() if v == scene)
    require(not output.exists(), "teacher release already exists; never overwrite it")
    require(set(evidence) == {"qualification", "protocol", "physics_replay", "normalization"},
            "all four pinned teacher evidence reports are required")
    blobs, reports = {}, {}
    for name, ref in evidence.items():
        path = Path(ref["path"])
        require(sha256(path) == ref["sha256"], name + " evidence changed before export")
        blobs[name] = path.read_bytes(); reports[name] = json.loads(blobs[name])
    qualification = reports["qualification"]
    require(_qualification_complete(qualification, scene)
            and reports["physics_replay"].get("status") == "complete"
            and reports["normalization"].get("status") == "complete",
            "qualification, replay and normalization must finish before export")
    collection_path = source / "collection.json"
    collection_hash = sha256(collection_path)
    collection = json.loads(collection_path.read_text())
    require(collection.get("status") == "complete" and collection.get("completed_episodes") == 300
            and collection_hash == qualification["cohorts"][scene]["evidence"][0]["sha256"],
            "source collection differs from complete qualification")
    if scene == SPATIAL_SCENE:
        require(collection.get("scene") == scene and collection.get("curriculum_only") is True
                and collection.get("full_obstacle_benchmark_qualified") is False,
                "spatial source must retain its one-obstacle curriculum identity")
    # Inspect every source identity before creating a release directory.
    prepared = []
    for case in qualification["records"]:
        if case["success"] is not True:
            continue
        relative = f"{case['task']}-seed{case['seed']}"
        folder = contained(source, relative); mp = folder / "episode.json"
        require(sha256(mp) == case["metadata_sha256"], "source episode changed before export")
        meta = json.loads(mp.read_text())
        require(meta["seed"] == case["seed"] and meta["task"] == case["task"], "source episode identity changed")
        require(set(meta["files"]).issubset({"navigation.npz", "control_trace.npz"})
                and "navigation.npz" in meta["files"], "unexpected teacher episode payload")
        for name, digest in meta["files"].items():
            require(sha256(contained(folder, name)) == digest, "source trajectory changed before export")
        prepared.append((folder, relative, meta, case["metadata_sha256"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    refs = {}
    for name, blob in blobs.items():
        path = staging / (name + ".json"); path.write_bytes(blob)
        refs[name] = {"path": path.name, "sha256": evidence[name]["sha256"]}
    shutil.copyfile(base / "dataset_stats.json", staging / "dataset_stats.json")
    entries = []
    for folder, relative, meta, digest in prepared:
        dest = staging / relative; dest.mkdir()
        shutil.copyfile(folder / "episode.json", dest / "episode.json")
        for name in meta["files"]:
            shutil.copyfile(folder / name, dest / name)
            require(sha256(dest / name) == meta["files"][name], "copied teacher payload changed")
        entries.append({"path": relative, "seed": meta["seed"], "task": meta["task"],
                        "split": meta["split"], "eligible": meta["eligible_training_demonstration"],
                        "search_visibility_gated": meta["search_visibility_gated"],
                        "frames": meta["decision_count"] + 1, "sha256": meta["files"]["navigation.npz"],
                        "metadata_sha256": digest})
    index = {"kind": kind, "scene": scene, "source_collection_sha256": collection_hash,
             "base_index_sha256": sha256(base / "index.json"),
             "base_qualification_sha256": sha256(base / "qualification_report.json"),
             "stats_sha256": sha256(staging / "dataset_stats.json"), "evidence": refs, "episodes": entries,
             "scope": "Additional complete teacher episodes; not DAgger or full-obstacle model acceptance."}
    (staging / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    index_hash = sha256(staging / "index.json")
    load_qualified_teacher_release(staging, base, index_hash)
    require(sha256(collection_path) == collection_hash, "source collection changed during export")
    receipt = {"status": "complete", "kind": kind, "scene": scene, "episodes": len(entries),
               "index_sha256": index_hash, "stats_sha256": index["stats_sha256"],
               "teacher_qualification_episodes": 300, "data_released": True,
               "normalization_changed": False, "training_started": False,
               "obstacle_avoidance_qualified": False, "learned_model_accepted": False,
               "hardware_validated": False}
    (staging / "release.json").write_text(json.dumps(receipt, indent=2) + "\n")
    require(not output.exists(), "release destination appeared during preparation")
    staging.rename(output)
    return receipt
