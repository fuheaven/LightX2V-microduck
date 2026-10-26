"""Qualified new-instruction demonstrations, kept distinct from DAgger labels.

The original physical/controller state is retained, but each color instruction
starts a new task clock. Identity includes the source task and original tick;
the zero tick in exported tensors must never collapse different source states.
Only RGB, proprio48, action5 and ticks enter model batches.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .navigation import NAVIGATION_CONTRACT
from .navigation_augmentation import PUBLIC_KEYS, contained, qualified_base, sha256, validate_tensors
from .navigation_retarget_qualification import assess


KIND = "retargeted_instruction_demonstrations"
TASKS = {f"find_{c}_ball" for c in ("red", "blue", "yellow")}


def require_teacher_qualification(value):
    if value.get("status") != "complete" or value.get("teacher_qualified") is not True:
        raise ValueError("the complete retarget teacher has not qualified")
    counts = value["per_target"]
    sources = value["source_states"]
    if (set(counts) != TASKS or isinstance(sources, bool)
            or not isinstance(sources, int) or sources < 150):
        raise ValueError("retarget teacher qualification lacks the full cohort")
    for count in counts.values():
        n, passed, changed, changed_passed = (count[k] for k in
            ("total", "successes", "changed_total", "changed_successes"))
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in (n, passed, changed, changed_passed))
                or n != value["source_states"] or not 0 <= changed_passed <= changed <= n
                or not changed_passed <= passed <= n or changed < 100
                or passed - changed_passed > n - changed
                or passed / n < .95 or changed_passed / changed < .95):
            raise ValueError("retarget teacher fails the unchanged per-target 95 percent gate")


def identity(entry):
    return entry["original_task"], entry["seed"], entry["original_source_tick"], entry["task"]


def _base_training(root, evaluation_protocol):
    from .wam_data import split_for_seed

    index, qualification = qualified_base(root, evaluation_protocol)
    reserved = set(evaluation_protocol["development_seeds"]) | set(evaluation_protocol["independent_final_seeds"])
    training = {(e["task"], e["seed"]): e for e in index["episodes"]
                if e["eligible"] and e["split"] == "train" and e.get("search_visibility_gated")}
    if any(seed in reserved or 10000 <= seed < 10200 or split_for_seed(seed) != "train"
           for _, seed in training):
        raise ValueError("retarget base contains held-out layouts")
    return training, qualification


def export_retargeted(source, base_dataset, teacher_protocol, expected_teacher_protocol_sha256,
                      launch, source_manifest, normalization_audit, evaluation_protocol,
                      expected_evaluation_protocol_sha256, output):
    from .wam_data import write_json

    source, base_dataset, output = map(lambda p: Path(p).resolve(), (source, base_dataset, output))
    if output.exists():
        raise FileExistsError("retarget releases are immutable")
    if sha256(evaluation_protocol) != expected_evaluation_protocol_sha256:
        raise ValueError("reserved evaluation protocol changed")
    evaluation = json.loads(Path(evaluation_protocol).read_text())
    training, base_qualification = _base_training(base_dataset, evaluation)
    qualification = assess(source, Path(teacher_protocol), expected_teacher_protocol_sha256,
                           Path(launch), Path(source_manifest))
    require_teacher_qualification(qualification)
    source_report = json.loads((source / "report.json").read_text())
    norm = json.loads(Path(normalization_audit).read_text())
    manifest = json.loads(Path(source_manifest).read_text())
    if (norm["status"] != "complete" or norm["source_report_sha256"] != sha256(source / "report.json")
            or norm["script_sha256"] != manifest["files"]["scripts/microduck/fastwam_retarget_normalization_audit.py"]
            or norm["stats_sha256"] != base_qualification["stats_sha256"]
            or norm["no_clipping"] is not True or norm["normalization_changed"] is not False
            or set(norm["per_field"]) != {"action", "state"}
            or any(v["clipped_values"] != 0 or not np.isfinite(v["roundtrip_max_abs"])
                   or v["roundtrip_max_abs"] > 1e-5 for v in norm["per_field"].values())):
        raise ValueError("matching complete reference-normalizer audit with no clipping is required")
    normalizer_files = {e["path"]: e["sha256"] for e in norm["files"]}
    if len(normalizer_files) != len(norm["files"]):
        raise ValueError("duplicate normalization audit entries")
    accepted, rejected, seen, files, total_rows = [], [], set(), {}, 0
    for case in source_report["cases"]:
        source_key = case["source_task"], case["seed"]
        if source_key not in training:
            raise ValueError("source anchor is outside qualified training layouts")
        parent = contained(base_dataset, training[source_key]["path"])
        if sha256(parent / "episode.json") != case["source_episode_sha256"]:
            raise ValueError("source anchor episode changed")
        for row in case["branches"]:
            branch_dir = contained(source, row["path"])
            branch = json.loads((branch_dir / "report.json").read_text())
            key = row["task"], case["seed"]
            if key not in training:
                raise ValueError("retarget instruction is outside qualified training layouts")
            target_parent = contained(base_dataset, training[key]["path"])
            if branch["instruction"] != json.loads((target_parent / "episode.json").read_text())["instruction"]:
                raise ValueError("retarget language disagrees with its target task")
            item = {"path": row["path"], "task": row["task"], "seed": case["seed"],
                    "original_task": case["source_task"], "original_source_tick": case["source_tick"],
                    "source_tick": 0, "label_decisions": branch["label_decisions"],
                    "source_navigation_sha256": branch["navigation_sha256"],
                    "source_report_sha256": sha256(branch_dir / "report.json"),
                    "first_rgb_sha256": branch["first_rgb_sha256"],
                    "first_proprio_sha256": branch["first_proprio_sha256"],
                    "instruction": branch["instruction"]}
            if identity(item) in seen:
                raise ValueError("duplicate retarget source/task identity")
            seen.add(identity(item))
            with np.load(branch_dir / "navigation.npz", allow_pickle=False) as data:
                validate_tensors(data, item["label_decisions"], 0)
                if not np.array_equal(data["proprio"][0], data["continuation_state61"][0, :48]):
                    raise ValueError("retarget proprio differs from the actual first control state")
            files[row["path"]] = branch["navigation_sha256"]
            total_rows += item["label_decisions"]
            (accepted if branch["teacher_continuation_success"] else rejected).append(item)
    if files != normalizer_files or total_rows != norm["rows"]:
        raise ValueError("normalization audit omitted or substituted retarget branches")
    if len(accepted) != sum(v["successes"] for v in qualification["per_target"].values()):
        raise ValueError("release selection differs from qualified teacher outcomes")
    output.mkdir(parents=True)
    entries = []
    for item in accepted:
        original, target = source / item["path"], output / item["path"]
        target.mkdir(parents=True)
        with np.load(original / "navigation.npz", allow_pickle=False) as data:
            np.savez_compressed(target / "navigation.npz", **{k: data[k] for k in sorted(PUBLIC_KEYS)})
        meta = {**item, "schema": NAVIGATION_CONTRACT, "kind": KIND, "split": "train",
                "eligible_training_supervision": True, "task_clock_restarted": True,
                "physical_state_reset": False, "teacher_history_reset": True,
                "sampling": "one new-instruction clip at a retained physical state; not a DAgger continuation"}
        write_json(target / "episode.json", meta)
        entries.append({**item, "split": "train", "eligible": True, "search_visibility_gated": True,
                        "frames": item["label_decisions"] + 1, "sample_starts": [0],
                        "sha256": sha256(target / "navigation.npz"),
                        "metadata_sha256": sha256(target / "episode.json")})
    audit = {"status": "complete", "accepted": accepted, "rejected": rejected,
             "source_report_sha256": sha256(source / "report.json"),
             "public_tensors_only": True, "kind": KIND, "weights_updated": False}
    write_json(output / "source_audit.json", audit)
    write_json(output / "teacher_qualification.json", qualification)
    for original, name in ((normalization_audit, "normalization_audit.json"),
                           (evaluation_protocol, "evaluation_protocol.json"),
                           (teacher_protocol, "teacher_protocol.json"),
                           (source_manifest, "source_manifest.json"),
                           (launch, "source_launch.json"),
                           (base_dataset / "dataset_stats.json", "dataset_stats.json")):
        shutil.copyfile(original, output / name)
    proofs = {p.name: sha256(p) for p in sorted(output.glob("*.json"))}
    index = {"schema": NAVIGATION_CONTRACT, "kind": KIND, "episodes": entries, "proof_sha256": proofs,
             "base_index_sha256": sha256(base_dataset / "index.json"),
             "base_qualification_sha256": sha256(base_dataset / "qualification_report.json"),
             "stats_sha256": sha256(base_dataset / "dataset_stats.json"),
             "source_report_sha256": audit["source_report_sha256"],
             "not_dagger": True, "scope": "qualified training-only instruction retargeting; statistics unchanged"}
    write_json(output / "index.json", index)
    receipt = {"status": "complete", "index_sha256": sha256(output / "index.json"),
               "retarget_clips": len(entries), "excluded_failed_branches": len(rejected),
               "label_decisions": sum(e["label_decisions"] for e in entries),
               "public_tensors_only": True, "weights_updated": False, "capability_accepted": False}
    write_json(output / "release.json", receipt)
    load_retargeted(output, base_dataset)
    return receipt


def load_retargeted(root, base_dataset):
    root, base_dataset = Path(root), Path(base_dataset)
    index = json.loads((root / "index.json").read_text())
    receipt = json.loads((root / "release.json").read_text())
    if (receipt["status"] != "complete" or receipt["index_sha256"] != sha256(root / "index.json")
            or index["schema"] != NAVIGATION_CONTRACT or index["kind"] != KIND or index["not_dagger"] is not True
            or index["base_index_sha256"] != sha256(base_dataset / "index.json")
            or index["base_qualification_sha256"] != sha256(base_dataset / "qualification_report.json")
            or index["stats_sha256"] != sha256(base_dataset / "dataset_stats.json")
            or index["stats_sha256"] != sha256(root / "dataset_stats.json")):
        raise ValueError("retarget release or frozen base artifact changed")
    required = {"source_audit.json", "teacher_qualification.json", "normalization_audit.json",
                "evaluation_protocol.json", "teacher_protocol.json", "source_manifest.json",
                "source_launch.json", "dataset_stats.json"}
    if set(index["proof_sha256"]) != required:
        raise ValueError("retarget release lacks its complete qualification provenance")
    for name, digest in index["proof_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError("retarget qualification proof changed")
    qualification = json.loads((root / "teacher_qualification.json").read_text())
    require_teacher_qualification(qualification)
    norm = json.loads((root / "normalization_audit.json").read_text())
    manifest = json.loads((root / "source_manifest.json").read_text())
    launch = json.loads((root / "source_launch.json").read_text())
    protocol = json.loads((root / "teacher_protocol.json").read_text())
    if (qualification["protocol_sha256"] != sha256(root / "teacher_protocol.json")
            or launch["protocol_sha256"] != qualification["protocol_sha256"]
            or qualification["source_archive_sha256"] != manifest["sha256"]
            or launch["source_archive_sha256"] != manifest["sha256"]
            or protocol["source_index_sha256"] != index["base_index_sha256"]
            or norm["source_report_sha256"] != index["source_report_sha256"]
            or norm["stats_sha256"] != index["stats_sha256"]
            or norm["script_sha256"] != manifest["files"]["scripts/microduck/fastwam_retarget_normalization_audit.py"]
            or norm["no_clipping"] is not True or norm["normalization_changed"] is not False):
        raise ValueError("retarget qualification and normalization provenance disagree")
    evaluation = json.loads((root / "evaluation_protocol.json").read_text())
    training, _ = _base_training(base_dataset, evaluation)
    audit = json.loads((root / "source_audit.json").read_text())
    approved = {identity(e): e for e in audit["accepted"]}
    if (len(approved) != len(audit["accepted"]) or len(index["episodes"]) != len(approved)
            or len(approved) != sum(v["successes"] for v in qualification["per_target"].values())
            or audit["source_report_sha256"] != qualification["collection_report_sha256"]
            or index["source_report_sha256"] != audit["source_report_sha256"]):
        raise ValueError("retarget release selection changed")
    identities = set()
    for entry in index["episodes"]:
        key = identity(entry)
        if (key in identities or key not in approved or entry["source_tick"] != 0
                or (entry["task"], entry["seed"]) not in training
                or (entry["original_task"], entry["seed"]) not in training
                or entry["split"] != "train" or not entry["eligible"] or not entry["search_visibility_gated"]
                or entry["sample_starts"] != [0]):
            raise ValueError("duplicate, held-out or semantically invalid retarget clip")
        identities.add(key)
        path = contained(root, entry["path"])
        if sha256(path / "navigation.npz") != entry["sha256"] or sha256(path / "episode.json") != entry["metadata_sha256"]:
            raise ValueError("retarget episode changed after export")
        meta = json.loads((path / "episode.json").read_text())
        if (any(meta.get(k) != v or entry.get(k) != v for k, v in approved[key].items())
                or meta["kind"] != KIND or not meta["task_clock_restarted"]
                or meta["physical_state_reset"] or not meta["teacher_history_reset"]):
            raise ValueError("retarget instruction, source identity or clock changed")
        parent = contained(base_dataset, training[(entry["task"], entry["seed"])]["path"])
        if meta["instruction"] != json.loads((parent / "episode.json").read_text())["instruction"]:
            raise ValueError("retarget instruction differs from the target task")
        with np.load(path / "navigation.npz", allow_pickle=False) as data:
            validate_tensors(data, entry["frames"] - 1, 0, public_only=True)
    return index["episodes"]
