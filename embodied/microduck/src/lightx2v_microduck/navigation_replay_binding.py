"""Bind retrospective labels to a qualified scene and its actual teacher.

Reset qualification is not qualification at arbitrary student states. This
module authorizes a diagnostic replay, never publication of its labels.
"""
import json
from pathlib import Path

from .navigation_augmentation import contained, sha256
from .navigation_teacher_data import load_qualified_teacher_release, SCENE
from .wam_data import split_for_seed


TEACHER = "visual_search_pose_diverse_persistent"
RUNTIME_MODULES = (
    "navigation.py", "navigation_search.py", "navigation_teacher_experiments.py",
    "navigation_motion.py", "navigation_active_search.py", "navigation_spawn_support.py",
    "navigation_view_route.py", "navigation_long_search.py", "navigation_persistent_search.py",
    "navigation_pose_checked_search.py", "teachers.py", "env.py", "scenes.py", "physics.py", "tasks.py",
)


def load_replay_binding(protocol, *, verify_runtime=True):
    """Check the complete qualified release and every selected training source."""
    ref = protocol["qualified_teacher_release"]
    if not isinstance(ref, dict) or set(ref) - {"root", "index_sha256", "base_dataset", "calibration"}:
        raise ValueError("unknown qualified replay binding fields")
    root, base = Path(ref["root"]), Path(ref["base_dataset"])
    entries = load_qualified_teacher_release(root, base, ref["index_sha256"])
    index = json.loads((root / "index.json").read_text())
    qualification_path = contained(root, index["evidence"]["qualification"]["path"])
    qualification = json.loads(qualification_path.read_text())
    teacher_protocol = json.loads(contained(root, index["evidence"]["protocol"]["path"]).read_text())
    if teacher_protocol.get("teacher") != TEACHER:
        raise ValueError("qualified replay requires the explicitly supported teacher kind")
    evidence = qualification["cohorts"][SCENE]["evidence"]
    if len(evidence) != 1:
        raise ValueError("qualified replay needs one unambiguous teacher source")
    sources = evidence[0].get("source_sha256", {})
    if not set(RUNTIME_MODULES).issubset(sources):
        raise ValueError("teacher qualification omits runtime source identities")
    runtime = {name: sources[name] for name in RUNTIME_MODULES}
    if verify_runtime:
        for name, digest in runtime.items():
            if sha256(Path(__file__).parent / name) != digest:
                raise ValueError("live teacher/physics differs from qualified source: " + name)
    evaluation_path = Path(protocol["evaluation_protocol_path"])
    if sha256(evaluation_path) != protocol["evaluation_protocol_sha256"]:
        raise ValueError("held-out evaluation protocol changed")
    evaluation = json.loads(evaluation_path.read_text())
    forbidden = set(evaluation["development_seeds"]) | set(evaluation["independent_final_seeds"])
    forbidden.update(range(10000, 10200))
    eligible = {(e["task"], e["seed"]): e for e in entries if e["eligible"] and e["split"] == "train"}
    seen, options = set(), None
    for source in protocol["sources"]:
        report_path = Path(source["path"]) / "report.json"
        if sha256(report_path) != source["files"]["report.json"]:
            raise ValueError("student source metadata changed")
        meta = json.loads(report_path.read_text())
        key = meta["task"], meta["seed"]
        if (key in seen or key not in eligible or key[1] in forbidden
                or split_for_seed(key[1]) != "train" or meta["split"] != "train"
                or meta.get("evaluator", {}).get("scene", {}).get("name") != SCENE):
            raise ValueError("replay requires distinct qualified training sources in the exact scene")
        seen.add(key)
        original = json.loads((contained(root, eligible[key]["path"]) / "episode.json").read_text())
        current = original.get("teacher_options", {})
        if (original.get("teacher") != TEACHER + "_astar_with_official_onnx"
                or set(current) != {"calibration", "expected_calibration_sha256"}
                or (options is not None and current != options)):
            raise ValueError("qualified teacher identity or calibration is inconsistent")
        options = dict(current)
    if not seen:
        raise ValueError("empty qualified replay cohort")
    # Relocation is allowed only with the original calibration content hash.
    options["calibration"] = str(ref.get("calibration", options["calibration"]))
    calibration = Path(options["calibration"])
    if sha256(calibration / "report.json") != options["expected_calibration_sha256"]:
        raise ValueError("qualified motion calibration report changed")
    motion = json.loads((calibration / "report.json").read_text())
    if sha256(calibration / "motion.npz") != motion["motion_sha256"]:
        raise ValueError("qualified motion calibration tensors changed")
    return {
        "scene": SCENE, "qualified_teacher_index_sha256": ref["index_sha256"],
        "qualification_path": str(qualification_path), "qualification_sha256": sha256(qualification_path),
        "teacher_kinds": {"original": "visual_search", "qualified": TEACHER},
        "teacher_options": options, "runtime_source_sha256": runtime,
        "scope": "Original teacher is a comparison only; qualified reset teacher still needs student-state assessment.",
        "general_student_state_teacher_qualified": False, "training_release_allowed": False,
    }


def make_replay_teachers(env, binding):
    from .navigation import VisualSearchTeacher
    from .navigation_teacher_experiments import PoseDiversePersistentVisualTeacher

    if env.scene != binding["scene"] or env.profile != "official_bam":
        raise ValueError("replay environment differs from qualified scene/profile")
    if binding["teacher_kinds"] != {"original": "visual_search", "qualified": TEACHER}:
        raise ValueError("unsupported replay teacher mapping")
    return {"original": VisualSearchTeacher(env),
            "qualified": PoseDiversePersistentVisualTeacher(env, **binding["teacher_options"])}


def verify_replayed_evaluator(actual, expected, model_check, binding):
    """Compare all final values across a pinned source-version boundary.

Model equality must first be checked numerically by verify_model. Only UUID,
verified asset-path spelling and the explicitly qualified source hash may
differ. No outcome, timing, contact or sensor field is removed.
"""
    actual, expected = json.loads(json.dumps(actual)), json.loads(json.dumps(expected))
    if (model_check.get("numeric_model_signature_matches") is not True
            or model_check["source_path_substituted_signature"] != expected["model_signature"]
            or model_check["local_signature"] != actual["model_signature"]
            or actual["evaluator_source_sha256"] != binding["runtime_source_sha256"]["env.py"]):
        raise ValueError("qualified evaluator or numerical model identity differs")
    for value in (actual, expected):
        value.pop("episode_id")
    actual["model_signature"] = expected["model_signature"]
    for key in ("source_xml", "robot_source_xml", "original_demo_root"):
        actual["scene"][key] = expected["scene"][key]
    source_hash, replay_hash = expected["evaluator_source_sha256"], actual["evaluator_source_sha256"]
    actual["evaluator_source_sha256"] = source_hash
    if actual != expected:
        raise ValueError("qualified replay changed the original complete evaluator state")
    return {"complete_final_evaluator_equal": True, "source_evaluator_sha256": source_hash,
            "replay_evaluator_sha256": replay_hash, "asset_paths_numerically_verified": True}
