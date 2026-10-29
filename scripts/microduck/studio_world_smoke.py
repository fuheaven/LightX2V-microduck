"""Real MuJoCo geometry, observer noninterference and dataset-loader check.

Three near-goal episodes check hold/recording only, not navigation generalization.
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Small real-physics/data plumbing check, not skill acceptance")
    parser.add_argument("--out", type=Path, default=Path("artifacts/studio-world-smoke"))
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    os.environ["MICRODUCK_ASSET_ROOT"] = str(args.assets.resolve())
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path[:0] = [str(root / "embodied/microduck/src"), str(root / "third_party/bam")]
    from lightx2v_microduck.navigation import intent_from_action
    from lightx2v_microduck.studio.assets import parts
    from lightx2v_microduck.studio.dataset import SkillDataset, audit_dataset
    from lightx2v_microduck.studio.scene import layout_identity, make_env, sample_scene
    from lightx2v_microduck.studio.schema import ACTION, OBSERVATION, SkillDraft, file_hash, write_json
    from lightx2v_microduck.studio.simulation import run_episode
    from lightx2v_microduck.studio.visual_state import manifest, snapshot

    s = SkillDraft().model_dump()
    s["scene"]["version"] = 2
    s["scene"]["size"] = 4
    s["scene"]["objects"][0].update(x=0.36, y=0, jitter=0.005)
    s["scene"]["spawn"].update(jitter=0.005, yaw_jitter=1)
    s["scene"]["objects"].append(
        dict(id="desk", kind="obstacle", color="gray", x=-1, y=0.5, yaw=30, width=0.9, depth=0.55, height=0.65, jitter=0, asset_id="table", asset_version="1.0.0", material="wood")
    )
    s["task"].update(timeout_s=8, categories=["ball"], colors=["red"], groups=["clear"])
    physics_scene = copy.deepcopy(s["scene"])
    physics_scene["objects"][0].update(x=0.8, y=0.3)
    a, b = [make_env(physics_scene, s["task"], render=False) for _ in range(2)]
    try:
        for env in (a, b):
            env.reset(seed=3, task=env.studio_task)
        v = manifest(a)
        observed = snapshot(a, v)
        # Match mesh transforms directly to authoritative MuJoCo arrays.
        pose_error = max(np.max(np.abs(np.array(g["position"]) - a.physics.data.geom_xpos[g["id"]])) for g in observed["geoms"])
        max_q = 0.0
        for tick in range(120):
            values = [0.03 if tick > 30 else 0, 0, 0.1 if tick > 60 else 0, 0, 0]
            for env in (a, b):
                env.step(intent_from_action(values, episode_id=env.episode_id, tick=env.tick, duration_ticks=1))
            snapshot(a, v)  # Observer export may not change physics/observation.
            max_q = max(max_q, float(np.max(np.abs(a.physics.data.qpos - b.physics.data.qpos))))
        obj = s["scene"]["objects"][-1]
        rot = np.array([[math.cos(math.pi / 6), -math.sin(math.pi / 6)], [math.sin(math.pi / 6), math.cos(math.pi / 6)]])
        shape_errors = []
        for i, p in enumerate(parts(obj)):
            gid = a.physics.mj.mj_name2id(a.physics.model, a.physics.mj.mjtObj.mjOBJ_GEOM, f"studio_geom_desk_{i}")
            expected = np.r_[rot @ p["pos"][:2] + [obj["x"], obj["y"]], p["pos"][2]]
            shape_errors.append(float(np.max(np.abs(a.physics.data.geom_xpos[gid] - expected))))
    finally:
        a.close()
        b.close()
    entries = []
    for i, split in enumerate(["train", "validation", "test"]):
        scene = sample_scene(s["scene"], 1000 + i, group="clear", split=split)
        case = dict(
            seed=1000 + i, split=split, group="clear", category="ball", color="red", scene=scene, layout_id=layout_identity(scene), instance_id=split + "-ball", instruction_family=split + ":0"
        )
        folder = out / "data" / split
        report = run_episode(s, case, folder, policy_kind="teacher")
        with np.load(folder / "sensors.npz") as d:
            n = len(d["tick"])
        entries.append(
            dict(
                id=split,
                path=split + "/sensors.npz",
                sha256=file_hash(folder / "sensors.npz"),
                **{k: case[k] for k in ["category", "color", "group", "split", "layout_id", "instance_id", "instruction_family"]},
                instruction=report["instruction"],
                eligible=report["success"] and not report["interventions"],
                frames=n,
            )
        )
        print(split, report["success"], report["reason"], n, flush=True)
    write_json(out / "data/index.json", dict(schema="studio_dataset_v1", observation=OBSERVATION, action=ACTION, episodes=entries, smoke_only=True))
    try:
        audit = audit_dataset(out / "data")
        ds = SkillDataset(out / "data")
        shape = {k: list(v.shape) for k, v in ds[0].items() if hasattr(v, "shape")}
    except Exception as e:
        audit = {"passed": False, "reason": str(e)}
        shape = {}
    result = dict(
        pose_error_m=pose_error,
        observer_physics_qpos_max_error=max_q,
        furniture_position_max_error_m=max(shape_errors),
        official_robot=a._robot_signature,
        recorded_episodes=entries,
        dataset_audit=audit,
        training_loader_shapes=shape,
        formal_skill_acceptance=False,
    )
    write_json(out / "report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "recorded_episodes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
