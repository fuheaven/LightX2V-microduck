"""Observer-only compiled MuJoCo geometry. Never added to policy observations."""

import gzip
import time

from .schema import digest, write_json


def manifest(env, *, robot_only=False):
    m, mj = env.physics.model, env.physics.mj
    geoms, meshes = [], {}
    root = m.body_rootid[env.physics.trunk]
    for i in range(m.ngeom):
        if robot_only and m.body_rootid[m.geom_bodyid[i]] != root:
            continue
        mat = int(m.geom_matid[i])
        rgba = (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[i]).tolist()
        if rgba[3] == 0 or m.geom_group[i] >= 3:
            continue
        mesh = int(m.geom_dataid[i]) if int(m.geom_type[i]) == int(mj.mjtGeom.mjGEOM_MESH) else None
        if mesh is not None and str(mesh) not in meshes:
            va, vn = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            fa, fn = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            meshes[str(mesh)] = {"vertices": m.mesh_vert[va : va + vn].reshape(-1).tolist(), "faces": m.mesh_face[fa : fa + fn].reshape(-1).tolist()}
        geoms.append(
            {
                "id": i,
                "name": mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}",
                "body": mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])),
                "type": int(m.geom_type[i]),
                "size": m.geom_size[i].tolist(),
                "rgba": rgba,
                "mesh": mesh,
                "collision": bool(m.geom_contype[i] or m.geom_conaffinity[i]),
            }
        )
    value = {
        "schema": "studio_visual_v1",
        "units": "m",
        "up_axis": "z",
        "rotation": "row_major_3x3",
        "robot_signature": env._robot_signature,
        "geoms": geoms,
        "meshes": meshes,
        "scope": "observer_only",
        "physics_profile": env.profile,
    }
    return {**value, "sha256": digest(value)}


def snapshot(env, visual, control_source="initializing"):
    d, m = env.physics.data, env.physics.model
    cam = env.physics.camera
    observer = env.physics.observer_camera
    return {
        "schema": "studio_visual_state_v1",
        "episode_id": env.episode_id,
        "tick": env.tick,
        "sim_time_s": env.sim_time_s,
        "captured_at": time.time(),
        "manifest_sha256": visual["sha256"],
        "control_source": control_source,
        "scope": "observer_only",
        "geoms": [{"id": g["id"], "position": d.geom_xpos[g["id"]].tolist(), "rotation": d.geom_xmat[g["id"]].tolist()} for g in visual["geoms"]],
        "camera": {"position": d.cam_xpos[cam].tolist(), "rotation": d.cam_xmat[cam].tolist(), "fov_y": float(m.cam_fovy[cam]), "metadata": env.physics.camera_metadata()},
        "observer_camera": None
        if observer is None
        else {
            "lookat": observer.lookat.tolist(),
            "distance": float(observer.distance),
            "azimuth_deg": float(observer.azimuth),
            "elevation_deg": float(observer.elevation),
            "fov_y": float(m.vis.global_.fovy),
            "scope": "observer_only",
        },
    }


class VisualRecorder:
    def __init__(self, env, out):
        self.visual = manifest(env)
        self.out = out
        write_json(out / "visual.json", self.visual)
        self.log = gzip.open(out / "poses.jsonl.gz", "wt", encoding="utf-8")

    def capture(self, env, source="initializing"):
        from .schema import canonical

        value = snapshot(env, self.visual, source)
        write_json(self.out / "state.json", value)
        self.log.write(canonical(value) + "\n")
        return value

    def close(self):
        self.log.close()


def prepare_robot(root, source_sha):
    """CPU worker prepares a static official model for the editor, without GL."""
    import json

    from .scene import make_env
    from .schema import SceneSpec, TaskConfig

    target = root / "assets/robot.json"
    if target.exists() and json.loads(target.read_text()).get("source_sha256") == source_sha:
        return
    env = make_env(SceneSpec().model_dump(), TaskConfig().model_dump(), render=False)
    try:
        env.reset(seed=0, task=env.studio_task)
        v = manifest(env, robot_only=True)
        write_json(target, {"source_sha256": source_sha, "visual": v, "state": snapshot(env, v)})
    finally:
        env.close()
