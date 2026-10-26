"""Pinned desk/ladder world and explicit climber→get-up execution contract."""

import json
import numpy as np
from .assets import sha256_file
from .community import CommunityPolicy, community_cache_root, load_catalog
from .contracts import JointResidualChunk, PolicyManifest


def load_scene(asset_root):
    path = community_cache_root(asset_root) / "climb-scene/compiled_scene.json"
    expected = load_catalog()["climb-scene"]["generated_files"]["compiled_scene.json"]["sha256"]
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError("Build the verified climb-scene asset with scripts/microduck/build_climb_scene.py")
    return json.loads(path.read_text())


def add_scene(spec, asset_root):
    import mujoco

    scene = load_scene(asset_root)
    for record in scene["records"]:
        name = "lx_climb_" + record["name"]
        kind = getattr(mujoco.mjtGeom, "mjGEOM_" + record["type"].upper())
        kwargs = {"name": name, "type": kind, "pos": record["pos"]}
        if record.get("quat"):
            kwargs["quat"] = record["quat"]
        if record["type"] == "mesh":
            mesh = spec.add_mesh(name=name + "_mesh")
            mesh.uservert = np.asarray(record["vertices"]).ravel().tolist()
            kwargs["meshname"] = mesh.name
        else:
            kwargs["size"] = record["size"] + [0.0] * (3 - len(record["size"]))
        geom = spec.worldbody.add_geom(**kwargs)
        geom.friction = [1.0, 0.005, 0.0001]
        geom.solref = [0.01, 1.0]
        geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
        geom.priority = 1
        geom.condim = 3
        geom.rgba = [0.78, 0.68, 0.55, 1.0] if record["name"] == "desktop" else [0.17, 0.30, 0.38, 1.0]
    return scene


def desk_state(env):
    physics = env.physics
    model, data, mj = physics.model, physics.data, physics.mj
    desk = load_scene(env.asset_root)["desktop"] if not hasattr(env, "_climb_desk") else env._climb_desk
    root = data.xpos[physics.trunk]
    foot_contact = [False, False]
    for a, b in physics.last_contacts:
        for ground, robot in ((a, b), (b, a)):
            if mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, ground) != "lx_climb_desktop":
                continue
            body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot])) or ""
            foot_contact[0] |= body.endswith("ankle_left")
            foot_contact[1] |= body.endswith("ankle_right")
    lo = np.asarray(desk["pos"]) - desk["size"]
    hi = np.asarray(desk["pos"]) + desk["size"]
    inside = lo[0] + 0.04 < root[0] < hi[0] - 0.05 and lo[1] + 0.05 < root[1] < hi[1] - 0.05 and root[2] > 0.66
    _, gravity, _, _ = physics.sensors()
    safe = (
        all(foot_contact)
        and inside
        and gravity[2] < -0.85
        and np.linalg.norm(data.qvel[physics.trunk_qvel : physics.trunk_qvel + 3]) < 0.25
        and np.linalg.norm(data.qvel[physics.trunk_qvel + 3 : physics.trunk_qvel + 6]) < 2.5
    )
    return {"eligible": bool(any(foot_contact) and inside), "standing_safe": bool(safe), "foot_desktop_contacts": foot_contact, "root_inside_margin": bool(inside), "uses_privileged_state": True}


class ClimbPolicy:
    def __init__(self, env):
        self.env = env
        self.climber = CommunityPolicy("climb-climber", env.asset_root)
        self.getup = CommunityPolicy("climb-getup", env.asset_root)
        self.manifest = PolicyManifest(
            "climb_then_getup",
            "joint_residual",
            "official_61d",
            env.profile,
            uses_privileged_state=True,
            artifacts={"climber_sha256": self.climber.manifest.artifacts["checkpoint_sha256"], "getup_sha256": self.getup.manifest.artifacts["checkpoint_sha256"]},
            metadata={
                "required_scene": "climb_desk",
                "simulation_profile": "official_bam",
                "supervisor": "physical_foot_desktop_contact_and_4cm_root_margin",
                "required_sensors": ["proprioception", "simulator_truth"],
                "gain_ratio_after_switch": 0.8,
                "filter_head_legs": [0.5, 0.7],
                "delay": "fixed_one_control_tick_20ms; author randomized 3..6 physics substeps",
                "hardware_validated": False,
            },
        )
        self.reset("")

    def reset(self, episode_id):
        self.recovering = False
        self.climber.reset(episode_id)
        self.getup.reset(episode_id)

    def predict(self, observation):
        status = desk_state(self.env)
        self.recovering |= status["eligible"]
        state = observation.state.copy()
        state[48:] = 0.0
        raw = (self.getup if self.recovering else self.climber).infer(state)
        return JointResidualChunk(
            raw[None],
            observation.episode_id,
            observation.tick,
            observation.sim_time_s,
            action_profile="climb_getup_raw" if self.recovering else "climb_raw",
            diagnostics={**status, "phase": "getup" if self.recovering else "climb", "filter_in_executor": True},
        )
