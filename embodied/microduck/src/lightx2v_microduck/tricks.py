"""Community tricks under our pinned BAM physics, with explicit task scoring.

Protocol reference: easyrider11/microduck_rl@146777b989f47160d26590e29a9d13957b2c6a3c
scripts/eval_roulade_entry.py and eval_roller_slalom.py (Apache-2.0).
Their reference harness uses position PD; these are BAM transfer experiments.
"""

import math
import numpy as np

from .community import CommunityPolicy
from .contracts import JointResidualChunk, PolicyManifest

SPACING, AMPLITUDE, CONES = 0.7, 0.18, 8
SOURCE_REVISION = "146777b989f47160d26590e29a9d13957b2c6a3c"


def roll_trigger_tick(seed):
    return 50 + int((2.0 + np.random.default_rng(seed).uniform(0.0, 0.6)) * 50)


class TrickPolicy:
    def __init__(self, env, *, baseline=False):
        self.env, self.baseline = env, baseline
        self.roll = env.task.task_id == "run_roll_recover"
        required = "flat" if self.roll else "slalom_cones"
        if env.scene != required:
            raise ValueError(f"trick requires {required}")
        identifier = "roulade-elan-v2" if self.roll else "roller-slalom-v2"
        self.actor = None if baseline else CommunityPolicy(identifier, env.asset_root)
        self.manifest = PolicyManifest(
            f"tricks_{'official' if baseline else identifier}",
            "joint_residual",
            "official_61d",
            env.profile,
            embodiment=env.task.embodiment,
            uses_privileged_state=not self.roll,
            artifacts={} if baseline else self.actor.manifest.artifacts,
            metadata={
                "source_revision": SOURCE_REVISION,
                "required_scene": required,
                "simulation_profile": "official_bam",
                "required_sensors": ["proprioception"],
                "protocol": "walk_handoff" if self.roll else "truth_pure_pursuit_physical_cones",
                "author_physics": "position_pd_1.75A",
                "evidence_scope": "BAM_transfer",
            },
        )

    def reset(self, episode_id):
        if self.actor:
            self.actor.reset(episode_id)

    def predict(self, observation):
        command = np.zeros(13, np.float32)
        state = observation.state.copy()
        if self.roll:
            trigger = roll_trigger_tick(self.env.seed)
            if observation.tick < 50:
                role = "stand"
            elif observation.tick < trigger:
                role, command[0] = "walk", 0.4
            elif observation.tick < trigger + 100:
                role = "roulade"
            else:
                role = "stand"
        else:
            role = "roller"
            if observation.tick >= 50:
                world = self.env.evaluator_state()
                x, y = world["robot_position"][:2]
                desired = math.atan2(AMPLITUDE * math.sin(math.pi * (x + 0.35) / SPACING) - y, 0.35)
                error = math.atan2(math.sin(desired - world["robot_yaw"]), math.cos(desired - world["robot_yaw"]))
                command[0], command[2] = 0.5, np.clip(error, -0.25, 0.25)
        state[48:] = command
        use_candidate = not self.baseline and (role == "roulade" if self.roll else True)
        raw = self.actor.infer(state) if use_candidate else self.env.bank.infer(role, state)
        return JointResidualChunk(
            raw[None],
            observation.episode_id,
            observation.tick,
            observation.sim_time_s,
            diagnostics={"phase": role, "command13": command.tolist(), "candidate_active": use_candidate, "uses_privileged_state": not self.roll},
        )


def score_trick(task_id, trace, *, seed, collision):
    v = np.asarray(trace)
    up, height = -float(v[-1, 4]), float(v[-1, 3])
    if task_id == "run_roll_recover":
        trigger = roll_trigger_tick(seed) * 0.02
        active = v[v[:, 0] >= trigger]
        angle = float(np.max(np.cumsum(active[:, 6]) * 0.02)) if len(active) else 0.0
        rolled, recovered = angle >= math.radians(260), up >= 0.85 and height >= 0.09
        window = v[(v[:, 0] >= trigger - 0.12) & (v[:, 0] <= trigger)]
        entry_speed = float(np.mean(window[:, 9])) if len(window) else 0.0
        return {
            "success": bool(rolled and recovered and entry_speed >= 0.10 and not collision),
            "rolled": rolled,
            "recovered": recovered,
            "entry_speed_m_s": entry_speed,
            "max_forward_pitch_deg": math.degrees(angle),
            "collision": collision,
            "author_terminal_metric": bool(rolled and recovered),
            "protocol": "run_roll_recover_bam_v1",
        }
    x, y = v[:, 1], v[:, 2]
    gates, hits = [], []
    for k in range(CONES):
        cx = (k + 1.5) * SPACING
        crossing = np.flatnonzero((x[:-1] < cx) & (x[1:] >= cx)) + 1
        if any(abs(y[i]) < 0.35 and y[i] * math.sin(math.pi * cx / SPACING) > 0 for i in crossing):
            gates.append(k)
        if np.any((x - cx) ** 2 + y**2 < 0.10**2):
            hits.append(k)
    fell = bool(np.any(-v[:, 4] < 0.5))
    finished = bool(x[-1] > (CONES + 2) * SPACING)
    return {
        "success": bool(finished and len(gates) == CONES and not hits and not collision and not fell),
        "gates_passed": gates,
        "cone_hits": hits,
        "fell": fell,
        "finished": finished,
        "distance_m": float(x[-1]),
        "rms_path_error_m": float(np.sqrt(np.mean((y - AMPLITUDE * np.sin(math.pi * x / SPACING)) ** 2))),
        "protocol": "physical_cone_slalom_bam_v1",
    }
