"""Paired, held-out development tasks at an identical retained physical state.

This module exports evaluation records, never training examples. The qualified
teacher source is replayed exactly; only instruction and task scoring change.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

from .navigation import DECISION_TICKS, NavigationChunk, intent_from_action, student_state
from .navigation_augmentation import contained, qualified_base, sha256
from .tasks import get_task
from .wam_data import split_for_seed


TASKS = ("find_red_ball", "find_blue_ball", "find_yellow_ball")


def development_sources(dataset, protocol_path, expected_protocol_sha256):
    if sha256(protocol_path) != expected_protocol_sha256:
        raise ValueError("paired evaluation protocol changed")
    protocol = json.loads(Path(protocol_path).read_text())
    index, _ = qualified_base(dataset, protocol)
    seeds = protocol["development_seeds"]
    if (not seeds or len(seeds) != len(set(seeds)) or set(protocol["tasks"]) != set(TASKS)
            or any(split_for_seed(s) != "validation" or s in protocol["independent_final_seeds"]
                   or 10000 <= s < 10200 for s in seeds)):
        raise ValueError("paired evaluation requires only the frozen validation cohort")
    entries = {(e["task"], e["seed"]): e for e in index["episodes"]}
    result = []
    for seed in seeds:
        for task in TASKS:
            entry = entries[task, seed]
            if entry["split"] != "validation" or not entry["eligible"] or not entry["success"]:
                raise ValueError("paired source is not a qualified validation episode")
            result.append(entry)
    return result


def replay_anchor(env, dataset, entry):
    """Rebuild the same pre-stop anchor used by training-side diagnostics."""
    if entry["split"] != "validation" or split_for_seed(entry["seed"]) != "validation":
        raise ValueError("instruction evaluation must not reuse training anchors")
    directory = contained(dataset, entry["path"])
    source = json.loads((directory / "episode.json").read_text())
    if (source["scene"]["name"] != "room_clear" or source["profile"] != "official_bam"
            or source["files"]["navigation.npz"] != entry["sha256"]):
        raise ValueError("source contract differs from the frozen evaluation dataset")
    for name, expected in source["files"].items():
        if sha256(contained(directory, name)) != expected:
            raise ValueError("source episode changed after qualification")
    stops = [i for i, d in enumerate(source["teacher_diagnostics"])
             if d.get("planner_phase", d.get("phase")) == "stop"]
    if not stops:
        raise ValueError("qualified source has no stop event")
    visible = [i for i, d in enumerate(source["teacher_diagnostics"][:stops[0] + 1])
               if d["target_visible_pixels"] >= 16]
    if not visible:
        raise ValueError("source has no target observation before stopping")
    anchor_index = visible[-1]
    anchor_tick = anchor_index * DECISION_TICKS
    env.render_enabled = False
    env.reset(entry["seed"], entry["task"])
    for _ in range(source["warmup_ticks"]):
        env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
    env.begin_task_at_current_pose()
    state_error = actor_error = 0.
    with np.load(directory / "navigation.npz", allow_pickle=False) as data, np.load(directory / "control_trace.npz", allow_pickle=False) as trace:
        actions, states = data["action"], data["proprio"]
        trace_ticks, trace_states, trace_actions = trace["ticks"], trace["state61"], trace["raw_action"]
        for tick in range(anchor_tick):
            if env.terminated or env.tick != tick or trace_ticks[tick] != tick:
                raise ValueError("source replay lost its original task clock")
            state_error = max(state_error, float(np.abs(env.observe().state - trace_states[tick]).max()))
            env.step(intent_from_action(actions[tick // DECISION_TICKS], episode_id=env.episode_id, tick=tick))
            actor_error = max(actor_error, float(np.abs(env.last_action - trace_actions[tick]).max()))
        anchor_error = float(np.abs(student_state(env.observe()) - states[anchor_index]).max())
    if max(state_error, actor_error, anchor_error) > 1e-5:
        raise ValueError("validation anchor does not reproduce the official actor and physics")
    return {"task": entry["task"], "seed": entry["seed"], "tick": anchor_tick,
            "episode_sha256": sha256(directory / "episode.json"),
            "replay_state_max_abs": state_error, "replay_actor_max_abs": actor_error,
            "anchor_state_max_abs": anchor_error}


@contextmanager
def new_instruction_at_anchor(env, task, first_rgb):
    """Preserve complete BAM/ONNX/physics/RNG state across instruction trials."""
    if (env.profile != "official_bam" or env.scene != "room_clear" or env.daemon_backend is not None
            or split_for_seed(env.seed) != "validation" or 10000 <= env.seed < 10200
            or task not in TASKS or env.tick % DECISION_TICKS):
        raise ValueError("paired instruction branches require an official development anchor")
    first_state = student_state(env.observe()).copy()
    snapshot = env.clone_state()
    renderer, rendering = env.physics.renderer, env.render_enabled
    try:
        env.physics.renderer = None
        env.render_enabled = False
        env.task = get_task(task)
        env.episode_id = f"{snapshot['env']['episode_id']}/instruction/{task}"
        env.begin_task_at_current_pose()
        env.physics.render()  # Initialize the branch renderer used by segmentation.
        env._rgb_cache, env._rgb_tick = first_rgb.copy(), 0
        env.render_enabled = True
        observation = env.observe()
        if not np.array_equal(observation.rgb, first_rgb) or not np.array_equal(student_state(observation), first_state):
            raise ValueError("instruction change altered the first sensor inputs")
        yield {"first_rgb_sha256": hashlib.sha256(first_rgb.tobytes()).hexdigest(),
               "first_proprio_sha256": hashlib.sha256(first_state.tobytes()).hexdigest(),
               "original_tick": snapshot["env"]["tick"], "physical_state_reset": False,
               "task_clock_restarted": True, "eligible_training_supervision": False}
    finally:
        if env.physics.renderer is not None and env.physics.renderer is not renderer:
            env.physics.renderer.close()
        env.physics.renderer = None
        env.render_enabled = False
        env.restore_state(snapshot)
        # The parked context still belongs to the original physical pose.
        env.physics.renderer = renderer
        env.render_enabled = rendering


class DevelopmentTeacherPolicy:
    """Fresh privileged teacher for offline feasibility controls, never a model score."""

    metadata = {"model_id": "initial_scan_development_teacher", "uses_privileged_state": True,
                "evidence": "teacher control only; no learned capability claim"}

    def __init__(self, env):
        self.env = env

    def reset(self, episode_id):
        from .navigation_teacher_experiments import InitialScanVisualSearchTeacher

        self.teacher = InitialScanVisualSearchTeacher(self.env)

    def predict(self, observation):
        action = self.teacher.predict(observation)
        # Offline teacher is recomputed every five control ticks. Repetition
        # would fabricate future teacher labels, so return exactly one action.
        return NavigationChunk(action[None], observation.episode_id, observation.tick,
                               observation.monotonic_ns, observation.tick + DECISION_TICKS)
