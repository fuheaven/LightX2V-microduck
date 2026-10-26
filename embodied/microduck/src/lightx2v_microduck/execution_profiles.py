"""Explicit simulator execution recipes shared by CLI and real robotd bridge."""
from __future__ import annotations

import copy
import json
import numpy as np

INITIAL_PROFILES = ("nominal", "preset_initial_v1", "preset_initial_bam_kick_v2", "qualified_initial_v2", "ground_lower_initial_v1")


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def is_project_profile(recipe):
    version = recipe if isinstance(recipe, str) else recipe.get("version") if isinstance(recipe, dict) else None
    return version in ("qualified_initial_v2", "ground_lower_initial_v1")


def resolve_execution_recipe(recipe, task):
    version = recipe if isinstance(recipe, str) else recipe.get("version") if isinstance(recipe, dict) else None
    if version not in INITIAL_PROFILES:
        raise ValueError(f"unknown execution recipe {version!r}; refusing nominal fallback")
    if version == "nominal":
        expected = {"version": "nominal"}
    elif version.startswith("preset_initial"):
        from .preset_campaign import resolve_condition_recipe
        expected = resolve_condition_recipe(recipe)
    else:
        from .qualified_teachers import initial_conditions, training_contracts
        if task not in training_contracts():
            raise ValueError("task is not registered in the qualified project suite")
        expected = initial_conditions(task)
        # qualified_initial_v2 selects the project suite, including its explicitly
        # independent ground_lower recipe; dicts still require an exact match.
        if version != expected["version"] and not (version == "qualified_initial_v2" and task == "ground_lower" and isinstance(recipe, str)):
            raise ValueError("execution recipe belongs to another project task")
    if isinstance(recipe, dict) and canonical(recipe) != canonical(expected):
        raise ValueError("execution recipe differs from its registered version")
    return copy.deepcopy(expected)


def make_execution_environment(task, *, recipe="nominal", asset_root=None, profile="official_bam", scene="flat", render=True):
    resolved = resolve_execution_recipe(recipe, task)
    version = resolved["version"]
    if version != "nominal" and scene != "flat":
        raise ValueError("versioned preset and project execution requires the flat scene")
    if is_project_profile(resolved):
        if profile != "official_bam":
            raise ValueError("qualified project execution requires official_bam")
        from .qualified_teachers import make_environment
        return make_environment(task, asset_root=asset_root, render=render)
    if version.startswith("preset_initial"):
        from .preset_campaign import PresetTrainingEnv
        return PresetTrainingEnv(asset_root=asset_root, profile=profile, render=render, condition_recipe=resolved)
    from .env import MicroDuckEnv
    return MicroDuckEnv(asset_root=asset_root, profile=profile, scene=scene, render=render)


def begin_execution_task(env):
    """Reset the task clock/latch once at runtime handoff, not on inference."""
    if hasattr(env, "_phase_state"):
        env._phase_state = {}
        env._qualified_result = None
    env.begin_task_at_current_pose()


def prescribed_command(env, state):
    """Advance project program once per body tick using public proprioception."""
    if env.task.task_id == "ground_lower":
        from .ground_lower import command_context
        return command_context(env.tick)[1]
    if hasattr(env, "teacher_version"):
        from .qualified_teachers import program_context
        return program_context(env.teacher_version, env.tick, state, env._phase_state, rollers=env.task.embodiment == "rollers")[1]
    if env.task.suite == "author_presets":
        from .tasks import task_command
        return task_command(env.task, env.tick)
    return np.zeros(13, np.float32)
