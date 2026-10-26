"""Fail-closed dataset-to-artifact contract for the project behavior suite.

The suite may contain different programs and initial-condition recipes, but a
model always targets one embodiment. Registration never grants acceptance.
"""
from __future__ import annotations

import copy
import json

DATASET_PROTOCOL = "project_qualified_dataset_v2"
ACCEPTANCE_SCOPE = "project_tasks_v2"
OBSERVATION_PROFILE = "microduck_rgb_proprio_qualified_command_v2"


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def dataset_artifact_contract(index, embodiment):
    """Validate exact per-task recipes before fitting or exporting a model."""
    from .qualified_teachers import training_contracts

    if (index.get("protocol") != DATASET_PROTOCOL
            or index.get("acceptance_scope") != ACCEPTANCE_SCOPE
            or index.get("observation_profile") != OBSERVATION_PROFILE
            or index.get("environment", {}).get("profile") != "official_bam"
            or index.get("embodiment") != embodiment):
        raise ValueError("qualified data requires its versioned official_bam observation and embodiment contract")
    rows = index.get("episodes", [])
    tasks = {row.get("task_id") for row in rows}
    contracts = index.get("provenance", {}).get("task_contracts", {})
    if not rows or not isinstance(contracts, dict) or set(contracts) != tasks:
        raise ValueError("qualified data must declare exactly one contract for every recorded task")
    expected = training_contracts()
    for task in tasks:
        contract = contracts[task]
        if task not in expected or _canonical(contract) != _canonical(expected[task]):
            raise ValueError(f"qualified task recipe differs from its registered execution version: {task}")
        if not contract.get("initial_conditions") or contract["task_spec"]["embodiment"] != embodiment:
            raise ValueError("qualified model requires explicit per-task conditions and a single embodiment")
    for row in rows:
        if row.get("neutral_command") is not False or row.get("embodiment") != embodiment:
            raise ValueError("qualified data cannot mix neutral commands or robot embodiments")
        recipe = (row.get("initial_condition") or {}).get("recipe")
        if _canonical(recipe) != _canonical(contracts[row["task_id"]]["initial_conditions"]):
            raise ValueError("episode initial conditions differ from the task contract")
    return {
        "acceptance_scope": ACCEPTANCE_SCOPE,
        "observation_profile": OBSERVATION_PROFILE,
        "dataset_protocol": DATASET_PROTOCOL,
        "task_contracts": copy.deepcopy(contracts),
        "trained_tasks": sorted(tasks),
        "initial_conditions": None,  # Recipes live under task_contracts.
        "uses_preset_command13": True,
        "claims_visual_language_autonomy": False,
        "author_program_reproduction": False,
        "counts_as_skill_success": False,
    }
