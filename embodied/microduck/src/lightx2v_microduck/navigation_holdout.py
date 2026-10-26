"""Count navigation layouts from initial physical evidence, independently of scores.

The compiled ``room_clear`` scene has no seeded ball layout: reset places its
balls afterwards. Conversely, a scene's layout ID can change with its seed even
when geometry does not. Neither field alone is an independence certificate.

Use states obtained at task tick zero by the pinned physical replay. These
helpers check the supplied exposure inventory; they do not establish that it
contains a model's entire training/selection lineage, verify a replay, run a
policy, or grant formal acceptance.
"""

import hashlib
import json
import math


COLORS = ("red", "blue", "yellow")
TASKS = frozenset("find_" + color + "_ball" for color in COLORS)
ROOMS = frozenset(("room", "room_clear", "room_obstacles", "room_randomized", "room_randomized_clear",
                   "room_curriculum_zero", "room_curriculum_one", "room_curriculum_two"))
SCHEMA = "microduck_navigation_initial_layout_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _numbers(values, length, label):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError("invalid " + label)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("non-finite or non-numeric " + label)
    # Sub-micrometre floating point noise is not another physical layout.
    return [0.0 if round(x, 6) == 0 else round(x, 6) for x in values]


def initial_layout_identity(world):
    """Fingerprint flat-room geometry at task start, without instruction/seed.

    Ball colors and camera/robot pose get separate identities. Permuting colors,
    selecting another target or rotating the robot does not create another
    independent layout. Objects moving during a rollout cannot become layouts.
    Other scene families need their own geometry specification.
    """
    if not isinstance(world, dict):
        raise ValueError("initial evaluator state must be an object")
    if (type(world.get("tick")) is not int or world["tick"] != 0
            or type(world.get("evaluation_elapsed_s")) not in (int, float)
            or world["evaluation_elapsed_s"] != 0):
        raise ValueError("layout evidence must be captured at task tick zero")
    scene = world.get("scene", {})
    if not isinstance(scene, dict) or scene.get("name") not in ROOMS:
        raise ValueError("unsupported scene family for flat-room layout identity")
    balls = world.get("balls")
    if not isinstance(balls, dict) or set(balls) != set(COLORS):
        raise ValueError("layout requires all three actual ball positions")
    positions = {color: _numbers(balls[color], 3, "ball position") for color in COLORS}
    boxes = []
    obstacles = world.get("obstacles")
    if not isinstance(obstacles, list):
        raise ValueError("layout requires actual obstacle geometry")
    for obstacle in obstacles:
        if not isinstance(obstacle, dict):
            raise ValueError("invalid obstacle geometry")
        xy = _numbers(obstacle.get("xy"), 2, "obstacle position")
        size = _numbers(obstacle.get("half_size"), 2, "obstacle size")
        height = _numbers([obstacle.get("height")], 1, "obstacle height")[0]
        if min(*size, height) <= 0:
            raise ValueError("obstacle size must be positive")
        boxes.append([*xy, *size, height])
    bounds = _numbers(world.get("room_bounds"), 4, "room bounds")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("room bounds must have positive area")
    # All supported rooms use the same flat floor and 3.5 cm physical spheres.
    # XY is the layout variable; vertical settling and target labels are not.
    geometry = {"schema": SCHEMA, "room_bounds": bounds, "ball_radius_m": .035,
                "balls_xy": sorted(pos[:2] for pos in positions.values()),
                "boxes_xy_half_size_height": sorted(boxes)}
    assignment = {color: positions[color][:2] for color in COLORS}
    return {"schema": SCHEMA, "geometry_sha256": _digest(geometry),
            "color_assignment_sha256": _digest(assignment), "geometry": geometry}


def _rows(records, label):
    if not isinstance(records, list):
        raise ValueError(label + " must be an explicit list")
    rows, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(label + " record must be an object")
        scene, seed, task = (record.get(k) for k in ("scene", "seed", "task"))
        if (not isinstance(scene, str) or scene not in ROOMS or type(seed) is not int or seed < 0
                or not isinstance(task, str) or task not in TASKS):
            raise ValueError("invalid scene/seed/task in " + label)
        key = scene, seed, task
        if key in seen:
            raise ValueError("duplicate case in " + label)
        seen.add(key)
        world = record.get("initial_world")
        identity = initial_layout_identity(world)
        if world["scene"]["name"] != scene or world.get("target_color") != task[5:-5]:
            raise ValueError("case identity differs from initial physical evidence")
        rows.append({"scene": scene, "seed": seed, "task": task, **identity})
    return rows


def audit_layout_independence(cohort, exposures, *, declared_exposures=None,
                              unresolved_scene_seeds=None):
    """Find overlap with *all supplied* train, validation and selection cases.

    The caller must independently bind these states to replay receipts and
    establish complete model lineage before using this as an acceptance input.
    Repeated colors contribute trials but never extra layout count. Optional
    report declarations exclude scene/seed pairs even without physical replay.
    An unresolved scene excludes its seed across scenes; neither input invents
    geometry or proves execution. The caller must bind their source census too.
    """
    rows = _rows(cohort, "cohort")
    if not rows:
        raise ValueError("empty held-out cohort")
    exposed = _rows(exposures, "exposures")
    prior_cases = {(r["scene"], r["seed"]) for r in exposed}
    prior_geometry = {r["geometry_sha256"] for r in exposed}
    declarations = [] if declared_exposures is None else declared_exposures
    unknown_seeds = [] if unresolved_scene_seeds is None else unresolved_scene_seeds
    if not isinstance(declarations, list) or not isinstance(unknown_seeds, list):
        raise ValueError("exposure declarations and unresolved seeds must be explicit lists")
    declared_cases = set()
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("invalid exposure declaration")
        scene, seed, task = (declaration.get(k) for k in ("scene", "seed", "task"))
        if (not isinstance(scene, str) or not scene or type(seed) is not int or seed < 0
                or not isinstance(task, str) or task not in TASKS):
            raise ValueError("invalid exposure declaration")
        if (scene, seed, task) in declared_cases:
            raise ValueError("duplicate exposure declaration")
        declared_cases.add((scene, seed, task))
    if any(type(seed) is not int or seed < 0 for seed in unknown_seeds):
        raise ValueError("invalid unresolved scene seed")
    declared_pairs = {(scene, seed) for scene, seed, _ in declared_cases}
    unresolved = set(unknown_seeds)
    # The same scene/seed must describe the same physical layout for each color.
    by_seed = {}
    for row in rows:
        key = row["scene"], row["seed"]
        identity = row["geometry_sha256"], row["color_assignment_sha256"]
        if key in by_seed and by_seed[key] != identity:
            raise ValueError("same scene/seed has inconsistent initial layouts")
        by_seed[key] = identity
    evidence = []
    for row in rows:
        issues = []
        if (row["scene"], row["seed"]) in prior_cases:
            issues.append("scene_seed_used_in_training_or_selection")
        if row["geometry_sha256"] in prior_geometry:
            issues.append("physical_layout_used_in_training_or_selection")
        if (row["scene"], row["seed"]) in declared_pairs:
            issues.append("scene_seed_declared_in_prior_experiment")
        if row["seed"] in unresolved:
            issues.append("seed_declared_without_resolved_scene")
        evidence.append({k: row[k] for k in ("scene", "seed", "task", "geometry_sha256",
                                            "color_assignment_sha256")} | {"issues": issues})
    all_geometry = {r["geometry_sha256"] for r in evidence}
    contaminated = {r["geometry_sha256"] for r in evidence if r["issues"]}
    unseen = all_geometry - contaminated
    return {"schema": "microduck_navigation_layout_independence_audit_v1",
            "status": "overlap_detected" if any(r["issues"] for r in evidence) else "no_overlap_in_supplied_inventory",
            "trials": len(rows), "unique_physical_layouts": len(all_geometry),
            "unique_unseen_physical_layouts": len(unseen),
            "meets_minimum_200_layout_count": len(unseen) >= 200,
            "supplied_exposure_cases": len(exposed), "rows": evidence,
            "supplied_exposure_declarations": len(declared_cases),
            "unresolved_scene_seed_exclusions": len(unresolved),
            "complete_training_and_selection_lineage_verified": False,
            "physical_replay_verified": False, "formal_acceptance": False,
            "hardware_validated": False}
