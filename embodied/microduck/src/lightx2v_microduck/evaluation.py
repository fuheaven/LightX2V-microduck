"""Recomputable episode evaluation; UI and reports share these outcomes."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .contracts import JointResidualChunk, PolicyManifest, RobotContract, SkillIntent
from .data import file_sha256, write_json


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def suite_tasks(suite):
    from .tasks import PROGRAMS

    if suite == "skills13":
        return tuple(PROGRAMS)
    if suite == "skills_legs10":
        return tuple(name for name in PROGRAMS if not name.startswith("roller_"))
    if suite == "skills_rollers3":
        return tuple(name for name in PROGRAMS if name.startswith("roller_"))
    if suite == "visual_goal":
        return ("find_red_ball", "find_blue_ball", "find_yellow_ball")
    raise ValueError(f"unknown suite {suite!r}")


class OfficialTaskPolicy:
    def __init__(self, env, privileged=False):
        from .teachers import ClassicalTeacher

        if env.task.suite == "visual_goal" and not privileged:
            raise ValueError("official policies alone have no visual navigation; select privileged teacher or pixel baseline")
        self.teacher = ClassicalTeacher(env)
        self.manifest = PolicyManifest(
            "privileged" if privileged else "official",
            "joint_residual",
            "privileged_teacher" if privileged else "scripted_program_with_privileged_recovery",
            env.profile,
            embodiment=env.task.embodiment,
            uses_privileged_state=True,
            metadata={"task_controller": "author_programs_and_recovery", "teacher_only": True, "oracle_recovery": True},
        )

    def reset(self, episode_id=""):
        self.teacher.reset(episode_id)

    def predict(self, observation):
        return self.teacher.predict(observation)


def make_policy(name, env, *, model_path=None, backbone_path=None, device="cpu", precision="fp32", compile_expert=False, endpoint=None, planner_model=None, planner_context="feedback", planner_prompt_profile="legacy_v1", planner_service_identity=None):
    if name == "official_project":
        from .qualified_teachers import initial_conditions, make_teacher
        recipe = getattr(env, "condition_recipe", None) or (getattr(env, "initial_condition", None) or {}).get("recipe")
        if recipe != initial_conditions(env.task.task_id):
            raise ValueError("official_project requires its matching qualified execution recipe")
        teacher = make_teacher(env)
        teacher.manifest = replace(teacher.manifest, metadata={**teacher.manifest.metadata, "teacher_only": True})
        return teacher
    if name == "lightx2v_tinyvla" and env.task.suite == "author_presets":
        env.preset_clock = "program"
    if name == 'community:climb-pair':
        from .climb import ClimbPolicy
        return ClimbPolicy(env)
    if name == "community:skill-planner":
        from .skill_planner import SkillPlannerPolicy
        return SkillPlannerPolicy(env, endpoint=endpoint, model=planner_model, context_mode=planner_context,
                                  prompt_profile=planner_prompt_profile,
                                  expected_service_identity=planner_service_identity)
    if name == "community:rgb-tracking":
        from .perception import TrackedBallPolicy
        return TrackedBallPolicy(profile=env.profile)
    if name == "community:duck-detector":
        from .perception import DuckDetectionPolicy
        return DuckDetectionPolicy(env.asset_root, profile=env.profile)
    if name in ("community:tricks", "community:tricks-official"):
        from .tricks import TrickPolicy
        return TrickPolicy(env, baseline=name.endswith("-official"))
    if name in ("official", "privileged"):
        return OfficialTaskPolicy(env, privileged=name == "privileged")
    if name in ("pixel", "pixel_rgb"):
        from .pixel_policy import PixelRulePolicy

        return PixelRulePolicy(profile=env.profile, use_tof=name == "pixel")
    if name == "tinyvla_reference":
        import os

        os.environ.setdefault("LIGHTX2V_IMPORT_MODE", "robotics")
        from lightx2v.models.networks.tinyvla.reference import TinyVLAReferencePolicyHTTP

        if not endpoint:
            raise ValueError("--endpoint must name the pinned author reference service")
        return TinyVLAReferencePolicyHTTP(endpoint)
    if name == "lightx2v_tinyvla":
        if endpoint:
            from .remote import TinyVLAHTTPPolicy

            return TinyVLAHTTPPolicy(endpoint)
        import os

        os.environ.setdefault("LIGHTX2V_IMPORT_MODE", "robotics")
        from lightx2v.models.networks.tinyvla.runtime import MicroDuckTinyVLAPolicy, TinyVLARuntime

        if not model_path:
            raise ValueError("--model-path must identify a complete model artifact, including normalization")
        runtime = TinyVLARuntime.from_directory(model_path, backbone_path=backbone_path, device=device, precision=precision, compile_expert=compile_expert)
        return MicroDuckTinyVLAPolicy(runtime)
    if name.startswith("onnx:"):
        from .policies import OnnxLocomotionPolicy

        return OnnxLocomotionPolicy(name.split(":", 1)[1], profile=env.profile, embodiment=env.task.embodiment)
    if name.startswith("community:"):
        from .community import CommunityPolicy, FollowOraclePolicy

        identifier = name.split(":", 1)[1]
        if identifier in ("official-velstand", "duckbatch-128"):
            # Run the declared command sequence without hidden truth-based
            # recovery restarting the task clock after a fall.
            env.preset_clock = "program"
        return FollowOraclePolicy(env, env.asset_root) if identifier in ("follow-v4", "follow-v4-oracle") else CommunityPolicy(identifier, asset_root=env.asset_root)
    raise ValueError(f"unknown policy {name!r}; no silent model fallback")


def actor_ablation(observation, settings):
    observation = replace(observation, state=observation.state.copy())
    if settings.get("rgb") == "black":
        if observation.rgb is None:
            raise ValueError("RGB ablation requires a camera")
        observation = replace(observation, rgb=np.zeros_like(observation.rgb))
    elif settings.get("rgb") == "flip":
        observation = replace(observation, rgb=observation.rgb[:, ::-1].copy())
    if "instruction" in settings:
        observation = replace(observation, instruction=settings["instruction"])
    if "command" in settings:
        state = observation.state.copy()
        command = np.asarray(settings["command"], dtype=np.float32)
        if command.shape != (13,):
            raise ValueError("ablation command must be 13D")
        state[48:] = command
        observation = replace(observation, state=state)
    return observation


@dataclass(frozen=True)
class EvaluationConfig:
    tasks: tuple[str, ...]
    seeds: tuple[int, ...] = (0,)
    max_steps: int | None = None
    execution_prefix: int = 5
    record_every: int = 5
    ablation: dict | None = None
    require_primary_eligibility: bool = False
    record_observer: bool = True

    def __post_init__(self):
        if not self.tasks or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("tasks and unique seeds are required")
        if self.execution_prefix < 1 or self.record_every < 1 or (self.max_steps is not None and self.max_steps < 1):
            raise ValueError("evaluation steps must be positive")


_OBSERVATION_SENSORS = {
    "official_61d": ("proprioception",),
    "rgb_proprio": ("rgb", "proprioception"),
    "rgb_proprio_tof": ("rgb", "proprioception", "tof"),
    "microduck_rgb_proprio_neutral_goal_v1": ("rgb", "proprioception"),
    "microduck_rgb_proprio_preset_command_v1": ("rgb", "proprioception"),
    "microduck_rgb_proprio_neutral_foundation_v1": ("rgb", "proprioception"),
    "microduck_rgb_proprio_preset_foundation_walk_v1": ("rgb", "proprioception"),
    "microduck_rgb_proprio_preset_ground_lower_v1": ("rgb", "proprioception"),
    "microduck_rgb_proprio_qualified_command_v2": ("rgb", "proprioception"),
    "tinyvla_author_compat": ("rgb", "proprioception"),
    "author": ("rgb", "proprioception"),
    "privileged_teacher": ("proprioception", "simulator_truth"),
    "scripted_program_with_privileged_recovery": ("proprioception", "simulator_truth"),
    "lab_obs_v2_80d_oracle_detector_true_odom": ("proprioception", "tof", "simulator_truth"),
}
_SENSOR_ALIASES = {"proprio": "proprioception", "state61": "proprioception", "imu": "proprioception", "camera": "rgb"}
_SENSOR_FIELDS = (
    "sensor_group",
    "policy_sensors",
    "policy_inputs",
    "declared_task_sensors",
    "extra_sensors",
    "uses_privileged_state",
    "teacher_only",
    "sensor_contract_declared",
    "eligible_for_primary_benchmark",
    "primary_ineligibility_reasons",
    "policy_priors",
    "sensor_availability",
)


def _sensor_contract(manifest, env, observation=None):
    """Classify declared inputs; unknown contracts never silently join the main score."""
    known = _OBSERVATION_SENSORS.get(manifest.observation_profile)
    explicit = manifest.metadata.get("required_sensors")
    if explicit is not None and (not isinstance(explicit, (list, tuple)) or not all(isinstance(s, str) for s in explicit)):
        raise ValueError("policy required_sensors must be a list of sensor names")
    # Explicit declarations may add sensors, never hide a known profile's ToF.
    sensors = {_SENSOR_ALIASES.get(s, s) for s in (*(known or ()), *(explicit or ()))}
    privileged = bool(manifest.uses_privileged_state or "simulator_truth" in sensors)
    teacher_only = bool(manifest.metadata.get("teacher_only") or privileged)
    if privileged:
        sensors.add("simulator_truth")
    declared = getattr(env.task, "allowed_sensors", None)
    allowed = {_SENSOR_ALIASES.get(s, s) for s in (declared or ())}
    extra = sorted(sensors - allowed)
    contract_declared = known is not None or explicit is not None
    reasons = []
    if privileged:
        reasons.append("privileged_state")
    if teacher_only:
        reasons.append("teacher_reference")
    if not contract_declared:
        reasons.append("undeclared_policy_sensors")
    if declared is None:
        reasons.append("undeclared_task_sensors")
    if extra:
        reasons.append("sensors_outside_task_contract")
    availability = (
        {}
        if observation is None
        else {"rgb": observation.rgb is not None, "proprioception": observation.state.shape == (61,), "tof": observation.tof is not None and observation.tof_metadata.get("available", False)}
    )
    if any(not availability.get(s, False) for s in sensors & {"rgb", "proprioception", "tof"}) and observation is not None:
        reasons.append("required_sensor_unavailable")
    group = "privileged" if privileged else "undeclared" if not contract_declared else "_".join(s for s in ("rgb", "proprioception", "tof") if s in sensors) or "other_declared"
    inputs = sorted(sensors)
    if "rgb" in sensors and manifest.metadata.get('uses_language', True):
        inputs.append("language")
    priors = dict(manifest.metadata.get("policy_priors", {}))
    if manifest.policy_id in ("pixel_color_fsm", "pixel_color_tof_fsm"):
        priors.update(ball_radius_prior_m=manifest.metadata.get("ball_radius_prior_m", 0.035), camera_intrinsics=True, camera_relative_fk=True, saturated_color_sphere_geometry=True)
    return {
        "sensor_group": group,
        "policy_sensors": sorted(sensors),
        "policy_inputs": inputs,
        "declared_task_sensors": sorted(allowed),
        "extra_sensors": extra,
        "uses_privileged_state": privileged,
        "teacher_only": teacher_only,
        "sensor_contract_declared": contract_declared,
        "eligible_for_primary_benchmark": not reasons,
        "primary_ineligibility_reasons": reasons,
        "policy_priors": priors,
        "sensor_availability": availability,
    }


def _restrict_actor_sensors(observation, validation):
    """A declared RGB policy never receives an extra ToF frame by accident."""
    if not validation["sensor_contract_declared"]:
        return observation  # Auxiliary only; do not invent an unknown adapter ABI.
    sensors = validation["policy_sensors"]
    # Intrinsics/relative FK are public calibration, not image pixels. Keep
    # them for the explicitly privileged follow oracle's projection too.
    return replace(observation, rgb=observation.rgb if "rgb" in sensors else None, tof=observation.tof if "tof" in sensors else None, tof_metadata=observation.tof_metadata if "tof" in sensors else {})


def _benchmark_summary(records):
    primary = [r for r in records if r.get("eligible_for_primary_benchmark") is True]
    primary_rate = sum(bool(r["success"]) for r in primary) / len(primary) if primary else None
    groups = sorted({r["sensor_group"] for r in records})
    return {
        "success_rate": primary_rate,
        "primary_success_rate": primary_rate,
        "success_rate_scope": "eligible_primary_episodes_only",
        "primary_count": len(primary),
        "system_success_rate": sum(bool(r["success"]) for r in records) / len(records) if records else None,
        "system_count": len(records),
        "count": len(records),
        "auxiliary_count": len(records) - len(primary),
        "eligible_for_primary_benchmark": bool(records) and len(primary) == len(records),
        "sensor_group": groups[0] if len(groups) == 1 else "mixed",
        "sensor_groups": groups,
        "policy_sensors": sorted({s for r in records for s in r["policy_sensors"]}),
        "policy_inputs": sorted({s for r in records for s in r["policy_inputs"]}),
        "uses_privileged_state": any(r["uses_privileged_state"] for r in records),
        "sensor_contract_declared": all(r["sensor_contract_declared"] for r in records),
        "policy_priors_by_policy": {r["policy"]: r["policy_priors"] for r in records},
        "declared_task_sensors": sorted({s for r in records for s in r["declared_task_sensors"]}),
        "extra_sensors": sorted({s for r in records for s in r["extra_sensors"]}),
        "primary_ineligibility_reasons": sorted({s for r in records for s in r["primary_ineligibility_reasons"]}),
    }


def validate_policy_environment(manifest, env, observation=None):
    """Enforce declared contracts without inventing an undeclared camera ABI."""
    if manifest is None:
        raise ValueError("evaluation policies must declare a PolicyManifest")
    if manifest.embodiment != env.task.embodiment:
        raise ValueError(f"policy embodiment {manifest.embodiment} does not match task {env.task.embodiment}")
    if manifest.metadata.get("acceptance_scope") == "project_tasks_v2" and manifest.observation_profile != "official_61d":
        from .qualified_teachers import training_contracts
        import json
        task_id = env.task.task_id
        expected_contract = training_contracts().get(task_id)
        actual_contract = manifest.metadata.get("task_contracts", {}).get(task_id)
        recipe = getattr(env, "condition_recipe", None) or (getattr(env, "initial_condition", None) or {}).get("recipe")
        if (expected_contract is None or json.dumps(actual_contract, sort_keys=True) != json.dumps(expected_contract, sort_keys=True)
                or recipe != expected_contract["initial_conditions"]):
            raise ValueError("project policy task program or initial conditions differ from the active environment")
    expected = manifest.metadata.get("embodiment_digest")
    if expected and expected != RobotContract(embodiment=env.task.embodiment).sha256:
        raise ValueError("policy embodiment digest differs from the active robot contract")
    required_scene = manifest.metadata.get("required_scene")
    required_scene = {"basketball_apex_supported_start": "ball_apex"}.get(required_scene, required_scene)
    if required_scene and env.scene != required_scene:
        raise ValueError(f"policy requires scene {required_scene!r}, got {env.scene!r}")
    simulation_profile = manifest.metadata.get("simulation_profile")
    if simulation_profile and simulation_profile != env.profile:
        raise ValueError(f"policy requires simulation profile {simulation_profile!r}, got {env.profile!r}")
    known_physics = {"official_bam", "tinyvla_author_compat", "official_daemon", "robotd_bam"}
    if manifest.control_profile in known_physics and manifest.control_profile != env.profile:
        raise ValueError("declared policy control profile differs from the environment")
    return {
        "embodiment_digest_verified": bool(expected),
        "required_scene": required_scene,
        "simulation_profile_declared": simulation_profile,
        "active_simulation_profile": env.profile,
        "camera_profile_verified": False if simulation_profile is None else None,
        **_sensor_contract(manifest, env, observation),
    }


def validate_action_source(action, observation):
    if action.episode_id != observation.episode_id:
        raise ValueError("cross-episode policy action source")
    if action.source_tick != observation.tick:
        age = "future" if action.source_tick > observation.tick else "stale"
        raise ValueError(f"{age} policy action source does not match the current observation tick")
    if isinstance(action, JointResidualChunk) and abs(action.source_time_s - observation.sim_time_s) > 1e-8:
        raise ValueError("policy action source time differs from the current observation")


def run_episode(env, policy, directory, config, *, task, seed, live=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    observation = env.observe()
    policy.reset(observation.episode_id)
    manifest = getattr(policy, "manifest", None)
    contract_validation = validate_policy_environment(manifest, env, observation)
    if config.require_primary_eligibility and not contract_validation["eligible_for_primary_benchmark"]:
        raise ValueError(f"policy is ineligible for the primary benchmark: {contract_validation['primary_ineligibility_reasons']}")
    if hasattr(policy, 'set_realtime'):
        policy.set_realtime(live is not None)
    states, raw, applied, ticks, latency, frames, decisions = [], [], [], [], [], [], []
    frame_ticks = []
    observer_frames, observer_frame_ticks, observer_error = [], [], None
    exception = None
    if live is not None:
        live.start(policy, observation)
    started = time.perf_counter()
    limit = config.max_steps or round(env.task.duration_s / 0.02)
    try:
        while not env.terminated and env.tick < limit:
            if live is not None and live.cancelled():
                exception = "cancelled_by_user"
                break
            actor = _restrict_actor_sensors(actor_ablation(observation, config.ablation or {}), contract_validation)
            before = time.perf_counter()
            action = policy.predict(actor) if live is None else live.predict(actor)
            latency.append((time.perf_counter() - before) * 1000)
            if isinstance(action, JointResidualChunk):
                action = replace(action, actions=action.actions[: config.execution_prefix])
            elif isinstance(action, SkillIntent):
                action = replace(action, duration_ticks=min(action.duration_ticks, config.execution_prefix))
            else:
                raise TypeError("policies must return JointResidualChunk or SkillIntent")
            validate_action_source(action, observation)
            decisions.append(
                {
                    "tick": observation.tick,
                    "kind": type(action).__name__,
                    "action_profile": getattr(action, 'action_profile', None),
                    "diagnostics": serializable(getattr(action, "diagnostics", getattr(policy, "last_diagnostics", {}))),
                    "skill": action.skill if isinstance(action, SkillIntent) else None,
                    "command": action.command.tolist() if isinstance(action, SkillIntent) else None,
                }
            )
            if env.task.suite == "community_perception":
                decisions[-1]["perception_truth"] = env.perception_truth()
            if isinstance(action, JointResidualChunk):
                rows = action.actions
            else:
                rows = [None] * action.duration_ticks
            for index, row in enumerate(rows):
                if live is not None and live.cancelled():
                    exception = "cancelled_by_user"
                    break
                if env.terminated or env.tick >= limit:
                    break
                current = env.observe()
                states.append(current.state)
                ticks.append(current.tick)
                if current.rgb is not None and current.tick % config.record_every == 0:
                    frames.append(current.rgb)
                    frame_ticks.append(current.tick)
                    observer = getattr(env, "render_observer", None)
                    if config.record_observer and callable(observer) and getattr(env, "daemon_backend", None) is None and observer_error is None:
                        try:
                            frame = observer()
                            if frame is not None:
                                frame = np.asarray(frame)
                                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                                    raise ValueError("observer image must be uint8 HxWx3")
                                observer_frames.append(frame.copy())
                                observer_frame_ticks.append(current.tick)
                        except Exception as exc:
                            # A review-image failure cannot change the physical
                            # task score. Preserve its explicit recording error.
                            observer_error = f"{type(exc).__name__}: {exc}"
                # Each row gets the same source identity and time semantics.
                step = replace(action, actions=action.actions[: index + 1]) if isinstance(action, JointResidualChunk) else replace(action, duration_ticks=1)
                observation = env.step(step)
                if observation.tick <= current.tick:
                    raise RuntimeError("offline simulator did not advance after an accepted action")
                status = env.evaluator_state()
                raw.append(status["previous_raw_action"])
                applied.append(status["joint_targets"])
                if live is not None:
                    live.after_step(observation)
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    if hasattr(policy, 'close'):
        policy.close()
    final = serializable(env.evaluator_state())
    outcome = bool(final["success"] and exception is None and env.terminated)
    if env.task.suite == "community_perception":
        from .perception import iou
        visible, matches, false_positives = 0, 0, 0
        for decision in decisions:
            truth = decision.get("perception_truth", {}).get("bbox")
            predictions = [t for t in decision['diagnostics'].get('perception', {}).get('tracks', []) if t['age_ticks'] == 0]
            matched = truth is not None and any(iou(truth, t['bbox']) >= .5 for t in predictions)
            visible += truth is not None
            matches += matched
            false_positives += max(0, len(predictions) - int(matched))
        quality = visible >= 10 and matches / max(visible, 1) >= .8 and false_positives == 0
        outcome = bool(outcome and quality)
        final['perception_metrics'] = {"visible_frames": visible, "matched_frames": matches,
                                       "recall_at_iou_0_5": matches / max(visible, 1), "false_positives": false_positives,
                                       "target": "fixed_root_official_duck", "success": outcome}
        final['success'] = outcome
    system_outcome = outcome
    policy_intervention_ticks = sum(bool(d['diagnostics'].get('intervention')) for d in decisions)
    if policy_intervention_ticks:
        outcome = False
    reason = exception or ("policy_intervention" if policy_intervention_ticks else None) or (
        "success"
        if outcome
        else final.get("termination_reason") or ("collision" if final["collision"] else "fall" if final["ever_fallen"] else "intervention" if final["recovery_events"] else "timeout_or_task_failure")
    )
    np.savez_compressed(
        directory / "trajectory.npz",
        state61=np.asarray(states, np.float32),
        raw_action14=np.asarray(raw, np.float32),
        applied_target14=np.asarray(applied, np.float32),
        tick=np.asarray(ticks, np.int64),
        rgb=np.asarray(frames, np.uint8),
        frame_tick=np.asarray(frame_ticks, np.int64),
        observer_rgb=np.asarray(observer_frames, np.uint8),
        observer_frame_tick=np.asarray(observer_frame_ticks, np.int64),
        evaluator_trace12=np.asarray(getattr(env, '_trajectory', []), np.float64),
    )
    if frames:
        from PIL import Image

        Image.fromarray(frames[0]).save(directory / "observation.jpg")
    if observer_frames:
        from PIL import Image

        Image.fromarray(observer_frames[0]).save(directory / "observer.jpg")
    report = {
        "schema_version": 1,
        "task": task,
        "seed": seed,
        "policy": asdict(manifest),
        "contract": asdict(RobotContract(embodiment=env.task.embodiment)),
        "config": asdict(config),
        "policy_environment_validation": contract_validation,
        **{key: contract_validation[key] for key in _SENSOR_FIELDS},
        "profile": env.profile,
        "scene": env.scene,
        "success": outcome,
        "author_success": final.get("author_success"),
        "scene_receipt": final.get("scene"),
        "observer_recording": {"requested": config.record_observer, "frames": len(observer_frames), "error": observer_error, "source": "MuJoCo free camera for offline review", "policy_input": False},
        "system_success": system_outcome,
        "policy_intervention_decisions": policy_intervention_ticks,
        "primary_success": outcome if contract_validation["eligible_for_primary_benchmark"] else None,
        "termination_reason": reason,
        "wall_time_s": time.perf_counter() - started,
        "inference_ms": {"count": len(latency), "p50": float(np.percentile(latency, 50)) if latency else None, "p95": float(np.percentile(latency, 95)) if latency else None},
        "final": final,
        "decisions": decisions,
        "trajectory_sha256": file_sha256(directory / "trajectory.npz"),
        "evidence": "simulation_evaluation",
        "hardware_validated": False,
    }
    if live is not None:
        live.finish(report)
    write_json(directory / "episode.json", serializable(report))
    return report


def evaluate(env_factory, policy_factory, output_dir, config):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise FileExistsError(f"evaluation already exists: {output}")
    records = []
    for task in config.tasks:
        for seed in config.seeds:
            env = env_factory()
            try:
                env.reset(seed, task)
                policy = policy_factory(env)
                relative = Path(task) / f"seed-{seed}"
                record = run_episode(env, policy, output / relative, config, task=task, seed=seed)
                records.append(
                    {
                        "path": str(relative / "episode.json"),
                        "task": task,
                        "seed": seed,
                        "success": record["success"],
                        "system_success": record["system_success"],
                        "primary_success": record["primary_success"],
                        **{key: record[key] for key in _SENSOR_FIELDS},
                        "author_success": record["author_success"],
                        "policy": record["policy"]["policy_id"],
                        "reason": record["termination_reason"],
                    }
                )
                write_json(output / "progress.json", {"episodes": records, "complete": False})
            finally:
                env.close()
    result = {"schema_version": 1, "config": asdict(config), "episodes": records, "complete": True, **_benchmark_summary(records), "hardware_validated": False}
    write_json(output / "evaluation.json", result)
    return result


def compare(experiment_path, *, asset_root=None):
    import yaml

    from .env import MicroDuckEnv

    spec_path = Path(experiment_path).resolve()
    spec = yaml.safe_load(spec_path.read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    output = (spec_path.parent / spec["output_dir"]).resolve()
    config = EvaluationConfig(
        tasks=tuple(spec.get("tasks") or suite_tasks(spec["suite"])),
        seeds=tuple(spec["seeds"]),
        max_steps=spec.get("max_steps"),
        execution_prefix=spec.get("execution_prefix", 5),
        require_primary_eligibility=bool(spec.get("require_primary_eligibility", False)),
        record_observer=bool(spec.get("record_observer", True)),
    )
    results = []
    for entry in spec["policies"]:
        name = entry["name"]
        options = {k: v for k, v in entry.items() if k not in ("name", "id", "ablation")}
        identifier = entry.get("id", name.replace(":", "_"))
        if Path(identifier).name != identifier:
            raise ValueError("policy experiment id must be a single directory name")
        result = evaluate(
            lambda: MicroDuckEnv(asset_root=asset_root, profile=spec.get("profile", "official_bam"), scene=spec.get("scene", "room")),
            lambda env: make_policy(name, env, **options),
            output / identifier,
            replace(config, ablation=entry.get("ablation")),
        )
        results.append({"id": identifier, "evaluation": f"{identifier}/evaluation.json", **_benchmark_summary(result["episodes"])})
    report = {
        "schema_version": 1,
        "experiment": spec,
        "results": results,
        "primary_leaderboard": [r for r in results if r["eligible_for_primary_benchmark"]],
        "auxiliary_results": [r for r in results if not r["eligible_for_primary_benchmark"]],
        "success_rate_scope": "eligible_primary_episodes_only",
        "hardware_validated": False,
    }
    write_json(output / "comparison.json", report)
    return report
