"""Prepare complete, paired development trials without allocating a device."""

from copy import deepcopy
from pathlib import PurePosixPath

from .navigation import navigation_expiry_limits
from .navigation_history import history_spec_from_contract
from .rendering import validate_camera_offsamples, validate_snapshot_camera_workers


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _absolute(value):
    value = str(value)
    _require(PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts,
             "runtime paths must be absolute without parent traversal")
    return value


def prepare_trial(development, receipt, baseline, *, evaluation_source_manifest, runtime_root, output_root,
                  model_config, repetitions=1, evaluation_source_root=None,
                  include_reference=False, reference_source_files=None, camera_offsamples=None,
                  snapshot_camera_workers=0, runtime_source_manifest=None, runtime_compatibility=None):
    """Preserve the whole admitted development cohort in every expiry/arm pair.

    This only constructs arguments and scoring protocols. A launcher must still
    verify immutable sources, actual artifact bytes, completed development and
    numerical alignment before allocating a GPU. Preparation grants no launch
    permission and makes no capability or performance claim.
    """
    _require(type(repetitions) is int and 1 <= repetitions <= 3, "declare one to three repetitions")
    _require(type(include_reference) is bool, "declare whether to include the reference backend")
    camera_offsamples = validate_camera_offsamples(camera_offsamples)
    snapshot_camera_workers = validate_snapshot_camera_workers(snapshot_camera_workers)
    _require(validate_camera_offsamples(development.get('camera_offsamples')) == camera_offsamples,
             "camera configuration must match the development evaluation; run matched-profile development first")
    _require(development.get("mode") == "offline" and development.get("training") is False
             and development.get("formal_acceptance") is False, "requires a development evaluation protocol")
    _require(development.get("max_ticks") == 1500 and development.get("warmup_ticks") == 50
             and development.get("warmup_predictions") == 1, "task duration or warmup changed")
    _require(development.get("contracts") == ["microduck_navigation_forward04_v1"],
             "requires the trained navigation execution contract")
    evaluator_sha = evaluation_source_manifest.get("files", {}).get(
        "embodied/microduck/src/lightx2v_microduck/env.py")
    declared_root = evaluation_source_manifest.get("source")
    source_root = declared_root if evaluation_source_root is None else _absolute(evaluation_source_root)
    _require(_digest(evaluator_sha) and source_root == development.get("source")
             and (declared_root is None or declared_root == source_root),
             "evaluation source manifest differs from the development runtime")
    camera_sources = None
    runtime_evaluator_sha = evaluator_sha
    if snapshot_camera_workers or runtime_source_manifest is not None:
        _require(isinstance(runtime_source_manifest, dict)
                 and runtime_source_manifest.get('source') == _absolute(runtime_root),
                 'snapshot camera implementation requires its pinned runtime source manifest')
        runtime_files = runtime_source_manifest.get('files', {})
        _require(isinstance(runtime_files, dict), 'runtime source file inventory is invalid')
        runtime_evaluator_sha = runtime_files.get('embodied/microduck/src/lightx2v_microduck/env.py')
        _require(_digest(runtime_evaluator_sha), 'runtime evaluator digest is missing')
        if runtime_evaluator_sha != evaluator_sha:
            _require(camera_offsamples is None and isinstance(runtime_compatibility, dict)
                     and runtime_compatibility.get('schema') == 'microduck_camera_api_compatibility_v1'
                     and runtime_compatibility.get('status') == 'verified'
                     and runtime_compatibility.get('default_camera_only') is True
                     and runtime_compatibility.get('development_evaluator_sha256') == evaluator_sha
                     and runtime_compatibility.get('runtime_evaluator_sha256') == runtime_evaluator_sha,
                     'realtime evaluator differs without verified camera-only source compatibility')
            compatibility_files = runtime_compatibility.get('files', {})
            _require(set(compatibility_files) == {'env.py', 'physics.py', 'tasks.py', 'contracts.py',
                                                 'scenes.py', 'policies.py', 'assets.py'},
                     'camera source compatibility dependency inventory is incomplete')
            prefix = 'embodied/microduck/src/lightx2v_microduck/'
            for name, row in compatibility_files.items():
                _require(isinstance(row, dict)
                         and row.get('development_sha256') == evaluation_source_manifest['files'].get(prefix+name)
                         and row.get('runtime_sha256') == runtime_files.get(prefix+name),
                         'camera source compatibility is bound to different source bytes')
            _require(_digest(runtime_compatibility.get('verifier_sha256'))
                     and runtime_compatibility['verifier_sha256'] == runtime_files.get(
                         prefix+'camera_runtime_compatibility.py'),
                     'camera source compatibility verifier is not pinned')
        if snapshot_camera_workers:
            camera_sources = {'lightx2v_microduck.' + name: runtime_files.get(
                'embodied/microduck/src/lightx2v_microduck/' + name + '.py')
                for name in ('snapshot_camera', 'rendering')}
            _require(all(_digest(digest) for digest in camera_sources.values()),
                     'snapshot camera implementation must be pinned in the runtime source')
    reference_source_files = deepcopy(reference_source_files)
    if include_reference:
        required = {"fastwam/models/wan22/fastwam.py", "fastwam/models/wan22/action_dit.py",
                    "fastwam/models/wan22/mot.py"}
        _require(isinstance(reference_source_files, dict) and required <= set(reference_source_files)
                 and all(isinstance(k, str) and k.startswith("fastwam/") and k.endswith(".py")
                         and ".." not in PurePosixPath(k).parts and _digest(v)
                         for k, v in reference_source_files.items()), "reference sources must be pinned")
    else:
        _require(reference_source_files is None, "reference sources require an explicit reference arm")
    for key in ("weights_sha256", "stats_sha256", "checkpoint_receipt_sha256"):
        _require(_digest(development.get(key)), "missing artifact digest: " + key)
    _require(type(development.get("checkpoint_step")) is int and development["checkpoint_step"] > 0
             and receipt.get("weights_sha256") == development["weights_sha256"]
             and receipt.get("stats_sha256") == development["stats_sha256"]
             and receipt.get("step") == development.get("checkpoint_step"), "checkpoint binding differs")
    history = development.get("input_history")
    _require(isinstance(history, dict) and history == receipt.get("input_history"), "trained input history differs")
    spec = history_spec_from_contract(history)
    _require(history.get("schema") == "microduck_rgb_target_keyframe_v2"
             and history.get("fallback_history_ticks") == 300, "requires trained target memory v2")
    contract = development["contracts"][0]
    binding = {"execution_contract": contract, "coordinate_contract": "microduck_navigation_v1",
               "stats_sha256": development["stats_sha256"]}
    _require(receipt.get("normalization_binding") == binding, "normalization binding differs")
    acceptance = baseline.get("acceptance", {})
    _require(baseline.get("profile") == "official_bam" and _digest(baseline.get("evaluator_sha256"))
             and acceptance.get("max_episode_seconds") == 30
             and acceptance.get("stop_surface_distance_m") == [.25, .40]
             and acceptance.get("stop_hold_seconds") == 2, "baseline acceptance criteria differ")

    groups = {}
    cases = development.get("cases", [])
    _require(isinstance(cases, list) and bool(cases), "development cohort is empty")
    seen = set()
    for case in cases:
        _require(isinstance(case, dict) and set(case) == {"scene", "seed", "task"}, "invalid development case")
        scene, seed, task = case["scene"], case["seed"], case["task"]
        _require(scene in ("room_clear", "room_curriculum_one") and type(seed) is int
                 and seed >= 0 and task in ("find_red_ball", "find_blue_ball", "find_yellow_ball"),
                 "invalid development task or scene")
        key = scene, seed, task
        _require(key not in seen, "duplicate development case")
        seen.add(key)
        group = groups.setdefault(scene, {"seeds": [], "tasks": [], "cases": []})
        if seed not in group["seeds"]:
            group["seeds"].append(seed)
        if task not in group["tasks"]:
            group["tasks"].append(task)
        group["cases"].append(case)
    _require(set(groups) == {"room_clear", "room_curriculum_one"}, "both development scenes are required")
    reserved = baseline.get("independent_final_seeds")
    _require(isinstance(reserved, list) and all(type(v) is int for v in reserved),
             "reserved final seeds must be explicit")
    for scene, group in groups.items():
        expected = {(scene, seed, task) for seed in group["seeds"] for task in group["tasks"]}
        actual = {(scene, c["seed"], c["task"]) for c in group["cases"]}
        _require(actual == expected and len(group["tasks"]) == 3,
                 "CLI task product would add or omit development cases")
        _require(not set(group["seeds"]) & set(reserved), "development overlaps reserved final seeds")

    root, output = PurePosixPath(_absolute(runtime_root)), PurePosixPath(_absolute(output_root))
    config = _absolute(model_config)
    checkpoint = _absolute(development["checkpoint"])
    checkpoint_receipt = _absolute(development["checkpoint_receipt"])
    dataset = str(PurePosixPath(_absolute(development["stats"])).parent)
    backbone = _absolute(development["backbone"])
    protocols, jobs, comparisons = {}, [], []
    history_args = ["--history-offsets-ticks", *map(str, spec.offsets_ticks),
                    "--history-target-keyframe", "--history-target-keyframe-fallback-ticks", "300"]
    for age_label, age_ms in (("age500", 500), ("age1000", 1000), ("no_age", None)):
        expiry = navigation_expiry_limits(age_ms, realtime=True)
        clock = {"schema": "microduck_navigation_execution_clock_v2",
                 "expiry_basis": "fresh_monotonic_at_admission_and_consumption",
                 "max_age_ticks": expiry["max_age_ticks"],
                 "max_wall_age_ns": None if age_ms is None else age_ms * 1_000_000,
                 "source_capture_timestamp_preserved": True}
        if age_ms is None:
            clock.update(observation_age_rejection_enabled=False,
                         ablation="no_age_limit_finite_source_aligned_horizon")
        for scene, group in groups.items():
            key = age_label + "/" + scene
            protocol = {"scene": scene, "profile": "official_bam", "development_seeds": group["seeds"],
                        "tasks": group["tasks"], "independent_final_seeds": reserved,
                        "acceptance": acceptance, "evaluator_sha256": runtime_evaluator_sha,
                        "development_evaluator_sha256": evaluator_sha,
                        "baseline_evaluator_sha256": baseline["evaluator_sha256"],
                        "navigation_contract": contract, "normalization_binding": binding,
                        "stats_sha256": development["stats_sha256"], "input_history": history,
                        "execution_clock": clock,
                        "camera_offsamples": camera_offsamples,
                        "snapshot_camera_workers": snapshot_camera_workers,
                        "expiry_comparison": {"schema": "microduck_realtime_expiry_comparison_v1",
                                              "max_action_age_ms": age_ms, "simulation_only": True}}
            protocols[key] = deepcopy(protocol)
            if camera_sources is not None:
                protocols[key]['camera_runtime_source_sha256'] = deepcopy(camera_sources)
            if include_reference:
                protocols[key]["reference_source_sha256"] = deepcopy(reference_source_files)
            for repeat in range(repetitions):
                pair = key + "/repeat" + str(repeat)
                if include_reference:
                    order = ("reference", "plain", "optimized")
                    arms = order[repeat % 3:] + order[:repeat % 3]
                else:
                    arms = ("plain", "optimized") if repeat % 2 == 0 else ("optimized", "plain")
                for arm in arms:
                    name = pair + "/" + arm
                    argv = ["python", "-u", str(root / "scripts/microduck/fastwam_navigation_eval.py"),
                            "--backend", "reference" if arm == "reference" else "native",
                            "--model-config", config, "--checkpoint", checkpoint,
                            "--checkpoint-receipt", checkpoint_receipt, "--backbone", backbone,
                            "--dataset", dataset, "--output", str(output / name), "--scene", scene,
                            "--tasks", *group["tasks"], "--seeds", *map(str, group["seeds"]),
                            "--mode", "realtime", "--inference-steps", "20", "--max-ticks", "1500",
                            "--warmup-predictions", "1", "--navigation-contract", contract,
                            "--record-frames", "--record-predictions", "--record-model-inputs",
                            "--record-stage-timings", *history_args]
                    argv += (["--no-action-age-limit"] if age_ms is None else ["--max-action-age-ms", str(age_ms)])
                    if camera_offsamples is not None:
                        argv += ["--camera-offsamples", str(camera_offsamples)]
                    if snapshot_camera_workers:
                        argv += ["--snapshot-camera-workers", str(snapshot_camera_workers)]
                    if arm == "optimized":
                        argv += ["--cache-prompt", "--cache-action-context", "--action-cuda-graph"]
                    jobs.append({"name": name, "pair": pair, "arm": arm, "protocol_key": key,
                                 "argv": argv, "expected_cases": len(group["cases"]),
                                 "report": str(output / name / "report.json")})
                comparisons.append({"pair": pair, "protocol_key": key,
                                    "control": str(output / pair / "plain/report.json"),
                                    "candidate": str(output / pair / "optimized/report.json"),
                                    "expected_cases": len(group["cases"])})
                if include_reference:
                    comparisons[-1].update(name=pair+"/native_optimization", comparison_kind="native_optimization",
                                           control_arm="plain", candidate_arm="optimized")
                    comparisons.append({"name": pair+"/reference_port", "pair": pair, "protocol_key": key,
                        "comparison_kind": "reference_port", "control_arm": "reference", "candidate_arm": "plain",
                        "control": str(output / pair / "reference/report.json"),
                        "candidate": str(output / pair / "plain/report.json"), "expected_cases": len(group["cases"])})
    return {"schema": "microduck_same_weight_realtime_trial_v2" if include_reference else "microduck_same_weight_realtime_trial_v1",
            "status": "prepared_not_launched", "backend_arms": ["reference", "plain", "optimized"] if include_reference else ["plain", "optimized"],
            "weights_sha256": development["weights_sha256"], "stats_sha256": development["stats_sha256"],
            "checkpoint_receipt_sha256": development["checkpoint_receipt_sha256"],
            "development_evaluator_sha256": evaluator_sha,
            "runtime_evaluator_sha256": runtime_evaluator_sha,
            "runtime_compatibility": deepcopy(runtime_compatibility),
            "input_history": deepcopy(history), "navigation_contract": contract,
            "camera_offsamples": camera_offsamples,
            "snapshot_camera_workers": snapshot_camera_workers,
            "camera_runtime_source_sha256": deepcopy(camera_sources),
            "required_environment": ({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
                                     if snapshot_camera_workers else {}),
            "repetitions": repetitions, "cases_per_arm_per_expiry": len(cases),
            "distinct_physical_layouts": len({(c["scene"], c["seed"]) for c in cases}),
            "planned_episode_count": sum(j["expected_cases"] for j in jobs),
            "protocols": protocols, "jobs": jobs, "comparisons": comparisons,
            "maximum_concurrent_gpus": 1, "same_gpu_uuid_required_for_all_arms": True,
            "requires_complete_development_and_replay": True,
            "requires_same_weight_numerical_alignment": True,
            "requires_frozen_runtime_sources": True, "gpu_started": False, "training_started": False,
            "performance_claim": False, "formal_acceptance": False, "model_promoted": False,
            "scope": "Full development cohort, separately scored by expiry and scene. Repeated trials do not add unseen physical layouts."}
