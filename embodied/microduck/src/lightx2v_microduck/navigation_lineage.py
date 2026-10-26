"""Inventory declared navigation checkpoint ancestry and its data-index union.

This metadata audit deliberately does not certify weight contents, all model
selection experiments, foundation pretraining, physical replay or capabilities.
It helps an eventual held-out evaluator avoid inspecting only the latest recipe.
"""

import hashlib
import json
import math
from pathlib import Path

import yaml


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("invalid content digest")
    return value


def _contained(root, path):
    value = Path(path).resolve()
    if not value.is_relative_to(root) or not value.is_file():
        raise ValueError("missing or external lineage input: " + str(value))
    return value


def _bound_file(root, path, expected=None):
    path = _contained(root, path)
    digest = sha256(path)
    if expected is not None and digest != _digest(expected):
        raise ValueError("lineage input changed: " + str(path))
    return {"path": str(path), "sha256": digest}


def _correction_contract(root, recipe):
    """Resolve an append-only correction tail, including its camera prefixes.

    These metadata checks use the training loader's contract. They do not load
    tensors or assert that every indexed example was sampled by an optimizer.
    """
    from .navigation_correction_tail import TAIL_KEYS, correction_tail_contract

    train, validation = recipe['data']['train'], recipe['data']['val']
    audit = recipe['audit']
    if not any(data.get(key) is not None for data in (train, validation) for key in TAIL_KEYS):
        if audit.get('correction_source_index_sha256') or audit.get('correction_parent_recipe_sha256'):
            raise ValueError('correction audit lacks its explicit data sources')
        return None
    for key in ('correction_parent_recipe_path', 'correction_parent_recipe_sha256'):
        if train.get(key) is None or train[key] != validation.get(key):
            raise ValueError('correction train/validation parent recipes differ')
    # Containment is checked before the training contract opens any files.
    _bound_file(root, train['correction_parent_recipe_path'], train['correction_parent_recipe_sha256'])
    for split, data in (('train', train), ('validation', validation)):
        if data.get('split', 'train') != split or not isinstance(data.get('correction_sources'), list):
            raise ValueError('correction sources require their explicit training/validation split')
        for source in data['correction_sources']:
            if not isinstance(source, dict) or not isinstance(source.get('root'), str):
                raise ValueError('correction source lacks its dataset path')
            _bound_file(root, Path(source['root']) / 'index.json', _digest(source.get('index_sha256')))
            if source.get('kind') == 'clear_recovery':
                if not isinstance(source.get('history_prefix_root'), str):
                    raise ValueError('clear correction lacks its camera prefix path')
                _bound_file(root, Path(source['history_prefix_root']) / 'index.json',
                            _digest(source.get('history_prefix_sha256')))
    tail = correction_tail_contract(train)
    correction_tail_contract(validation)
    if (audit.get('correction_source_index_sha256') != [s['index_sha256'] for s in tail['sources']]
            or audit.get('correction_parent_recipe_sha256') != tail['parent_recipe_sha256']):
        raise ValueError('correction sources differ from their recipe audit')
    return tail


def _recipe_inputs(root, recipe):
    """Resolve supported dataset families, preserving declared train/val roots."""
    audit = recipe["audit"]
    if any(audit.get(k) for k in ("augmentation_index_sha256", "retarget_index_sha256",
                                  "qualified_teacher_index_sha256")):
        raise ValueError("unsupported data family requires an explicit lineage resolver")
    indices = {}
    for split in ("train", "val"):
        data = recipe["data"][split]
        base = _bound_file(root, Path(data["root"]) / "index.json", audit["index_sha256"])
        indices[base["sha256"]] = base | {"kind": "base"}
        _bound_file(root, data["pretrained_norm_stats"], audit["stats_sha256"])
        for roots_key, digest_list, kind in (
            ("recovery_roots", audit.get("recovery_index_sha256", []), "recovery"),
            ("history_prefix_roots", data.get("history_prefix_sha256", []), "history_prefix"),
            ("guarded_recovery_roots", audit.get("guarded_recovery_index_sha256", []), "guarded_recovery"),
        ):
            paths = data.get(roots_key, [])
            if roots_key == 'guarded_recovery_roots':
                if data.get('guarded_recovery_sha256', []) != (digest_list if split == 'train' else []):
                    raise ValueError('guarded recovery hashes disagree with recipe audit')
                if split == 'val' and paths:
                    raise ValueError('guarded recovery cannot enter validation')
            # Recovery labels are train-only. No roots in validation means no
            # validation use; it does not erase their training provenance.
            if split == "val" and not paths:
                continue
            if not isinstance(paths, list) or not isinstance(digest_list, list) or len(paths) != len(digest_list):
                raise ValueError("data roots and hashes disagree: " + roots_key)
            for path, digest in zip(paths, digest_list):
                entry = _bound_file(root, Path(path) / "index.json", digest)
                indices[entry["sha256"]] = entry | {"kind": kind}
        extended = data.get("extended_release_root")
        if extended:
            digest = data.get("extended_release_sha256")
            if digest != audit.get("extended_teacher_index_sha256"):
                raise ValueError("extended teacher binding disagrees with recipe audit")
            entry = _bound_file(root, Path(extended) / "index.json", digest)
            indices[entry["sha256"]] = entry | {"kind": "extended_teacher"}
    tail = _correction_contract(root, recipe)
    if tail is not None:
        for source in tail['sources']:
            entry = _bound_file(root, Path(source['root']) / 'index.json', source['index_sha256'])
            # Preserve the established family names consumed by physical-layout
            # inventories; the additional fields identify the correction stage.
            kind = 'recovery' if source['kind'] == 'clear_recovery' else 'guarded_recovery'
            indices[entry['sha256']] = entry | dict(kind=kind, source_role='correction_tail',
                student_checkpoint_sha256=source['student_checkpoint_sha256'])
            if source['kind'] == 'clear_recovery':
                prefix = _bound_file(root, Path(source['history_prefix_root']) / 'index.json',
                                     source['history_prefix_sha256'])
                indices[prefix['sha256']] = prefix | dict(kind='history_prefix',
                    source_role='correction_tail', recovery_index_sha256=entry['sha256'])
    return indices


def _verify_extended_metadata(root, receipt, recipe):
    from .navigation_extended_training import MEMORY_REHEARSAL_TARGET, verify_extended_input_history

    data = recipe["data"]["train"]
    extended = data.get("extended_release_root")
    contract = receipt.get("extended_navigation_training")
    tail = _correction_contract(root, recipe)
    recorded_tail = contract.get('correction_tail') if isinstance(contract, dict) else None
    if tail != recorded_tail:
        raise ValueError('checkpoint correction provenance differs from recipe')
    if not extended:
        if contract is not None:
            raise ValueError("checkpoint declares extended data missing from recipe")
        return
    if not isinstance(contract, dict):
        raise ValueError("extended recipe lacks its checkpoint provenance")
    memory = data.get('_target_') == MEMORY_REHEARSAL_TARGET
    verify_extended_input_history(receipt)
    for split in ('train', 'val'):
        item = recipe['data'][split]
        if memory:
            if (item.get('_target_') != MEMORY_REHEARSAL_TARGET
                    or item.get('memory_input_contract') != receipt.get('input_history')
                    or item.get('history_offsets_ticks') != [300, 150, 50, 0]
                    or item.get('history_target_keyframe', False) is not False
                    or item.get('history_target_keyframe_fallback_ticks') is not None):
                raise ValueError('memory checkpoint and paired dataset input contracts differ')
        elif item.get('memory_input_contract') is not None or item.get('_target_') == MEMORY_REHEARSAL_TARGET:
            raise ValueError('memory inputs require the explicit memory rehearsal dataset')
    expected = {
        "schema": ('microduck_memory_rehearsal_training_v1' if memory
                   else 'microduck_rehearsal_navigation_training_v1'),
        "execution_contract": data["navigation_contract"],
        "base_index_sha256": recipe["audit"]["index_sha256"],
        "extended_teacher_index_sha256": data["extended_release_sha256"],
        "retained_recipe_sha256": data["retained_recipe_sha256"],
        "retained_sampling": "complete_recipe_once_in_original_order",
    }
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("extended checkpoint provenance differs from recipe")
    appended = data.get('appended_recovery_sha256') or []
    expected_append = (dict(index_sha256=list(appended),
                           student_checkpoint_sha256=data['appended_recovery_checkpoint_sha256'],
                           history_prefix_sha256=list(data['history_prefix_sha256'][-len(appended):]),
                           sampling='after_complete_retained_and_extended_samples') if appended else None)
    if contract.get('appended_recoveries') != expected_append:
        raise ValueError('checkpoint appended recovery provenance differs from recipe')
    from .navigation_guarded_recovery import guarded_recovery_config
    guarded = guarded_recovery_config(data.get('guarded_recovery_roots', []),
        data.get('guarded_recovery_sha256', []), data.get('guarded_recovery_checkpoint_sha256'), 'train',
        checkpoints=data.get('guarded_recovery_checkpoint_sha256s'))
    if contract.get('guarded_recoveries') != guarded:
        raise ValueError('checkpoint guarded recovery provenance differs from recipe')
    for key, path in (("base_root", data["root"]), ("extended_root", extended),
                      ("retained_recipe_path", data["retained_recipe_path"])):
        if Path(contract.get(key, "")).resolve() != Path(path).resolve():
            raise ValueError("extended checkpoint source path differs from recipe")
    _bound_file(root, data["retained_recipe_path"], data["retained_recipe_sha256"])
    binding = {"execution_contract": data["navigation_contract"],
               "coordinate_contract": "microduck_navigation_v1",
               "stats_sha256": recipe["audit"]["stats_sha256"]}
    if (contract.get("normalization_binding") != binding
            or receipt.get("normalization_binding") != binding
            or receipt.get("navigation_contract") != data["navigation_contract"]):
        raise ValueError("extended checkpoint execution/normalization binding differs")


def _retention_teacher(root, receipt, recipe):
    """Bind a distillation teacher without silently hiding a second ancestry.

    Only a teacher already in the declared checkpoint chain is supported. The
    caller resolves that constraint after following the entire chain. Recorded
    source and sampling hashes are metadata here, not new code/data execution.
    """
    from .navigation_axis_loss import verify_recorded_axis_objective

    axis_objective = verify_recorded_axis_objective(recipe, receipt)
    config = recipe.get("retention")
    manifest = receipt.get("retention", {"kind": "disabled"})
    if config is None:
        if manifest != {"kind": "disabled"}:
            raise ValueError("checkpoint retention teacher is missing from recipe")
        return None
    if (not isinstance(config, dict) or set(config) not in (
            {"teacher", "weight", "data"}, {"teacher", "weight", "data", "input_pair"})
            or not isinstance(manifest, dict)
            or manifest.get("kind") != "frozen_parent_action_flow"):
        raise ValueError("unsupported or missing retention provenance")
    teacher = config["teacher"]
    if (not isinstance(teacher, dict)
            or set(teacher) != {"path", "sha256", "parent_receipt", "receipt_sha256"}
            or not all(isinstance(teacher[k], str) and teacher[k]
                       for k in ("path", "parent_receipt"))):
        raise ValueError("invalid retention teacher binding")
    weight = config["weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("invalid retention weight")
    expected = {"teacher_weights_sha256": _digest(teacher["sha256"]),
                "teacher_receipt_sha256": _digest(teacher["receipt_sha256"]),
                "weight": weight, "apply_to": "retained_training_items",
                "supervised_objective": "action_axis" if axis_objective else "author_uniform",
                "inference_teacher_required": False}
    if axis_objective:
        expected['supervised_axis_objective'] = axis_objective
    elif 'supervised_axis_objective' in manifest:
        raise ValueError('axis retention supervision is missing from the recipe')
    if (any(manifest.get(k) != v for k, v in expected.items())
            or type(manifest.get("weight")) not in (int, float)
            or manifest.get("inference_teacher_required") is not False
            or recipe.get("training_scope") != "action_and_proprio"
            or receipt.get("training_scope", {}).get("kind") != "action_and_proprio"):
        raise ValueError("retention checkpoint objective differs from recipe")
    for key in ("source_sha256", "data_source_sha256"):
        _digest(manifest.get(key))
    data = config["data"]
    if (not isinstance(data, dict) or set(data) != {"retained_recipe_sha256", "train", "validation"}
            or manifest.get("data") != data):
        raise ValueError("retention sampling provenance differs from recipe")
    for split in ("train", "validation"):
        item = data[split]
        if (not isinstance(item, dict) or set(item) != {"total_samples", "retained_prefix_samples",
                "retained_sampling_sha256", "complete_sampling_sha256"}
                or type(item["total_samples"]) is not int
                or type(item["retained_prefix_samples"]) is not int
                or not 0 < item["retained_prefix_samples"] < item["total_samples"]):
            raise ValueError("invalid retention sampling membership")
        _digest(item["retained_sampling_sha256"])
        _digest(item["complete_sampling_sha256"])
    binding = _bound_file(root, teacher["parent_receipt"], teacher["receipt_sha256"])
    path = Path(binding["path"])
    parent = json.loads(path.read_text())
    if (parent.get("weights_sha256") != teacher["sha256"]
            or parent.get("stats_sha256") != receipt["stats_sha256"]
            or Path(parent.get("weights_path", "")).resolve() != Path(teacher["path"]).resolve()):
        raise ValueError("retention teacher receipt, weights path or normalizer differs")
    pair = config.get('input_pair')
    recorded_pair = manifest.get('input_pair')
    if 'input_pair' in config:
        from .navigation_history import HistoryGridSpec, history_spec_from_contract

        if (not isinstance(pair, dict) or set(pair) != {'teacher_history', 'student_history'}
                or pair['teacher_history'] != HistoryGridSpec((300, 150, 50, 0)).as_dict()
                or not isinstance(pair['student_history'], dict)
                or pair['student_history'].get('schema') != 'microduck_rgb_target_keyframe_v2'
                or not isinstance(recorded_pair, dict)
                or set(recorded_pair) != {'schema', 'teacher_history', 'student_history', 'dataset_source_sha256'}
                or recorded_pair.get('schema') != 'microduck_memory_retention_pair_v1'
                or any(recorded_pair.get(k) != v for k, v in pair.items())
                or parent.get('input_history') != pair['teacher_history']
                or receipt.get('input_history') != pair['student_history']
                or receipt.get('extended_navigation_training', {}).get('schema') != 'microduck_memory_rehearsal_training_v1'):
            raise ValueError('retention paired inputs differ from recipe or trained histories')
        history_spec_from_contract(pair['student_history'])
        _digest(recorded_pair['dataset_source_sha256'])
    elif recorded_pair is not None or parent.get('input_history') != receipt.get('input_history'):
        raise ValueError('changed retention inputs require an explicit paired-input stage')
    teacher_recipe = _bound_file(root, path.parents[3] / "config.yaml", data["retained_recipe_sha256"])
    result = {"teacher_receipt": binding, "teacher_recipe": teacher_recipe,
            "teacher_weights_sha256": teacher["sha256"],
            "sampling_membership_recomputed": False, "weight_contents_verified": False}
    if recorded_pair is not None:
        result['input_pair'] = recorded_pair
    return result


def _resolve_retention_teachers(stages):
    for index, stage in enumerate(stages):
        retention = stage.get("retention_provenance")
        if retention is None:
            continue
        matches = [i for i in range(index + 1, len(stages))
                   if stages[i]["receipt"] == retention["teacher_receipt"]
                   and stages[i]["recipe"] == retention["teacher_recipe"]
                   and stages[i]["weights_sha256"] == retention["teacher_weights_sha256"]]
        if len(matches) != 1:
            raise ValueError("retention teacher requires an additional explicit ancestry resolver")
        retention.update(teacher_ancestor_stage=matches[0], additional_teacher_branch=False)


def _full_state_preparation(root, item, plan_binding, report_binding, plan, report):
    """Resolve the recorded full-state workflow; never infer a missing binding."""
    if (report.get("status") != "passed"
            or report.get("plan_sha256") != plan_binding["sha256"]):
        raise ValueError("full-state continuation lacks its hash-bound completion")
    files = plan.get("files", {})
    recipe_path = _contained(root, item["train_recipe"])
    if recipe_path != Path(plan["train_recipe"]).resolve():
        raise ValueError("full-state continuation recipe differs from its plan")
    recipe_binding = _bound_file(root, recipe_path, _digest(files.get(str(recipe_path))))
    recipe = yaml.safe_load(recipe_path.read_text())
    parent_path = _contained(root, Path(plan["checkpoint_state"]) / "navigation_checkpoint.json")
    parent_binding = _bound_file(root, parent_path, _digest(plan["checkpoint_receipt_sha256"]))
    parent = json.loads(parent_path.read_text())
    declared_recipe = _contained(root, plan["parent_recipe"])
    declared_binding = _bound_file(root, declared_recipe, _digest(files.get(str(declared_recipe))))
    parent_recipe = _bound_file(root, parent_path.parents[3] / "config.yaml", declared_binding["sha256"])
    clock_path = parent_path.parent / "trainer_state.json"
    clock_binding = _bound_file(root, clock_path, _digest(files.get(str(clock_path))))
    clock = json.loads(clock_path.read_text())
    step, target, batch = plan.get("source_step"), plan.get("target_step"), plan.get("source_batch_in_epoch")
    run = Path(plan["train_output"]).resolve()
    if (not run.is_relative_to(root) or run == parent_path.parents[3]
            or run == Path(plan["verify_output"]).resolve()
            or type(step) is not int or type(target) is not int or not 0 < step < target
            or type(batch) is not int or batch < 0
            or parent.get("step") != step
            or parent.get("weights_sha256") != _digest(plan["parent_weights_sha256"])
            or Path(parent["weights_path"]).resolve() != Path(plan["parent_weights"]).resolve()
            or set(clock) != {"global_step", "epoch", "batch_in_epoch"}
            or any(type(v) is not int or v < 0 for v in clock.values())
            or clock["global_step"] != step or clock["batch_in_epoch"] != batch
            or plan.get("training_steps") != target
            or plan.get("maximum_optimizer_steps") != target - step
            or report.get("source_step") != step or report.get("final_step") != target
            or report.get("new_optimizer_steps") != target - step
            or Path(recipe.get("output_dir", "")).resolve() != run
            or Path(recipe.get("resume", "")).resolve() != parent_path.parent
            or recipe.get("max_steps") != target
            or recipe.get("verify_resume_only", False)
            or recipe.get("initialize_weights") is not None
            or recipe.get("initialize_released_backbone") is not None):
        raise ValueError("full-state continuation parent, clock or training plan differs")
    head_binding = _bound_file(root, run / "checkpoints/state" / ("step_%06d" % target)
                              / "navigation_checkpoint.json", _digest(report["checkpoint_receipt_sha256"]))
    head = json.loads(Path(head_binding["path"]).read_text())
    if head.get("step") != target or head.get("weights_sha256") != _digest(report["weights_sha256"]):
        raise ValueError("full-state continuation completion differs from its checkpoint")
    preparation = {"kind": "verified_full_state_continuation", "plan": plan_binding,
        "report": report_binding, "training_recipe": recipe_binding, "parent_receipt": parent_binding,
        "parent_recipe": parent_recipe, "declared_parent_recipe": declared_binding,
        "parent_clock_file": clock_binding, "parent_clock": clock,
        "completed_checkpoint": head_binding, "recipe": recipe}
    selection = item.get('intermediate_checkpoint')
    if selection is not None:
        if not isinstance(selection, dict) or set(selection) != {
                'receipt', 'receipt_sha256', 'trainer_state', 'trainer_state_sha256'}:
            raise ValueError('intermediate checkpoint needs explicit receipt and clock bindings')
        selected_binding = _bound_file(root, selection['receipt'], _digest(selection['receipt_sha256']))
        selected_path = Path(selected_binding['path'])
        selected = json.loads(selected_path.read_text())
        selected_step = selected.get('step')
        cadence = recipe.get('save_every')
        if (type(selected_step) is not int or not step < selected_step < target
                or type(cadence) is not int or cadence <= 0 or selected_step % cadence
                or selected_path != run / 'checkpoints/state' / ('step_%06d' % selected_step)
                    / 'navigation_checkpoint.json'
                or Path(selected.get('state_path', '')).resolve() != selected_path.parent
                or Path(selected.get('weights_path', '')).resolve() != run / 'checkpoints/weights'
                    / ('step_%06d.pt' % selected_step)):
            raise ValueError('intermediate checkpoint step, cadence or path differs from the completed run')
        _digest(selected.get('weights_sha256'))
        clock_binding_selected = _bound_file(root, selection['trainer_state'],
                                             _digest(selection['trainer_state_sha256']))
        selected_clock = json.loads(Path(clock_binding_selected['path']).read_text())
        if (Path(clock_binding_selected['path']) != selected_path.parent / 'trainer_state.json'
                or set(selected_clock) != {'global_step', 'epoch', 'batch_in_epoch'}
                or any(type(v) is not int or v < 0 for v in selected_clock.values())
                or selected_clock['global_step'] != selected_step
                or (selected_clock['epoch'], selected_clock['batch_in_epoch'])
                    <= (clock['epoch'], clock['batch_in_epoch'])):
            raise ValueError('intermediate checkpoint clock differs from its step or source clock')
        moving = {'step', 'state_path', 'weights_path', 'weights_sha256', 'training_state_files'}
        if {k:v for k,v in selected.items() if k not in moving} != {k:v for k,v in head.items() if k not in moving}:
            raise ValueError('intermediate checkpoint training provenance differs from the completed run')
        preparation.update(selected_checkpoint=selected_binding, selected_clock=clock_binding_selected,
                           selected_step=selected_step, completed_step=target,
                           selection_kind='explicit_intermediate_checkpoint_from_completed_run')
    return run, preparation


def _resume_preparations(root, evidence):
    """Bind existing CPU preparation receipts without changing training recipes.

    New checkpoints declare their loss coefficients, so their recipe must not
    carry the legacy loss override. A separately pinned preparation can bind
    the parent recipe instead. These are metadata checks, not restore tests.
    """
    prepared = {}
    for item in evidence:
        plan_binding = _bound_file(root, item["plan"], _digest(item["plan_sha256"]))
        report_binding = _bound_file(root, item["report"], _digest(item["report_sha256"]))
        plan = json.loads(Path(plan_binding["path"]).read_text())
        report = json.loads(Path(report_binding["path"]).read_text())
        if plan.get("schema") == "microduck_verified_full_state_continuation_v1":
            run, preparation = _full_state_preparation(root, item, plan_binding, report_binding, plan, report)
            if run in prepared:
                raise ValueError("duplicate resume preparation")
            prepared[run] = preparation
            continue
        if item.get('intermediate_checkpoint') is not None:
            raise ValueError('intermediate checkpoints require a verified completed full-state continuation')
        if (plan.get("schema") != "microduck_cpu_continuation_preparation_v1"
                or report.get("status") != "complete"
                or report.get("plan_sha256") != plan_binding["sha256"]):
            raise ValueError("resume preparation lacks a completed, hash-bound plan")
        recipe_path = _contained(root, item["train_recipe"])
        recipe_binding = _bound_file(root, recipe_path,
            _digest(report.get("files", {}).get(str(recipe_path))))
        recipe = yaml.safe_load(recipe_path.read_text())
        parent_path = _contained(root, plan["parent_receipt"])
        parent_binding = _bound_file(root, parent_path, plan["parent_receipt_sha256"])
        parent = json.loads(parent_path.read_text())
        parent_recipe = _bound_file(root, plan["parent_recipe"], plan["parent_recipe_sha256"])
        run = Path(plan["train_output"]).resolve()
        step, target = parent.get("step"), plan.get("final_global_step")
        if (not run.is_relative_to(root) or run in prepared
                or run == Path(plan["resume_output"]).resolve()
                or run == parent_path.parents[3]
                or Path(plan["parent_state"]).resolve() != parent_path.parent
                or Path(parent_recipe["path"]) != parent_path.parents[3]/"config.yaml"
                or parent.get("weights_sha256") != _digest(plan["parent_weights_sha256"])
                or type(step) is not int or type(target) is not int or not 0 < step < target
                or plan.get("maximum_new_optimizer_steps") != target-step
                or plan.get("parent_clock", {}).get("global_step") != step
                or report.get("parent_clock") != plan.get("parent_clock")
                or report.get("maximum_new_optimizer_steps") != target-step
                or report.get("final_global_step") != target
                or Path(recipe.get("output_dir", "")).resolve() != run
                or Path(recipe.get("resume", "")).resolve() != parent_path.parent
                or recipe.get("max_steps") != target
                or recipe.get("verify_resume_only", False)
                or recipe.get("initialize_weights") is not None
                or recipe.get("initialize_released_backbone") is not None):
            raise ValueError("resume preparation parent, recipe or training output differs")
        prepared[run] = {"plan": plan_binding, "report": report_binding,
            "training_recipe": recipe_binding, "parent_receipt": parent_binding,
            "parent_recipe": parent_recipe, "recipe": recipe}
    return prepared


def _resume_parent(root, run, recipe, receipt, preparation=None):
    """Resolve an explicitly bound, separate-output complete-state continuation.

    Restore reports are provenance inputs here, not newly executed tensor/RNG
    checks. All returned file hashes remain inspectable by a later evaluator.
    """
    from .navigation_loss_binding import loss_weight_manifest, verify_resume_loss_weights

    if (recipe.get("initialize_weights") is not None
            or recipe.get("initialize_released_backbone") is not None
            or receipt.get("initialization") is not None
            or recipe.get("verify_resume_only", False)):
        raise ValueError("ambiguous or verification-only resume ancestry")
    if (Path(recipe.get("output_dir", "")).resolve() != run
            or type(recipe.get("max_steps")) is not int or recipe["max_steps"] < receipt["step"]):
        raise ValueError("resume output or final step differs from its recipe")
    parent_path = _contained(root, Path(recipe["resume"]) / "navigation_checkpoint.json")
    parent_binding = _bound_file(root, parent_path)
    parent = json.loads(parent_path.read_text())
    step = parent.get("step")
    if (type(step) is not int or not 0 < step < receipt["step"]
            or parent_path.parent.name != "step_%06d" % step
            or parent_path.parent.parent.name != "state"
            or parent_path.parent.parent.parent.name != "checkpoints"
            or parent_path.parents[3] == run):
        raise ValueError("unsupported resume parent layout, step or shared output")
    expected_recipe = recipe.get("resume_parent_recipe_sha256", recipe.get("resume_loss_recipe_sha256"))
    preparation_provenance = None
    if preparation is not None:
        if (recipe != preparation["recipe"]
                or parent_binding != preparation["parent_receipt"]
                or (expected_recipe is not None
                    and expected_recipe != preparation["parent_recipe"]["sha256"])):
            raise ValueError("resume differs from its pinned CPU preparation")
        expected_recipe = preparation["parent_recipe"]["sha256"]
        preparation_provenance = {k: v for k, v in preparation.items() if k != "recipe"}
    parent_recipe_binding = _bound_file(root, parent_path.parents[3]/"config.yaml", _digest(expected_recipe))
    parent_recipe = yaml.safe_load(Path(parent_recipe_binding["path"]).read_text())
    allowed = {"output_dir", "resume", "max_steps", "verify_resume_only", "initialize_weights",
               "resume_loss_recipe_sha256", "resume_parent_recipe_sha256"}
    full_state = preparation is not None and preparation.get("kind") == "verified_full_state_continuation"
    semantic_recipe = recipe
    semantic_parent_recipe = parent_recipe
    if full_state:
        source_clock = preparation["parent_clock"]
        expected_continuation = {"source_step": step, "source_batch_in_epoch": source_clock["batch_in_epoch"],
                                 "restores_optimizer_rng_and_sampler": True, "changes_model_or_data": False}
        if (recipe.get("audit", {}).get("continuation") != expected_continuation
                or ('continuation' in parent_recipe.get('audit', {}) and parent_recipe.get('resume') is None)
                or any(type(cfg.get(k)) is not int or cfg[k] <= 0
                       for cfg in (recipe, parent_recipe) for k in ("save_every", "eval_every"))
                or preparation.get("selected_checkpoint", preparation["completed_checkpoint"]) != _bound_file(root,
                    run / "checkpoints/state" / ("step_%06d" % receipt["step"]) / "navigation_checkpoint.json")):
            raise ValueError("full-state continuation metadata, cadence or final checkpoint differs")
        allowed.update({"save_every", "eval_every"})
        semantic_recipe = recipe | {"audit": {k:v for k,v in recipe["audit"].items() if k != "continuation"}}
        # A second continuation has a different source clock. Its parent's
        # clock is independently checked when the ancestry walker visits that
        # run; it must not be mistaken for a changed data/training objective.
        semantic_parent_recipe = parent_recipe | {
            'audit': {k:v for k,v in parent_recipe['audit'].items() if k != 'continuation'}}
    if ({k:v for k,v in semantic_recipe.items() if k not in allowed}
            != {k:v for k,v in semantic_parent_recipe.items() if k not in allowed}):
        raise ValueError("resume changed training semantics beyond the supported continuation")
    objective = loss_weight_manifest(recipe["model"].get("loss"))
    if receipt.get("loss_weights") != objective:
        raise ValueError("resumed checkpoint loss weights differ from recipe")
    loss_binding = verify_resume_loss_weights(parent, objective, state_dir=parent_path.parent,
        legacy_recipe_sha256=recipe.get("resume_loss_recipe_sha256"))
    proof_bindings = {name: _bound_file(root, run/name) for name in
                      ("report.json", "resume-verification.json", "restore-clock-verification.json")}
    report, restored, clock = [json.loads(Path(proof_bindings[name]["path"]).read_text()) for name in
                              ("report.json", "resume-verification.json", "restore-clock-verification.json")]
    completed_receipt = (json.loads(Path(preparation['completed_checkpoint']['path']).read_text())
                         if full_state else receipt)
    if (report.get("status") != "complete" or report.get("stage") != "complete"
            or report.get("optimizer_steps") != completed_receipt["step"]
            or report.get("last_checkpoint", {}).get("weights_sha256") != completed_receipt["weights_sha256"]
            or report.get("resume_verification") != restored
            or report.get("loss_weights") != objective
            or report.get("resume_loss_binding") != loss_binding):
        raise ValueError("resume training report does not bind this checkpoint and restore")
    if (restored.get("all_trainable_tensors_exact") is not True
            or restored.get("optimizer_moments_and_fp32_masters_exact") is not True
            or restored.get("optimizer_step") != step or restored.get("weights_updated") is not False):
        raise ValueError("resume lacks its complete model and optimizer restore report")
    clock_values = clock.get("clock", {}) if full_state else clock
    if (clock.get("status") != "verified" or clock.get("source_receipt_sha256") != parent_binding["sha256"]
            or (full_state and clock_values != preparation["parent_clock"])
            or clock_values.get("global_step") != step
            or (not full_state and clock.get("max_steps") != recipe["max_steps"])
            or clock.get("scheduler_exact") is not True or clock.get("random_states_exact") is not True
            or type(clock_values.get("batch_in_epoch")) is not int or clock_values["batch_in_epoch"] < 0
            or clock.get("sampler_resume_batch_offset") != clock_values["batch_in_epoch"]
            or clock.get("optimizer_steps_performed") != 0):
        raise ValueError("resume clock report does not bind the parent or complete state")
    return parent_path, {"kind": "complete_state_resume", "parent_receipt": parent_binding,
        "parent_recipe": parent_recipe_binding, "restore_reports": proof_bindings,
        "preparation": preparation_provenance,
        "loss_binding": loss_binding, "restore_contents_reverified": False}, {
        "weights_sha256": _digest(parent["weights_sha256"]), "parent_step": step,
        "stats_sha256": _digest(parent["stats_sha256"])}


def inventory_checkpoint_lineage(head_receipt, project_root, current_recipe, *, resume_preparations=()):
    """Follow explicit parent receipts, reject gaps/cycles and hash all indices.

    Complete-weight stage initialization and explicitly bound separate-output
    complete-state continuations are supported. Other/partial initialization
    is never silently treated as a foundation root. Each optional preparation
    supplies explicit plan/report paths and SHA256 hashes plus a train_recipe
    path; its completed report must bind that entire recipe. A selected
    intermediate checkpoint additionally needs pinned receipt and clock files;
    its parent run's final checkpoint remains separately identified. Unused or
    duplicate preparation evidence is rejected.
    """
    root = Path(project_root).resolve()
    preparations = _resume_preparations(root, resume_preparations)
    used_preparations = set()
    current_binding = _bound_file(root, current_recipe)
    current = yaml.safe_load(Path(current_binding["path"]).read_text())
    current_inputs = _recipe_inputs(root, current)
    receipt_path = _contained(root, head_receipt)
    seen, stages, all_inputs = set(), [], {}
    expected_parent = None
    while True:
        binding = _bound_file(root, receipt_path)
        receipt = json.loads(receipt_path.read_text())
        weight_hash = _digest(receipt["weights_sha256"])
        if weight_hash in seen:
            raise ValueError("cyclic checkpoint ancestry")
        seen.add(weight_hash)
        if (type(receipt.get("step")) is not int or receipt["step"] <= 0
                or receipt_path.parent.parent.name != "state"
                or receipt_path.parent.parent.parent.name != "checkpoints"):
            raise ValueError("unsupported checkpoint receipt layout or step")
        if expected_parent is not None:
            if (weight_hash != expected_parent["weights_sha256"]
                    or receipt["step"] != expected_parent["parent_step"]
                    or receipt["stats_sha256"] != expected_parent["stats_sha256"]):
                raise ValueError("parent receipt does not match child initialization")
        run = receipt_path.parents[3]
        recipe_binding = _bound_file(root, run / "config.yaml")
        recipe = yaml.safe_load(Path(recipe_binding["path"]).read_text())
        audit, data = recipe["audit"], recipe["data"]["train"]
        if (receipt["dataset_index_sha256"] != audit["index_sha256"]
                or receipt["stats_sha256"] != audit["stats_sha256"]):
            raise ValueError("checkpoint dataset or normalizer differs from recipe")
        for key in ("recovery_index_sha256", "augmentation_index_sha256", "retarget_index_sha256",
                    "qualified_teacher_index_sha256"):
            if (receipt.get(key) or []) != (audit.get(key) or []):
                raise ValueError("checkpoint provenance differs from recipe: " + key)
        if (receipt.get("history_prefix_index_sha256") or []) != (data.get("history_prefix_sha256") or []):
            raise ValueError("checkpoint history sources differ from recipe")
        _verify_extended_metadata(root, receipt, recipe)
        inputs = _recipe_inputs(root, recipe)
        all_inputs.update(inputs)
        initialization = receipt.get("initialization")
        init_recipe = recipe.get("initialize_weights")
        stage = {"weights_sha256": weight_hash, "step": receipt["step"],
                 "receipt": binding, "recipe": recipe_binding,
                 "input_index_sha256": sorted(inputs), "initialization": initialization}
        retention = _retention_teacher(root, receipt, recipe)
        if retention is not None:
            stage["retention_provenance"] = retention
        stages.append(stage)
        if recipe.get("resume") is not None:
            preparation = preparations.get(run)
            receipt_path, provenance, expected_parent = _resume_parent(root, run, recipe, receipt, preparation)
            if preparation is not None:
                used_preparations.add(run)
            stage["resume_provenance"] = provenance
            continue
        if initialization is None:
            if init_recipe is not None:
                raise ValueError("missing initialization receipt is not a foundation root")
            model = recipe["model"]
            stage["declared_foundation"] = {k: model.get(k) for k in
                ("_target_", "model_id", "tokenizer_model_id", "action_dit_pretrained_path",
                 "skip_dit_load_from_pretrain")}
            break
        if (not isinstance(initialization, dict)
                or initialization.get("kind") != "new_stage_from_complete_weights"
                or initialization.get("restored_modules") != ["mot", "proprio_encoder"]
                or not isinstance(init_recipe, dict)):
            raise ValueError("partial or unsupported checkpoint initialization")
        if init_recipe.get("sha256") != initialization.get("weights_sha256"):
            raise ValueError("recipe and initialization parent hashes differ")
        parent_path = _contained(root, init_recipe["parent_receipt"])
        parent = json.loads(parent_path.read_text())
        if Path(parent["weights_path"]).resolve() != Path(init_recipe["path"]).resolve():
            raise ValueError("initialized weights path differs from parent receipt")
        expected_parent = initialization
        receipt_path = parent_path
    if set(preparations) != used_preparations:
        raise ValueError("resume preparation does not belong to this checkpoint ancestry")
    _resolve_retention_teachers(stages)
    missing = sorted(set(all_inputs) - set(current_inputs))
    return {"schema": "microduck_navigation_declared_lineage_inventory_v1", "status": "complete",
            "head_weights_sha256": stages[0]["weights_sha256"], "stages": stages,
            "declared_chain_closed_at_root_recipe": True,
            "input_indices": [all_inputs[k] for k in sorted(all_inputs)],
            "current_recipe": current_binding,
            "current_input_indices": [current_inputs[k] for k in sorted(current_inputs)],
            "ancestral_indices_missing_from_current_recipe": missing,
            "all_declared_ancestral_inputs_present_in_current_recipe": not missing,
            "retention_teacher_count": sum("retention_provenance" in stage for stage in stages),
            "declared_retention_teacher_ancestry_verified": True,
            "weight_contents_verified": False, "foundation_initialization_verified": False,
            "all_selection_exposures_verified": False, "formal_acceptance": False,
            "hardware_validated": False}
