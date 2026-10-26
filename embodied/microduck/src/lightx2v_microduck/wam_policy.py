"""Sensor-only FastWAM navigation inference; no environment or teacher access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .navigation import DECISION_TICKS, NAVIGATION_CONTRACT, NavigationChunk, student_state
from .navigation_action_contracts import action_contract, core_action_contract, verify_artifact_action_contract


def artifact_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_navigation_config(backbone, checkpoint, stats, *, history_spec=None,
                             contract=NAVIGATION_CONTRACT, normalization_binding=None):
    """A complete native config derived from the verified Wan architecture."""
    backbone, checkpoint, stats = (Path(value).resolve() for value in (backbone, checkpoint, stats))
    architecture = json.loads((backbone / "config.json").read_text())
    if tuple(architecture.get(key) for key in ("dim", "num_heads", "num_layers", "in_dim")) != (3072, 24, 30, 48):
        raise ValueError("navigation v1 requires the pinned Wan2.2 TI2V 5B architecture")
    config = {key: architecture[key] for key in ("dim", "ffn_dim", "num_heads", "num_layers", "in_dim", "out_dim", "freq_dim", "eps")}
    config.update(model_path=str(backbone), adapter_model_path=str(checkpoint), dataset_stats_path=str(stats),
                  action_chunk_size=32, actions_per_plan=1, action_infer_steps=20, action_sample_shift=1.,
                  action_dim_hidden=1024, action_dim=5, robot_state_dim=48, camera_size=256,
                  policy_profile="microduck_navigation", gripper_postprocess=False, binarize_gripper=False,
                  vae_std_reciprocal_fp32=True,
                  vae_rms_norm_mode="fastwam_reference",
                  normalize_mode="min/max", seq_parallel=False, cpu_offload=False,
                  rms_norm_type="torch", layer_norm_type="torch", fastwam_rope_type="torch_complex_rope",
                  default_prompt="A video recorded from a robot's point of view executing the following instruction: {task_prompt}")
    if action_contract(contract).name != NAVIGATION_CONTRACT:
        config['navigation_contract'] = contract
    if normalization_binding is not None:
        config['normalization_binding'] = dict(normalization_binding)
    if contract != NAVIGATION_CONTRACT or normalization_binding is not None:
        verify_artifact_action_contract(
            dict(contract=contract, embodiment='legs', normalization_binding=normalization_binding,
                 files={'dataset_stats.json': artifact_sha256(stats)}),
            config, json.loads(stats.read_text()))
    if history_spec is not None:
        config.update(camera_size=history_spec.output_size, navigation_history=history_spec.as_dict())
    return config


class FastWAMNavigationPolicy:
    def __init__(self, core, *, model_id, artifact_sha256, validity_ticks=25, contract=None,
                 record_model_inputs=False):
        if validity_ticks is not None and (type(validity_ticks) is not int or not 1 <= validity_ticks <= 256 * DECISION_TICKS):
            raise ValueError("navigation validity must be an integer within the maximum chunk horizon")
        if (core.policy_profile, core.action_dim, core.robot_state_dim, core.camera_size) != ("microduck_navigation", 5, 48, 256):
            raise ValueError("FastWAM core does not implement the MicroDuck navigation contract")
        if core.gripper_postprocess:
            raise ValueError("gripper postprocessing would corrupt navigation decisions")
        self.core, self.validity_ticks = core, validity_ticks
        self.contract = core_action_contract(core, contract)
        self.metadata = {"model_id": model_id, "artifact_sha256": artifact_sha256,
                         "contract": self.contract, "hardware_validated": False,
                         "input": "head_rgb+instruction+proprio48", "uses_privileged_state": False}
        self.episode_id = ""
        from .navigation_model_inputs import configure_recording
        configure_recording(self, record_model_inputs)

    @classmethod
    def from_artifact(cls, path, *, device="cuda"):
        """A manipulator checkpoint cannot silently become a duck policy."""
        path = Path(path).resolve()
        manifest = json.loads((path / "navigation_model.json").read_text())
        action_contract(manifest["contract"])
        for filename in ("model.pt", "dataset_stats.json", "lightx2v_config.json"):
            if artifact_sha256(path / filename) != manifest["files"][filename]:
                raise ValueError(f"navigation artifact checksum mismatch: {filename}")
        config = json.loads((path / "lightx2v_config.json").read_text())
        contract = verify_artifact_action_contract(manifest, config,
            json.loads((path / "dataset_stats.json").read_text()))
        config.update(adapter_model_path=str(path / "model.pt"), dataset_stats_path=str(path / "dataset_stats.json"), device=device)
        import lightx2v_platform.set_ai_device  # noqa: F401
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy

        return cls(FastWAMPolicy.from_config(config), model_id=manifest["model_id"],
                   artifact_sha256=manifest["files"]["model.pt"], contract=contract)

    def reset(self, episode_id):
        self.episode_id = str(episode_id)
        self.core.reset()

    def predict(self, observation):
        if observation.episode_id != self.episode_id:
            raise ValueError("reset the navigation policy for this episode first")
        if observation.rgb is None or observation.camera_tick != observation.tick:
            raise ValueError("FastWAM navigation requires a fresh head-camera frame")
        if not observation.instruction:
            raise ValueError("FastWAM navigation requires a task instruction")
        state = student_state(observation)
        receipt = None
        if self.record_model_inputs:
            from .navigation_model_inputs import argument_receipt
            receipt = argument_receipt(observation, state)
        values = np.asarray(self.core.predict_action_chunk(
            images={"head_camera": observation.rgb}, state=state,
            task_description=observation.instruction), dtype=np.float32)
        if receipt is not None and argument_receipt(observation, state) != receipt:
            raise ValueError('core mutated its recorded RGB, state or language inputs')
        if values.ndim != 2 or values.shape[1] != 5 or not 1 <= len(values) <= 256 or not np.isfinite(values).all():
            raise ValueError("FastWAM returned an invalid navigation prediction")
        low, high = action_contract(self.contract).bounds()
        bounded = np.clip(values, low, high)
        # None removes the additional age cap, never the predicted action horizon.
        horizon = len(values) * DECISION_TICKS
        validity = horizon if self.validity_ticks is None else min(self.validity_ticks, horizon)
        return NavigationChunk(bounded, observation.episode_id, observation.tick,
                               observation.monotonic_ns,
                               observation.tick + validity,
                               diagnostics={"clipped_values": int(np.count_nonzero(bounded != values)), **self.metadata,
                                            **({"model_core_inputs": receipt} if receipt is not None else {})},
                               contract=self.contract)
