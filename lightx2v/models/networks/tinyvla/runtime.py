"""Native TinyVLA sessions and raw MicroDuck action chunks."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import load_raw_manifest, verified_file, verify_local_backbone
from .configuration_tinyvla import TinyVLAConfig
from .modeling_tinyvla import TinyVLAModel
from .processing import MicroDuckNormalizer, TinyVLAPreprocessor


@dataclass
class SessionState:
    episode_id: str
    generator: torch.Generator
    instruction: str | None = None
    latent: torch.Tensor | None = None
    refresh_tick: int = -1
    last_tick: int = -1


class TinyVLARuntime:
    """The simulator owns execution. This runtime never sleeps or steps physics."""

    def __init__(self, model, preprocessor, *, device="cpu", precision="fp32", semantic_refresh_steps=None, max_sessions=8, seed=42, manifest=None, compile_expert=False, vision_backend=None, action_padding=None):
        if semantic_refresh_steps is None:
            # Historical artifacts retain their original 1 Hz behavior.
            semantic_refresh_steps = (manifest or {}).get("inference", {}).get("semantic_refresh_steps", 50)
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16; INT8 needs separate parity validation")
        action_padding = action_padding or (manifest or {}).get("inference", {}).get("action_padding", "free")
        if action_padding not in ("free", "training_noise_path"):
            raise ValueError("unsupported action padding trajectory")
        self.action_padding = action_padding
        if max_sessions < 1 or isinstance(semantic_refresh_steps, bool) or not isinstance(semantic_refresh_steps, int) or semantic_refresh_steps < 1:
            raise ValueError("session capacity and refresh steps must be positive")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        if vision_backend is not None:
            if vision_backend not in ("hf", "fixed_eager", "fixed_compile"):
                raise ValueError("unsupported vision backend")
            self.model.semantic.vision_backend = vision_backend
        self.preprocessor = preprocessor
        self.preprocessor.device = self.device
        self.precision, self.refresh_steps = precision, semantic_refresh_steps
        self.max_sessions, self.seed, self.manifest = max_sessions, int(seed), manifest
        # Supplying a dict to the low-level constructor is not artifact proof.
        # Only a loader that verifies hashes and strict-loads tensors sets this.
        self.checkpoint_verified = False
        self.sessions = OrderedDict()
        self.lock = threading.RLock()
        if compile_expert:
            # Keep state_dict naming unchanged: compile the call, not the module.
            self.model.expert.forward = torch.compile(self.model.expert.forward, dynamic=False)

    @classmethod
    def from_directory(cls, directory, *, backbone_path=None, **kwargs):
        manifest = load_raw_manifest(directory)
        configured_backbone = backbone_path or TinyVLAConfig.from_pretrained(directory).lm_model_name
        if Path(configured_backbone).is_dir():
            verify_local_backbone(configured_backbone, manifest["backbone_revision"])
        model = TinyVLAModel.from_pretrained(directory, backbone_path=backbone_path, backbone_revision=manifest["backbone_revision"])
        normalizer = MicroDuckNormalizer.from_file(verified_file(directory, manifest["normalization"]))
        support = None
        if manifest["support"]["mode"] == "required":
            with np.load(verified_file(directory, manifest["support"]), allow_pickle=False) as archive:
                support = {key: archive[key].copy() for key in ("images", "actions")}
        processor = TinyVLAPreprocessor.from_config(model.config, normalization=normalizer, support=support)
        runtime = cls(model, processor, manifest=manifest, **kwargs)
        runtime.checkpoint_verified = True
        return runtime

    def reset(self, session_id=None):
        with self.lock:
            if session_id is None:
                self.sessions.clear()
            else:
                self.sessions.pop(session_id, None)

    def _session(self, session_id, episode_id):
        if not session_id or not episode_id:
            raise ValueError("session_id and episode_id must be nonempty")
        session = self.sessions.pop(session_id, None)
        if session is None or session.episode_id != episode_id:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            session = SessionState(episode_id, generator)
        self.sessions[session_id] = session
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
        return session

    @torch.inference_mode()
    def predict(self, image_rgb, state61, instruction, *, session_id, episode_id, control_index, normalized=False, noise=None):
        if not isinstance(control_index, int) or isinstance(control_index, bool) or control_index < 0:
            raise ValueError("control_index must be a nonnegative integer")
        if not normalized and self.manifest is None:
            raise ValueError("raw predictions require a verified artifact manifest")
        # Qwen's internal positional caches are shared with the model, so all
        # model calls are serialized even when clients have separate sessions.
        with self.lock:
            session = self._session(session_id, episode_id)
            if control_index <= session.last_tick:
                raise ValueError("control ticks must increase within an episode; reset or use a new episode_id")
            batch = self.preprocessor.prepare(image_rgb, state61, instruction, normalized=normalized)
            refresh = session.latent is None or session.instruction != instruction or control_index - session.refresh_tick >= self.refresh_steps
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
                if refresh:
                    session.latent = self.model._semantic_latent(batch).detach()
                    session.instruction, session.refresh_tick = instruction, control_index
                batch["semantic_latent"] = session.latent
                if noise is not None:
                    noise = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
                    if noise.ndim == 2:
                        noise = noise[None]
                padding_kwargs = {"active_action_dim": 14} if self.action_padding == "training_noise_path" else {}
                result = self.model.predict_action_chunk(batch, noise=noise, generator=session.generator, **padding_kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            actions = result[0, :, :14].float().cpu().numpy()
            if not normalized:
                actions = self.preprocessor.normalization.denormalize_action(actions)
            if not np.isfinite(actions).all():
                self.sessions.pop(session_id, None)
                raise ValueError("model produced nonfinite actions")
            session.last_tick = control_index
            return {
                "actions": actions,
                "episode_id": episode_id,
                "control_index": control_index,
                "action_kind": "normalized_action14" if normalized else "joint_residual",
                "execute_steps": 5,
                "max_age_ticks": (self.manifest or {}).get("action_contract", {}).get("max_age_ticks", 10),
                "control_hz": 50,
                "semantic_refreshed": refresh,
                "latency_ms": elapsed_ms,
                "precision": self.precision,
                "vision_backend": getattr(getattr(self.model, "semantic", None), "vision_backend", "fixture"),
                "semantic_refresh_steps": self.refresh_steps,
                "action_padding": self.action_padding,
            }


class MicroDuckTinyVLAPolicy:
    """Adapter for the standalone environment observation object."""

    def __init__(self, runtime, *, session_id="microduck"):
        if runtime.manifest is None:
            raise ValueError("an environment policy requires verified raw artifacts")
        self.runtime, self.session_id = runtime, session_id
        from lightx2v_microduck.contracts import PolicyManifest
        from lightx2v_microduck.artifact_identity import policy_artifacts, policy_metadata

        record = runtime.manifest
        self.manifest = PolicyManifest(
            policy_id="tinyvla:" + record["model_variant"],
            action_kind="joint_residual",
            observation_profile=record["observation_profile"],
            control_profile=record.get("control_profile", "raw_residual"),
            embodiment=record.get("embodiment", "legs"),
            horizon=runtime.model.config.chunk_size,
            execution_prefix=5,
            recurrent=True,
            artifacts=policy_artifacts(record),
            metadata={
                "precision": runtime.precision,
                "embodiment_digest": record["embodiment_digest"],
                "max_age_ticks": record["action_contract"].get("max_age_ticks", 10),
                "engine": "lightx2v-native-tinyvla",
                "model_variant": record["model_variant"],
                "checkpoint_verified": runtime.checkpoint_verified,
                "simulation_profile": record.get("simulation_profile"),
                "initial_conditions": record.get("initial_conditions"),
                **policy_metadata(record),
            },
        )

    def reset(self, episode_id=None):
        self.runtime.reset(self.session_id)

    def predict(self, observation, *, noise=None):
        from lightx2v_microduck.contracts import JointResidualChunk

        result = self.runtime.predict(
            observation.rgb, observation.state, observation.instruction, session_id=self.session_id, episode_id=str(observation.episode_id), control_index=int(observation.tick), noise=noise
        )
        diagnostics = {key: value for key, value in result.items() if key != "actions"}
        return JointResidualChunk(
            result["actions"],
            observation.episode_id,
            observation.tick,
            source_time_s=observation.sim_time_s,
            valid_until_tick=observation.tick + self.runtime.manifest["action_contract"].get("max_age_ticks", 10),
            diagnostics=diagnostics,
        )
