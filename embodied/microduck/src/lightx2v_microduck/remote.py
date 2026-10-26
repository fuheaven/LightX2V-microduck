"""Transport-only TinyVLA client; does not import Torch or ROS."""

import base64
import io
import json
import urllib.request
import uuid

import numpy as np
from PIL import Image

from .contracts import JointResidualChunk, PolicyManifest
from .artifact_identity import policy_artifacts, policy_metadata


class TinyVLAHTTPPolicy:
    def __init__(self, endpoint, *, timeout=30.0, session_id=None):
        self.endpoint, self.timeout, self.session_id = endpoint.rstrip("/"), timeout, session_id or uuid.uuid4().hex
        health = self.call("/health")
        if health.get("engine") != "lightx2v-native-tinyvla" or health.get("raw_artifacts_verified") is not True:
            raise ValueError("endpoint must serve verified raw LightX2V TinyVLA artifacts")
        record = health["manifest"]
        self.artifact_manifest = record
        declared = record.get("inference", {})
        effective = health.get("runtime_inference")
        if not isinstance(effective, dict) or effective.get("schema_version") != 1:
            raise ValueError("native service must disclose its effective runtime inference settings")
        refresh = effective.get("semantic_refresh_steps")
        padding = effective.get("action_padding")
        if isinstance(refresh, bool) or not isinstance(refresh, int) or refresh < 1 or padding not in ("free", "training_noise_path"):
            raise ValueError("native service disclosed invalid runtime inference settings")
        if refresh != declared.get("semantic_refresh_steps", 50) or padding != declared.get("action_padding", "free"):
            raise ValueError("service runtime inference settings differ from its artifact; remove diagnostic overrides before deployment")
        correction = record.get("runtime_correction", {})
        if not isinstance(correction, dict):
            raise ValueError("invalid artifact runtime correction")
        semantic = declared.get("semantic_padding_policy", correction.get("policy", "legacy"))
        if (semantic not in ("legacy", "batch_consistent_v1")
                or correction.get("policy", semantic) != semantic
                or effective.get("semantic_padding_policy", "legacy") != semantic):
            raise ValueError("service semantic padding differs from its artifact")
        if health.get("precision") not in ("fp32", "bf16"):
            raise ValueError("native service must disclose its effective precision")
        self.runtime_inference = dict(effective)
        self.max_age_ticks = int(record.get("action_contract", {}).get("max_age_ticks", 10))
        if not 1 <= self.max_age_ticks <= 10:
            raise ValueError("unsupported action age limit")
        self.manifest = PolicyManifest(
            "lightx2v_tinyvla_http:" + record["model_variant"],
            "joint_residual",
            record["observation_profile"],
            record["control_profile"],
            embodiment=record["embodiment"],
            horizon=health["horizon"],
            execution_prefix=5,
            recurrent=True,
            artifacts=policy_artifacts(record),
            metadata={
                "transport": "lossless_png_http",
                "engine": health["engine"],
                "precision": health["precision"],
                "model_variant": record["model_variant"],
                "checkpoint_verified": True,
                "embodiment_digest": record["embodiment_digest"],
                "simulation_profile": record.get("simulation_profile"),
                "runtime_inference": self.runtime_inference,
                **policy_metadata(record),
            },
        )

    def call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.endpoint + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def reset(self, episode_id=""):
        self.call("/reset", {"session_id": self.session_id})

    def predict(self, observation):
        if observation.rgb is None:
            raise ValueError("TinyVLA needs an RGB frame")
        stream = io.BytesIO()
        Image.fromarray(observation.rgb).save(stream, format="PNG")
        result = self.call(
            "/predict",
            {
                "session_id": self.session_id,
                "episode_id": observation.episode_id,
                "control_index": observation.tick,
                "state61": observation.state.tolist(),
                "instruction": observation.instruction,
                "image_png_base64": base64.b64encode(stream.getvalue()).decode(),
            },
        )
        if result.get("episode_id") != observation.episode_id or result.get("control_index") != observation.tick or result.get("action_kind") != "joint_residual":
            raise ValueError("server returned a different observation or action contract")
        return JointResidualChunk(
            np.asarray(result["actions"], np.float32),
            observation.episode_id,
            observation.tick,
            observation.sim_time_s,
            valid_until_tick=observation.tick + self.max_age_ticks,
            diagnostics={k: v for k, v in result.items() if k != "actions"},
        )
