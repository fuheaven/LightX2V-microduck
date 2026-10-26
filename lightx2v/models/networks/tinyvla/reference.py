"""Explicit reference integration; never substitute native code for the author."""

from __future__ import annotations

import base64
import importlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .artifacts import load_raw_manifest, sha256_file, verified_file, verify_local_backbone
from .configuration_tinyvla import TinyVLAConfig
from .processing import MicroDuckNormalizer, TinyVLAPreprocessor
from .runtime import TinyVLARuntime

SOURCE_REVISION = "603f5045b45126af011418befc899e3c67d04346"
SOURCE_HASHES = {
    "__init__.py": "5d8181421d7c277e8d272f1069884c276a308d3c70218eadc0bfe9657842aadb",
    "configuration_tinyvla.py": "5579d03281f172c1c6b7e7765781d8b2c2c089b7f914b273c21e494b78863819",
    "modeling_tinyvla.py": "2239946592cc097a1e4bcd362ab4c51912df71dc914b20268c7e183c2dfe3ea9",
    "processor_tinyvla.py": "4b87f14192428ea6cca96d3e368536a838c4e2b9ad0fe0d2698fb8588ebf212e",
    "modules/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "modules/semantic.py": "7751e9ac8739d87508683a51b557354d571405f203dbffb9d791310b0455ccc8",
    "modules/expert.py": "b5a10ba5626e4e8650dec965b83369454d943e704eeaab71f31931e88b40d25d",
    "modules/embodiment.py": "f37afc5096ab1346ccb62debb81fd9e6b77eb5d9c548b4c7bc6cc27e2bd2b15f",
}


def verify_author_source(source_root):
    root = Path(source_root).resolve()
    package = root / "tinyvla"
    for relative, expected in SOURCE_HASHES.items():
        path = package / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"author reference source missing/modified: {relative}; expected {SOURCE_REVISION}")
    return root


class AuthorModelAdapter(torch.nn.Module):
    """Call the unchanged author's model with isolated RNG state."""

    def __init__(self, author):
        super().__init__()
        self.author = author
        self.config = author.config

    @property
    def semantic(self):
        return self.author.semantic

    def _semantic_latent(self, batch):
        return self.author._semantic_latent(batch)

    def predict_action_chunk(self, batch, *, noise=None, generator=None):
        if noise is not None:
            raise ValueError("unchanged author sampler does not expose explicit noise; use identical seeded session RNG")
        device = next(self.author.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if generator is not None:
                if device.type == "cuda":
                    torch.cuda.set_rng_state(generator.get_state(), device)
                else:
                    torch.set_rng_state(generator.get_state())
            result = self.author.predict_action_chunk(batch)
            if generator is not None:
                generator.set_state(torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state())
        return result


def load_author_model(source_root, checkpoint, *, backbone_path, device="cpu", config_only_init=False):
    """Load the real author class; optional initialization-only weight shortcut.

    config_only_init skips downloading Qwen base weights that the complete VLA
    checkpoint immediately replaces. The original forward and sampler methods
    are unchanged. This is useful for normalized tensor diagnostics and does
    not satisfy missing raw normalization/support requirements.
    """
    root = verify_author_source(source_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        configuration = importlib.import_module("tinyvla.configuration_tinyvla")
        modeling = importlib.import_module("tinyvla.modeling_tinyvla")
    except ImportError as exc:
        raise RuntimeError("author reference needs its separate Python 3.12 / LeRobot 0.6.1 / Transformers 5.5.4 environment; native backend is not a fallback") from exc
    if Path(modeling.__file__).resolve() != root / "tinyvla/modeling_tinyvla.py":
        raise RuntimeError("another tinyvla package is already imported; run reference in its own process")
    # Filter only integration metadata absent from the author's dataclass;
    # retain every architecture field and construct its own LeRobot types.
    from dataclasses import fields

    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature

    payload = json.loads((Path(checkpoint) / "config.json").read_text())
    names = {field.name for field in fields(configuration.TinyVLAConfig)}
    payload = {key: value for key, value in payload.items() if key in names}
    for key in ("input_features", "output_features"):
        payload[key] = {name: PolicyFeature(FeatureType(value["type"]), tuple(value["shape"])) for name, value in payload[key].items()}
    if "normalization_mapping" in payload:
        payload["normalization_mapping"] = {key: NormalizationMode(value) for key, value in payload["normalization_mapping"].items()}
    payload.update(lm_model_name=str(backbone_path), device=device)
    config = configuration.TinyVLAConfig(**payload)
    if config_only_init:
        from unittest.mock import patch

        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForImageTextToText

        def initialize_backbone(model_name, **kwargs):
            backbone = AutoConfig.from_pretrained(model_name)
            backbone.text_config.num_hidden_layers = config.lm_num_layers
            backbone.text_config.layer_types = backbone.text_config.layer_types[: config.lm_num_layers]
            return AutoModelForImageTextToText.from_config(backbone, dtype=kwargs.get("dtype", torch.float32))

        with patch.object(AutoModelForImageTextToText, "from_pretrained", side_effect=initialize_backbone):
            author = modeling.TinyVLAPolicy(config)
        # Strictly replace ALL parameters: no random initialization survives.
        author.load_state_dict(load_file(str(Path(checkpoint) / "model.safetensors")), strict=True, assign=True)
        author.to(device)
    else:
        # This calls the ORIGINAL constructor, including pretrained Qwen weight
        # loading. A config-only backbone cache is insufficient for this mode.
        author = modeling.TinyVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
    return author


def load_author_runtime(source_root, checkpoint, *, backbone_path, device="cpu", precision="fp32", config_only_init=False):
    manifest = load_raw_manifest(checkpoint)
    verify_local_backbone(backbone_path, manifest["backbone_revision"])
    author = load_author_model(source_root, checkpoint, backbone_path=backbone_path, device=device, config_only_init=config_only_init)
    processor_config = TinyVLAConfig.from_pretrained(checkpoint)
    processor_config.lm_model_name = str(backbone_path)
    processor_config.backbone_revision = manifest["backbone_revision"]
    support = None
    if manifest["support"]["mode"] == "required":
        with np.load(verified_file(checkpoint, manifest["support"]), allow_pickle=False) as archive:
            support = {key: archive[key].copy() for key in ("images", "actions")}
    normalizer = MicroDuckNormalizer.from_file(verified_file(checkpoint, manifest["normalization"]))
    processor = TinyVLAPreprocessor.from_config(processor_config, normalization=normalizer, support=support)
    runtime = TinyVLARuntime(AuthorModelAdapter(author), processor, manifest=manifest, device=device, precision=precision)
    runtime.checkpoint_verified = True
    return runtime


class TinyVLAReferencePolicyHTTP:
    """Lossless-image adapter for a verified, separately running author service."""

    def __init__(self, endpoint, *, timeout=30.0, session_id="reference", headers=None):
        from lightx2v_microduck.contracts import PolicyManifest

        self.endpoint, self.timeout, self.session_id = endpoint.rstrip("/"), timeout, session_id
        self.headers = dict(headers or {})
        health = self._request("/health")
        if health.get("engine") != "tinyvla-author-reference" or health.get("source_revision") != SOURCE_REVISION:
            raise ValueError("reference endpoint must identify the pinned original author implementation")
        if health.get("raw_artifacts_verified") is not True or health.get("horizon") != 50:
            raise ValueError("reference service does not have verified raw 50-step artifacts")
        self.artifact_manifest = record = health["manifest"]
        self.manifest = PolicyManifest(
            policy_id="tinyvla_reference:" + record["model_variant"],
            action_kind="joint_residual",
            observation_profile=record["observation_profile"],
            control_profile=record.get("control_profile", "raw_residual"),
            embodiment=record.get("embodiment", "legs"),
            horizon=50,
            execution_prefix=5,
            recurrent=True,
            artifacts={"checkpoint_sha256": record["checkpoint"]["sha256"], "source_revision": SOURCE_REVISION},
            metadata={"transport": "http_png_lossless", "precision": health["precision"]},
        )

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        headers = {"Content-Type": "application/json", **self.headers}
        request = urllib.request.Request(self.endpoint + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"author reference endpoint failed: {exc}") from exc

    def reset(self, episode_id=None):
        self._request("/reset", {"session_id": self.session_id})

    def predict(self, observation):
        from lightx2v_microduck.contracts import JointResidualChunk

        stream = io.BytesIO()
        Image.fromarray(observation.rgb).save(stream, format="PNG")
        result = self._request(
            "/predict",
            {
                "session_id": self.session_id,
                "episode_id": observation.episode_id,
                "control_index": observation.tick,
                "instruction": observation.instruction,
                "state61": observation.state.tolist(),
                "image_png_base64": base64.b64encode(stream.getvalue()).decode(),
            },
        )
        if result.get("action_kind") != "joint_residual" or result.get("episode_id") != observation.episode_id or result.get("control_index") != observation.tick:
            raise ValueError("reference response has stale identity or incompatible actions")
        return JointResidualChunk(
            result["actions"],
            observation.episode_id,
            observation.tick,
            source_time_s=observation.sim_time_s,
            valid_until_tick=observation.tick + self.artifact_manifest["action_contract"].get("max_age_ticks", 10),
            diagnostics={key: value for key, value in result.items() if key != "actions"},
        )
