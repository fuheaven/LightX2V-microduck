"""Pinned FastWAM reference behind the same sensor-only navigation interface.

The reference environment imports FastWAM, never the native LightX2V backend.
An untrained preflight checkpoint can validate numerics, not navigation ability.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from .navigation import NAVIGATION_CONTRACT
from .navigation_action_contracts import verify_artifact_action_contract
from .wam_data import PROMPT, reference_processor


class ReferenceNavigationCore:
    policy_profile = "microduck_navigation"
    action_dim = 5
    robot_state_dim = 48
    camera_size = 256
    gripper_postprocess = False

    def __init__(self, model, stats, *, compile_action_infer=False, cache_prompt=False,
                 seed=0, inference_steps=20, horizon=32, history_spec=None,
                 navigation_contract=NAVIGATION_CONTRACT, normalization_binding=None,
                 stats_sha256=None):
        if (model.proprio_dim, model.action_expert.action_dim) != (48, 5):
            raise ValueError("reference model is not a MicroDuck navigation model")
        self.navigation_contract = verify_artifact_action_contract(
            dict(contract=navigation_contract, embodiment='legs',
                 normalization_binding=normalization_binding, files={'dataset_stats.json': stats_sha256}),
            dict(navigation_contract=navigation_contract, normalization_binding=normalization_binding), stats)
        self.model = model.eval().requires_grad_(False)
        self.processor = reference_processor(stats, horizon)
        self.compile_action_infer = bool(compile_action_infer)
        self.cache_prompt = bool(cache_prompt)
        self.seed, self.inference_steps, self.horizon = seed, int(inference_steps), int(horizon)
        self._prompt_cache = None
        self.history_contract = None if history_spec is None else history_spec.as_dict()
        self.camera_size = 256 if history_spec is None else history_spec.output_size

    @classmethod
    def from_checkpoint(cls, model_config, checkpoint, stats_path, *, device="cuda", **kwargs):
        stats_bytes = Path(stats_path).read_bytes()
        stats, digest = json.loads(stats_bytes), hashlib.sha256(stats_bytes).hexdigest()
        if kwargs.get('stats_sha256', digest) != digest:
            raise ValueError('reference normalization digest differs from its actual file')
        kwargs['stats_sha256'] = digest
        contract = kwargs.get('navigation_contract', NAVIGATION_CONTRACT)
        binding = kwargs.get('normalization_binding')
        verify_artifact_action_contract(
            dict(contract=contract, embodiment='legs', normalization_binding=binding,
                 files={'dataset_stats.json': digest}),
            dict(navigation_contract=contract, normalization_binding=binding), stats)
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config = OmegaConf.load(model_config)
        if (int(config.proprio_dim), int(config.action_dit_config.action_dim)) != (48, 5):
            raise ValueError("reference config has an incompatible embodiment")
        config.skip_dit_load_from_pretrain = True
        config.load_text_encoder = True
        config.action_dit_pretrained_path = None
        model = instantiate(config, model_dtype=torch.bfloat16, device=device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if "mot" not in payload or "proprio_encoder" not in payload:
            raise ValueError("reference checkpoint requires both experts and proprioception weights")
        # The upstream convenience loader uses strict=False for legacy weights.
        # A navigation artifact is fully specified and must match every tensor.
        model.mot.load_state_dict(payload["mot"], strict=True)
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        return cls(model, stats, **kwargs)

    def reset(self):
        self._prompt_cache = None

    def predict_action_chunk(self, images, state, task_description, seed=None):
        import torch

        rgb = np.asarray(images["head_camera"])
        if rgb.shape != (self.camera_size, self.camera_size, 3) or rgb.dtype != np.uint8:
            raise ValueError("navigation reference expects uint8 RGB matching its explicit input contract")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (48,) or not np.isfinite(state).all():
            raise ValueError("navigation reference expects finite proprio48")
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1)
        proprio = self.processor.normalizer.normalizers["state"]["default"].forward(torch.from_numpy(state))
        prompt = PROMPT.format(task=task_description)
        with torch.inference_mode():
            extra = {}
            if self.cache_prompt:
                if self._prompt_cache is None or self._prompt_cache[0] != prompt:
                    context, mask = self.model.encode_prompt(prompt)
                    self._prompt_cache = prompt, context, mask
                extra = {"context": self._prompt_cache[1], "context_mask": self._prompt_cache[2]}
                prompt = None
            result = self.model.infer_action(
                prompt=prompt, input_image=image, proprio=proprio, action_horizon=self.horizon,
                num_inference_steps=self.inference_steps, sigma_shift=1.,
                seed=self.seed if seed is None else seed, rand_device="cpu",
                compile_action_infer=self.compile_action_infer, **extra)
            action = self.processor.normalizer.normalizers["action"]["default"].backward(result["action"])
        return action.cpu().numpy().astype(np.float32)

    def close(self):
        self.reset()
