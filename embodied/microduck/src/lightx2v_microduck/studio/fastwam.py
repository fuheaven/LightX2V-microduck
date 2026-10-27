"""Public FastWAM architecture training and native LightX2V deployment adapters."""

import json
from pathlib import Path
import numpy as np
import torch
from .schema import OBSERVATION
from .dataset import affine, grid


def reference_model(backbone, checkpoint, stats, parent_stats=None):
    from fastwam.models.wan22.fastwam import FastWAM
    from .models import expand_state_projection

    root = Path(__file__).resolve().parents[5]
    cfg = json.loads((root / "examples/microduck/studio/fastwam-model.json").read_text())
    model = FastWAM.from_wan22_pretrained(
        model_id=str(backbone),
        tokenizer_model_id=str(backbone),
        device="cuda",
        torch_dtype=torch.bfloat16,
        tokenizer_max_len=128,
        load_text_encoder=True,
        proprio_dim=176,
        redirect_common_files=False,
        skip_dit_load_from_pretrain=True,
        **cfg,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.mot.load_state_dict(payload["mot"], strict=True)
    projection = payload["proprio_encoder"]
    if projection["weight"].shape[1] == 48:
        if parent_stats is None:
            raise ValueError("48D transfer requires the parent normalizer")
        old_scale, old_offset = affine(parent_stats, "state")
        scale, offset = affine(stats, "state")
        old_weight = projection["weight"]
        w = expand_state_projection(old_weight)
        # Preserve the old state mapping under newly measured training normalization.
        ratio = torch.as_tensor(old_scale / scale[:48], dtype=old_weight.dtype)
        w[:, :48] = old_weight * ratio
        bias = projection["bias"] + old_weight @ torch.as_tensor(old_offset - old_scale / scale[:48] * offset[:48], dtype=old_weight.dtype)
        projection = {"weight": w, "bias": bias}
    elif projection["weight"].shape[1] != 176:
        raise ValueError("unsupported state projection")
    model.proprio_encoder.load_state_dict(projection, strict=True)
    from ..navigation_training_scope import apply_training_scope, scope_manifest

    apply_training_scope(model, scope_manifest("action_and_proprio"))
    return model


class FastWAMRuntime:
    def __init__(self, path, backbone, variant="optimized"):
        import os

        os.environ.setdefault("LIGHTX2V_IMPORT_MODE", "robotics")
        os.environ.setdefault("DTYPE", "BF16")
        import lightx2v_platform.set_ai_device  # noqa: F401
        from .models import verify_model
        from ..wam_policy import native_navigation_config
        from ..navigation_history import HistoryGridSpec
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy

        self.manifest, self.stats = verify_model(path)
        if self.manifest["backend"] != "fastwam":
            raise ValueError("wrong backend")
        cfg = native_navigation_config(backbone, Path(path) / "model.pt", Path(path) / "dataset_stats.json", history_spec=HistoryGridSpec((300, 150, 50, 0)))
        cfg.update(
            robot_state_dim=176,
            studio_observation=OBSERVATION,
            seed=42,
            fastwam_cache_prompt=variant == "optimized",
            fastwam_cache_action_context=variant == "optimized",
            fastwam_action_cuda_graph=variant == "optimized",
            checkpoint_mmap=True,
        )
        self.core = FastWAMPolicy.from_config(cfg)

    def reset(self):
        self.core.reset()

    def predict(self, frames, states, category, color, instruction):
        return self.core.predict_action_chunk(images={"head_camera": grid(frames)}, state=states[-1], task_description=instruction)


class ReferenceRuntime:
    """Validation shares training normalization without reloading the 5B model."""

    def __init__(self, model, stats):
        self.model, self.stats = model, stats

    def predict(self, frames, states, category, color, instruction):
        scale, offset = affine(self.stats, "state")
        image = torch.from_numpy(grid(frames).copy()).permute(2, 0, 1).float() / 127.5 - 1
        proprio = torch.from_numpy(np.clip(states[-1] * scale + offset, -5, 5))
        with torch.inference_mode():
            result = self.model.infer_action(
                prompt="A video recorded from a robot's point of view executing the following instruction: " + instruction,
                input_image=image,
                proprio=proprio,
                action_horizon=32,
                num_inference_steps=20,
                sigma_shift=1.0,
                seed=42,
                rand_device="cpu",
            )
        scale, offset = affine(self.stats, "action")
        return (result["action"].float().cpu().numpy() - offset) / scale
