"""Trainable small policy and strict model packaging. Torch is worker-only."""

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .dataset import affine
from .schema import OBSERVATION, ACTION, file_hash, write_json


class LightweightPolicy(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.state_encoder = nn.Sequential(nn.Linear(176, 128), nn.LayerNorm(128), nn.SiLU())
        self.category = nn.Embedding(3, 16)
        self.color = nn.Embedding(3, 16)
        self.gru = nn.GRU(672, 256, batch_first=True)
        self.head = nn.Linear(256, 32 * 5)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, rgb, state, category, color):
        b, t, h, w, c = rgb.shape
        image = rgb.reshape(b * t, h, w, c).permute(0, 3, 1, 2).float() / 255
        image = torch.nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        visual = self.encoder((image - self.mean) / self.std).reshape(b, t, 512)
        target = torch.cat((self.category(category), self.color(color)), dim=-1)[:, None, :].expand(-1, t, -1)
        x = torch.cat((visual, self.state_encoder(state), target), dim=-1)
        out, _ = self.gru(x)
        return self.head(out[:, -1]).reshape(b, 32, 5)


def lightweight_loss(prediction, action, valid):
    from torch.nn import functional as F

    mask = valid.float()
    continuous = F.smooth_l1_loss(prediction[..., :4], action[..., :4], reduction="none").mean(-1)
    stop = F.binary_cross_entropy_with_logits(prediction[..., 4], action[..., 4], reduction="none")
    return ((continuous + stop) * mask).sum() / mask.sum().clamp_min(1)


def expand_state_projection(old_weight, new_dim=176):
    if old_weight.ndim != 2 or old_weight.shape[1] != 48 or new_dim != 176:
        raise ValueError("expected a 48D parent and explicitly new 176D state contract")
    result = torch.zeros((old_weight.shape[0], new_dim), dtype=old_weight.dtype, device=old_weight.device)
    result[:, :48] = old_weight
    return result


def verify_model(root):
    root = Path(root).resolve()
    m = json.loads((root / "model.json").read_text())
    if m["observation"] != OBSERVATION or m["action"] != ACTION or m["horizon"] != 32:
        raise ValueError("incompatible model observation/action contract")
    for name, sha in m["files"].items():
        p = (root / name).resolve()
        if root not in p.parents or file_hash(p) != sha:
            raise ValueError(f"model file mismatch: {name}")
    s = json.loads((root / "dataset_stats.json").read_text())
    if s["normalization_split"] != "train" or s["observation"] != m["observation"] or len(s["state"]["default"]["global_mean"]) != 176:
        raise ValueError("model normalizer contract mismatch")
    return m, s


def package_model(root, backend, stats_path, *, parent=None, step=0):
    import shutil

    root = Path(root)
    if Path(stats_path).resolve() != (root / "dataset_stats.json").resolve():
        shutil.copy2(stats_path, root / "dataset_stats.json")
    files = {name: file_hash(root / name) for name in ("model.pt", "dataset_stats.json")}
    manifest = {
        "schema": "studio_model_v1",
        "backend": backend,
        "observation": OBSERVATION,
        "action": ACTION,
        "horizon": 32,
        "history_ticks": [300, 150, 50, 0],
        "hardware_validated": False,
        "status": "experimental",
        "files": files,
        "parent": parent,
        "step": step,
        "trained": step > 0,
    }
    write_json(root / "model.json", manifest)
    return manifest


class SmallRuntime:
    def __init__(self, path, device):
        self.manifest, self.stats = verify_model(path)
        if self.manifest["backend"] != "lightweight":
            raise ValueError("wrong backend")
        self.device = device
        self.model = LightweightPolicy(pretrained=False).to(device)
        self.model.load_state_dict(torch.load(Path(path) / "model.pt", map_location=device, weights_only=True), strict=True)
        self.model.eval()

    def predict(self, frames, states, category, color, instruction):
        scale, offset = affine(self.stats, "state")
        state = np.clip(states * scale + offset, -5, 5)
        with torch.inference_mode():
            output = self.model(
                torch.as_tensor(frames[None].copy(), device=self.device),
                torch.as_tensor(state[None], device=self.device),
                torch.tensor([category], device=self.device),
                torch.tensor([color], device=self.device),
            )[0].float()
            output[:, 4] = output[:, 4].sigmoid()
        scale, offset = affine(self.stats, "action")
        action = output.cpu().numpy()
        action[:, :4] = (action[:, :4] - offset[:4]) / scale[:4]
        return action
