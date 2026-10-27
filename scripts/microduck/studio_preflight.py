"""Bounded architecture test on recorded sensors; never publishes a trained skill.

Run in the Studio worker image. The temporary checkpoint is diagnostic-only.
"""

import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch

from lightx2v_microduck.studio.dataset import grid, affine, historical_indices
from lightx2v_microduck.studio.schema import OBSERVATION, ACTION, write_json
from lightx2v_microduck.studio.fastwam import reference_model, ReferenceRuntime, FastWAMRuntime
from lightx2v_microduck.studio.models import package_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(42)
    with np.load(args.sensors) as source:
        d = {k: source[k].copy() for k in source.files}
    stats = {"schema": ACTION, "observation": OBSERVATION, "normalization_split": "train", "diagnostic_only": True}
    for key in ("state", "action"):
        a = d[key]
        stats[key] = {"default": {"global_min": a.min(0).tolist(), "global_max": a.max(0).tolist(), "global_mean": a.mean(0).tolist(), "global_std": a.std(0).tolist()}}
    write_json(out / "dataset_stats.json", stats)
    print("loading_176d_reference", flush=True)
    parent = args.data / "models/microduck"
    backbone = args.data / "models/Wan2.2-TI2V-5B"
    model = reference_model(backbone, parent / "model.pt", stats, json.loads((parent / "dataset_stats.json").read_text()))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-6)
    frame_indices = [historical_indices(d["tick"], d["tick"][min(i, len(d["tick"]) - 1)]) for i in range(0, 33, 4)]
    video = torch.from_numpy(np.stack([grid(d["rgb"][idx]) for idx in frame_indices])).permute(3, 0, 1, 2)[None].cuda().float() / 127.5 - 1
    ss, so = affine(stats, "state")
    sa, oa = affine(stats, "action")
    instruction = "Find the red ball and stop nearby."
    context, mask = model.encode_prompt("A video recorded from a robot's point of view executing the following instruction: " + instruction)
    sample = {
        "video": video,
        "proprio": torch.from_numpy(np.clip(d["state"][:1][None] * ss + so, -5, 5)).cuda(),
        "action": torch.from_numpy(np.clip(d["action"][np.minimum(np.arange(32), len(d["action"]) - 1)][None] * sa + oa, -5, 5)).cuda(),
        "action_is_pad": torch.zeros((1, 32), dtype=torch.bool, device="cuda"),
        "image_is_pad": torch.zeros((1, 9), dtype=torch.bool, device="cuda"),
        "context": context,
        "context_mask": mask,
    }
    print("full_forward_backward", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = model.training_loss(sample)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()
    report = {"diagnostic_only": True, "formal_acceptance": False, "loss": float(loss.detach()), "gradient_norm": float(grad), "state_dim": 176}
    print("save_reload_and_reference_inference", flush=True)
    model.eval()
    frames = d["rgb"][[0] * 4]
    states = d["state"][[0] * 4]
    ref = ReferenceRuntime(model, stats).predict(frames, states, 0, 0, instruction)
    torch.save({"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict()}, out / "model.pt")
    package_model(out, "fastwam", out / "dataset_stats.json", step=1, parent="diagnostic_176d_preflight")
    weights = torch.load(out / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.mot.load_state_dict(weights["mot"])
    model.proprio_encoder.load_state_dict(weights["proprio_encoder"])
    repeated = ReferenceRuntime(model, stats).predict(frames, states, 0, 0, instruction)
    report["reload_max_abs"] = float(np.abs(ref - repeated).max())
    np.testing.assert_allclose(ref, repeated, atol=1e-5, rtol=1e-5)
    write_json(out / "training-check.json", report)
    del model, optimizer, weights, context, mask, sample, video, loss, parts
    gc.collect()
    torch.cuda.empty_cache()
    print("native_lightx2v_176d_inference", flush=True)
    runtime = FastWAMRuntime(out, backbone, "plain")
    native = runtime.predict(frames, states, 0, 0, instruction)
    if native.shape != (32, 5) or not np.isfinite(native).all():
        raise ValueError("native invalid output")
    report["native_reference_max_abs"] = float(np.abs(ref - native).max())
    np.testing.assert_allclose(ref, native, atol=0.02, rtol=0.02)
    report["output_shape"] = list(native.shape)
    report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["gpu_seconds"] = time.monotonic() - started
    write_json(out / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
