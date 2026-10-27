"""One bounded supervised round. Resumption restores optimizer, sampler and RNG."""

import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .schema import digest, file_hash, write_json
from .dataset import SkillDataset, affine
from .models import LightweightPolicy, lightweight_loss, package_model


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(dataset, out, recipe, *, device, backbone=None, parent=None, resume=None, stop=lambda: False, progress=lambda x: None, validate=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    backend = recipe["backend"]
    ds = SkillDataset(dataset, "train", include_video=backend == "fastwam")
    SkillDataset(dataset, "validation")  # Require a separately audited validation split.
    binding = digest({"dataset": file_hash(Path(dataset) / "index.json"), "stats": file_hash(Path(dataset) / "dataset_stats.json"), "recipe": recipe})
    torch.manual_seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    random.seed(recipe["seed"])
    generator = torch.Generator().manual_seed(recipe["seed"])
    if backend == "fastwam":
        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("FastWAM training requires a CUDA worker")
        if not backbone or not parent:
            raise ValueError("FastWAM requires a verified MicroDuck parent and backbone")
        from .fastwam import reference_model

        model = reference_model(backbone, Path(parent) / "model.pt", ds.stats, json.loads((Path(parent) / "dataset_stats.json").read_text()))
    else:
        model = LightweightPolicy(pretrained=resume is None).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=recipe["learning_rate"], weight_decay=0.01)
    start_step = 0
    best_score = -1.0
    best_path = None
    validated_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["binding"] != binding:
            raise ValueError("resume data, normalization or recipe changed")
        if backend == "fastwam":
            model.mot.load_state_dict(checkpoint["model"]["mot"], strict=True)
            model.proprio_encoder.load_state_dict(checkpoint["model"]["proprio_encoder"], strict=True)
        else:
            model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["sampler_rng"])
        random.setstate(checkpoint["python_rng"])
        nr = checkpoint["numpy_rng"]
        np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
        if device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_step = checkpoint["step"]
        best_score = checkpoint["best_score"]
        best_path = checkpoint.get("best_path")
        validated_step = checkpoint.get("validated_step", 0)
    state_scale, state_offset = affine(ds.stats, "state")
    action_scale, action_offset = affine(ds.stats, "action")
    started = time.monotonic()
    step = start_step
    metrics = []
    contexts = {}

    def weights():
        return {"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict()} if backend == "fastwam" else model.state_dict()

    def save():
        save_checkpoint(
            out / "resume.pt",
            {
                "model": weights(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "binding": binding,
                "torch_rng": torch.get_rng_state(),
                "sampler_rng": generator.get_state(),
                "python_rng": random.getstate(),
                "numpy_rng": [np.random.get_state()[0], np.random.get_state()[1].tolist(), *np.random.get_state()[2:]],
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "best_score": best_score,
                "best_path": str(best_path) if best_path else None,
                "validated_step": validated_step,
            },
        )
        write_json(out / "resume.json", {"step": step, "binding": binding, "sha256": file_hash(out / "resume.pt")})

    def validate_checkpoint():
        nonlocal best_score, best_path, validated_step
        candidate = out / f"step-{step:06d}"
        candidate.mkdir(exist_ok=True)
        save_checkpoint(candidate / "model.pt", weights())
        package_model(
            candidate,
            backend,
            Path(dataset) / "dataset_stats.json",
            parent="lightx2v/microduck@dad4beaf998780df1f7c17aba46c63ecf2a4e72d" if backend == "fastwam" else "ResNet18_IMAGENET1K_V1",
            step=step,
        )
        save()
        if validate:
            model.eval()
            rng, nr, pr = torch.get_rng_state(), np.random.get_state(), random.getstate()
            cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else None
            try:
                result = validate(candidate, model, ds.stats)
                write_json(candidate / "validation.json", result)
                if result["rate"] > best_score:
                    best_score, best_path = result["rate"], candidate
            finally:
                torch.set_rng_state(rng)
                np.random.set_state(nr)
                random.setstate(pr)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                if backend == "fastwam":
                    from ..navigation_training_scope import apply_training_scope, scope_manifest

                    apply_training_scope(model, scope_manifest("action_and_proprio"))
                else:
                    model.train()
        else:
            best_path = candidate
        validated_step = step
        save()

    try:
        # A cancellation during validation must retry that validation, including
        # the last training step, rather than report a missing best candidate.
        if start_step > validated_step and (start_step % recipe["validate_every"] == 0 or start_step == recipe["max_steps"]):
            validate_checkpoint()
        for step in range(start_step + 1, recipe["max_steps"] + 1):
            if stop() or time.monotonic() - started > recipe["max_hours"] * 3600:
                step -= 1
                save()
                raise InterruptedError("training cancelled or per-run time limit reached; checkpoint saved")
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(recipe["batch_size"] // recipe["micro_batch"]):
                ids = torch.randint(len(ds), (recipe["micro_batch"],), generator=generator).tolist()
                batch = [ds[i] for i in ids]
                state = np.stack([b["state"] for b in batch])
                action = np.stack([b["action"] for b in batch])
                state = torch.as_tensor(np.clip(state * state_scale + state_offset, -5, 5), device=device)
                normalized = torch.as_tensor(np.clip(action * action_scale + action_offset, -5, 5), device=device)
                valid = torch.as_tensor(np.stack([b["valid"] for b in batch]), device=device)
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=recipe["precision"] == "bf16"):
                    if backend == "lightweight":
                        rgb = torch.as_tensor(np.stack([b["rgb"] for b in batch]), device=device)
                        pred = model(rgb, state, torch.tensor([b["category"] for b in batch], device=device), torch.tensor([b["color"] for b in batch], device=device))
                        normalized[..., 4] = torch.as_tensor(action[..., 4], device=device)
                        loss = lightweight_loss(pred, normalized, valid)
                    else:
                        encoded = []
                        for b in batch:
                            instruction = b["instruction"]
                            if instruction not in contexts:
                                with torch.no_grad():
                                    contexts[instruction] = model.encode_prompt("A video recorded from a robot's point of view executing the following instruction: " + instruction)
                            encoded.append(contexts[instruction])
                        video = torch.as_tensor(np.stack([b["video"] for b in batch]), device=device).permute(0, 4, 1, 2, 3).float() / 127.5 - 1
                        sample = {
                            "video": video,
                            "proprio": state[:, -1:],
                            "action": normalized,
                            "action_is_pad": ~valid,
                            "image_is_pad": ~torch.as_tensor(np.stack([b["video_valid"] for b in batch]), device=device),
                            "context": torch.cat([e[0] for e in encoded]),
                            "context_mask": torch.cat([e[1] for e in encoded]),
                        }
                        loss, _ = model.training_loss(sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                losses.append(float(loss.detach()))
                (loss / (recipe["batch_size"] // recipe["micro_batch"])).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            point = {
                "phase": "training",
                "step": step,
                "max_steps": recipe["max_steps"],
                "loss": float(np.mean(losses)),
                "elapsed_s": time.monotonic() - started,
                "gpu_seconds": time.monotonic() - started if device == "cuda" else 0,
            }
            metrics.append(point)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(point) + "\n")
            progress(point)
            if step % recipe["validate_every"] == 0 or step == recipe["max_steps"]:
                validate_checkpoint()
        return {"candidate": str(best_path), "step": step, "best_validation_rate": best_score, "binding": binding, "trained": True, "accepted": False}
    finally:
        optimizer = None
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
