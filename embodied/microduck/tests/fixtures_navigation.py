import hashlib
import json

import numpy as np
import pytest

from lightx2v_microduck.wam_data import FastWAMNavigationDataset, build_index, split_for_seed


def episode(root, seed, *, success=True, value=.1):
    path = root / f"find_red_ball-seed{seed}"
    path.mkdir()
    frames = np.stack([np.full((256, 256, 3), i, np.uint8) for i in range(10)])
    np.savez_compressed(path / "navigation.npz", rgb=frames,
                        proprio=np.full((10, 48), value, np.float32),
                        action=np.tile([value, 0, 0, 0, 0], (9, 1)).astype(np.float32), ticks=np.arange(10) * 5)
    np.savez(path / "control_trace.npz", raw_action=np.zeros((45, 14), np.float32))
    meta = {"schema": "microduck_navigation_v1", "task": "find_red_ball", "seed": seed,
            "split": split_for_seed(seed), "eligible_training_demonstration": success,
            "search_visibility_gated": True,
            "instruction": "Find the red ball", "result": {"success": success},
            "files": {f: hashlib.sha256((path / f).read_bytes()).hexdigest()
                      for f in ("navigation.npz", "control_trace.npz")}}
    (path / "episode.json").write_text(json.dumps(meta))
    return path
