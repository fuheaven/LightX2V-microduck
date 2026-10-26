"""Deployment must retain action/observation identity and fail on foreign files."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts/microduck" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_incompatible_or_missing_artifact_is_not_silently_replaced(tmp_path, monkeypatch):
    m = module("fastwam_handoff")
    data = tmp_path / "data"
    (data / "models/microduck").mkdir(parents=True)
    files = {"model.pt": b"weights", "dataset_stats.json": b"{}"}
    from lightx2v_microduck.navigation_history import history_spec
    for name, contents in files.items():
        (data / "models/microduck" / name).write_bytes(contents)
    receipt = {"input_history": history_spec((300, 150, 50, 0)).as_dict(), "step": 250,
               "weights_sha256": m.sha256(data / "models/microduck/model.pt"),
               "stats_sha256": m.sha256(data / "models/microduck/dataset_stats.json")}
    (data / "models/microduck/navigation_checkpoint.json").write_text(json.dumps(receipt))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"files": {str(p.relative_to(data)): {
        "bytes": p.stat().st_size, "sha256": m.sha256(p), "distribution": "obtain_from_project_owner"}
        for p in (data / "models/microduck").iterdir()}}))
    monkeypatch.setattr(m, "LOCK", lock)
    assert len(m.verify(data)) == 3
    (data / "models/microduck/model.pt").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="differs"):
        m.verify(data)
    (data / "models/microduck/model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing"):
        m.verify(data)


@pytest.mark.parametrize("variant,count", [("plain", 0), ("prompt", 1), ("context", 2), ("optimized", 3)])
def test_eval_variant_keeps_observation_history_ttl_and_output_contract(tmp_path, variant, count):
    m = module("fastwam_handoff")
    args = SimpleNamespace(data=tmp_path, output=tmp_path / "out", mode="realtime", scene="room_clear",
        tasks=["find_red_ball"], seeds=[2000], variant=variant, max_action_age_ms=500, max_ticks=1500)
    command = m.eval_command(args)
    assert command[command.index("--history-offsets-ticks")+1:command.index("--history-offsets-ticks")+5] == ["300", "150", "50", "0"]
    assert command[command.index("--max-action-age-ms")+1] == "500"
    assert command[command.index("--inference-steps")+1] == "20"
    assert sum(flag in command for flag in ("--cache-prompt", "--cache-action-context", "--action-cuda-graph")) == count
    assert "--record-model-inputs" in command and "--record-predictions" in command


def test_video_holds_last_measured_frame_across_camera_gaps():
    m = module("fastwam_recording_video")
    np.testing.assert_array_equal(m.frame_indices([0, 5, 20], 24), [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError, match="increasing"):
        m.frame_indices([0, 5, 5], 24)
