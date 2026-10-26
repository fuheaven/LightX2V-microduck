"""Deployment must retain action/observation identity and fail on foreign files."""
import importlib.util
import json
from pathlib import Path
import sys
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


def test_model_download_requires_immutable_revision_before_network(tmp_path, monkeypatch):
    m = module("fastwam_handoff")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"published_bundle": {"repo": "lightx2v/microduck", "revision": "main"}, "files": {}}))
    monkeypatch.setattr(m, "LOCK", lock)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **kwargs: pytest.fail("Mutable revision reached network")))
    with pytest.raises(ValueError, match="pinned model release"):
        m.fetch(SimpleNamespace(component="model", data=tmp_path / "data"))


@pytest.mark.parametrize("component", ["model", "all"])
def test_model_download_preserves_layout_and_rejects_corrupt_remote_file(tmp_path, monkeypatch, component):
    m = module("fastwam_handoff")
    from lightx2v_microduck.navigation_history import history_spec
    payloads = {"models/microduck/model.pt": b"weights", "models/microduck/dataset_stats.json": b"{}"}
    import hashlib
    receipt = {"input_history": history_spec((300, 150, 50, 0)).as_dict(), "step": 250,
               "weights_sha256": hashlib.sha256(payloads["models/microduck/model.pt"]).hexdigest(),
               "stats_sha256": hashlib.sha256(payloads["models/microduck/dataset_stats.json"]).hexdigest()}
    payloads["models/microduck/navigation_checkpoint.json"] = json.dumps(receipt).encode()
    specs = {p:{"bytes":len(data), "sha256":hashlib.sha256(data).hexdigest(),
               "distribution":"published_microduck"} for p,data in payloads.items()}
    backbone = "models/Wan2.2-TI2V-5B/config.json"
    specs[backbone] = {"bytes":2, "sha256":hashlib.sha256(b"{}").hexdigest(), "distribution":"public_backbone"}
    if component == "all":
        payloads[backbone] = b"{}"
    lock = tmp_path / "lock.json"
    revision = "a" * 40
    lock.write_text(json.dumps({"published_bundle":{"repo":"lightx2v/microduck", "revision":revision}, "files":specs}))
    monkeypatch.setattr(m, "LOCK", lock)
    corrupt = False
    fetched_sources = []

    def fetch_source(source, root):
        assert root == tmp_path / "data/upstream"
        fetched_sources.append(source)
        return {"ok": True}

    monkeypatch.setitem(sys.modules, "lightx2v_microduck.assets", SimpleNamespace(fetch_source=fetch_source))

    def download(**kwargs):
        assert kwargs["repo_id"] == "lightx2v/microduck"
        assert kwargs["revision"] == revision
        assert set(kwargs["allow_patterns"]) == set(payloads)
        for relative, data in payloads.items():
            p = kwargs["local_dir"] / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"corrupt" if corrupt and relative.endswith("model.pt") else data)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    args = SimpleNamespace(component=component, data=tmp_path / "data")
    m.fetch(args)
    assert (args.data / backbone).exists() == (component == "all")
    assert fetched_sources == (["microduck_rl", "microduck-policies"] if component == "all" else [])
    corrupt = True
    with pytest.raises(ValueError, match="differs"):
        m.fetch(args)
