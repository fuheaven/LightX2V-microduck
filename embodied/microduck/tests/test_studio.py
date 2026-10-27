"""Workspace, provenance, leakage, cancellation and model portability regressions."""

import io
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from lightx2v_microduck.studio.schema import SkillDraft, OBSERVATION, ACTION, file_hash, write_json
from lightx2v_microduck.studio.store import Store
from lightx2v_microduck.studio.scene import validate_scene, sample_scene
from lightx2v_microduck.studio.simulation import cases, instruction
from lightx2v_microduck.studio.evidence import summarize, compare

P = {"source_sha256": "verified-source", "worker_image": "image@sha256:fixed"}
W = {"source_sha256": P["source_sha256"], "image": P["worker_image"]}


@pytest.fixture
def store(tmp_path):
    instance = Store(tmp_path)
    yield instance
    instance.close()


def skill(store):
    return store.save_skill(SkillDraft().model_dump())


def test_reject_sqlite_runtime_with_known_unix_deadlock(tmp_path, monkeypatch):
    from lightx2v_microduck.studio import store as store_module

    monkeypatch.setattr(store_module.sqlite3, "sqlite_version_info", (3, 51, 1))
    with pytest.raises(RuntimeError, match="locking deadlock"):
        Store(tmp_path)


def test_queue_idempotency_and_fenced_concurrent_claim(store):
    s = skill(store)
    a = store.submit(s["id"], {"kind": "preview", "idempotency_key": "once"}, P)
    assert store.submit(s["id"], {"kind": "preview", "idempotency_key": "once"}, P)["id"] == a["id"]
    with pytest.raises(ValueError):
        store.submit(s["id"], {"kind": "preview", "idempotency_key": "once", "seed": 1}, P)
    with ThreadPoolExecutor(2) as pool:
        leases = list(pool.map(lambda name: store.claim(name, ["cpu"], W), ["first", "second"]))
    lease = next(x for x in leases if x)
    assert sum(x is not None for x in leases) == 1
    assert store.claim(lease["worker"] or "first", ["cpu"], W) is None
    with pytest.raises(ValueError):
        store.update(a["id"], "stolen", status="complete")
    store.cancel(a["id"])
    assert store.update(a["id"], lease["lease"], progress={"tick": 5})["cancel"]
    store.update(a["id"], lease["lease"], status="cancelled")
    with pytest.raises(ValueError):
        store.update(a["id"], lease["lease"], status="complete")


def test_expired_lease_does_not_restart_job(store):
    s = skill(store)
    a = store.submit(s["id"], {"kind": "preview"}, P)
    job = store.claim("cpu", ["cpu"], W)
    with store.connect() as db:
        db.execute("UPDATE runs SET heartbeat=? WHERE id=?", (time.time() - 100, a["id"]))
    assert store.claim("another", ["cpu"], W) is None
    assert store.run(a["id"])["status"] == "interrupted"
    with pytest.raises(ValueError):
        store.update(a["id"], job["lease"], status="complete")


def test_source_image_and_gpu_are_required(store):
    s = skill(store)
    a = store.submit(s["id"], {"kind": "pipeline"}, P)
    assert store.claim("cpu", ["cpu"], W) is None
    assert store.claim("gpu", ["gpu"], {**W, "image": "floating"}) is None
    assert store.claim("gpu", ["gpu"], {**W, "source_sha256": "dirty"}) is None
    assert store.claim("gpu", ["gpu"], W)["id"] == a["id"]


def test_teacher_gate_cannot_be_bypassed_by_process_exit(store):
    s = skill(store)
    r = store.submit(s["id"], {"kind": "teacher"}, P)
    j = store.claim("cpu", ["cpu"], W)
    store.update(r["id"], j["lease"], status="complete", result={"teacher_qualified": False})
    with pytest.raises(ValueError, match="teacher gate"):
        store.submit(s["id"], {"kind": "collect", "teacher_run": r["id"]}, P)


def test_scene_validation_and_sampling_are_real_changes():
    scene = SkillDraft().model_dump()["scene"]
    first = sample_scene(scene, 1)
    second = sample_scene(scene, 2)
    assert first == sample_scene(scene, 1) and first != second
    assert first["objects"][0]["x"] != second["objects"][0]["x"]
    scene["objects"][0]["x"] = 0
    scene["objects"][0]["y"] = 0
    with pytest.raises(ValueError, match="重叠"):
        validate_scene(scene)


def test_split_templates_and_geometry_do_not_leak():
    s = SkillDraft().model_dump()
    sets = [cases(s, split, 9) for split in ("train", "validation", "test")]
    assert len({c["layout_id"] for group in sets for c in group}) == 27
    assert not {instruction(c) for c in sets[0]} & {instruction(c) for c in sets[1] + sets[2]}
    assert not {c["instance_id"] for c in sets[0]} & {c["instance_id"] for c in sets[1] + sets[2]}


def episodes(n=200):
    return [{"success": True, "reason": "success", "group": "clear", "category": "ball", "color": "red", "layout_id": str(i), "split": "test", "rtf": 1} for i in range(n)]


def test_acceptance_needs_independent_sufficient_realtime_evidence():
    rows = episodes()
    assert summarize(rows, model_sha="fixed", case_sha="cases")["accepted"]
    assert not summarize(rows[:199], model_sha="fixed", case_sha="cases")["accepted"]
    rows[-1]["layout_id"] = "0"
    assert not summarize(rows, model_sha="fixed", case_sha="cases")["accepted"]
    rows = episodes()
    rows[0]["rtf"] = 0.2
    assert not summarize(rows, mode="realtime", model_sha="fixed", case_sha="cases")["accepted"]
    assert summarize(episodes(100), teacher=True, groups=["clear"])["teacher_qualified"]
    assert not summarize(episodes(100), teacher=True, groups=["clear", "obstacle"])["teacher_qualified"]


def test_comparison_rejects_mismatched_cases_and_unknown_cost():
    a = {**summarize(episodes(), model_sha="a", case_sha="one"), "p95_ms": 200, "execution": "ttl500", "resources": {"gpu_seconds": 3600}}
    b = {**a, "cases_sha256": "two", "p95_ms": 100}
    result = compare([a, b])
    assert not result["comparable"] and result["recommended_model"] is None
    assert all(r["cost_per_success"] is None for r in result["rows"])


def fixture_data(root):
    """Small synthetic fixture for tensor/optimizer tests; not a capability dataset."""
    from lightx2v_microduck.studio.dataset import audit_dataset

    root.mkdir()
    rows = []
    for i, split in enumerate(("train", "validation", "test")):
        path = root / (split + ".npz")
        n = 8
        state = np.zeros((n, 176), np.float32)
        state[:, 48:112] = 0.5 + i
        state[:, 112:] = 1
        action = np.zeros((n, 5), np.float32)
        action[:, 0] = 0.1
        action[4:, 4] = 1
        np.savez_compressed(path, state=state, action=action, rgb=np.full((n, 256, 256, 3), 60 + i, np.uint8), tick=np.arange(n) * 5, tof_tick=np.arange(n) * 5)
        rows.append(
            {
                "id": split,
                "path": path.name,
                "sha256": file_hash(path),
                "frames": n,
                "split": split,
                "layout_id": split,
                "instance_id": split,
                "instruction_family": split,
                "eligible": True,
                "category": "ball",
                "color": "red",
                "group": "clear",
                "instruction": "diagnostic fixture",
            }
        )
    write_json(root / "index.json", {"observation": OBSERVATION, "action": ACTION, "episodes": rows})
    audit_dataset(root)
    return root


def test_data_audit_uses_train_only_and_rejects_leakage(tmp_path):
    from lightx2v_microduck.studio.dataset import audit_dataset, SkillDataset

    data = fixture_data(tmp_path / "data")
    stats = json.loads((data / "dataset_stats.json").read_text())
    assert stats["state"]["default"]["global_max"][48] == 0.5
    item = SkillDataset(data)[0]
    assert item["state"].shape == (4, 176) and item["action"].shape == (32, 5)
    assert item["valid"].sum() == 8 and "video" not in item
    index = json.loads((data / "index.json").read_text())
    index["episodes"][1]["layout_id"] = "train"
    write_json(data / "index.json", index)
    with pytest.raises(ValueError, match="leakage"):
        audit_dataset(data)


def test_complete_model_backward_and_portable_bundle(tmp_path):
    import torch
    from lightx2v_microduck.studio.models import LightweightPolicy, lightweight_loss, package_model, SmallRuntime, verify_model
    from lightx2v_microduck.studio.packages import export_bundle, import_bundle

    torch.set_num_threads(1)
    torch.manual_seed(42)
    data = fixture_data(tmp_path / "data")
    model = LightweightPolicy(pretrained=False)
    rgb = torch.zeros((1, 4, 256, 256, 3), dtype=torch.uint8)
    state = torch.zeros((1, 4, 176))
    labels = torch.zeros((1, 32, 5))
    valid = torch.ones((1, 32), dtype=torch.bool)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = lightweight_loss(model(rgb, state, torch.tensor([0]), torch.tensor([0])), labels, valid)
    loss.backward()
    opt.step()
    assert np.isfinite(float(loss.detach()))
    store = Store(tmp_path / "ws")
    s = skill(store)
    out = store.root / "model"
    out.mkdir()
    torch.save(model.state_dict(), out / "model.pt")
    package_model(out, "lightweight", data / "dataset_stats.json", step=1)
    runtime = SmallRuntime(out, "cpu")
    prediction = runtime.predict(rgb[0].numpy(), state[0].numpy(), 0, 0, "fixture")
    assert prediction.shape == (32, 5) and np.isfinite(prediction).all()
    v = store.add_version(
        s["id"],
        {
            "name": "fixture",
            "backend": "lightweight",
            "model_path": "model",
            "manifest_sha256": file_hash(out / "model.json"),
            "status": "experimental",
            "report": {},
            "skill_snapshot": s,
            "task_contract": s["task"],
        },
    )
    bundle = export_bundle(store, v["id"])
    other = Store(tmp_path / "other")
    imported = import_bundle(other, bundle)
    restored = SmallRuntime(other.root / imported["model_path"], "cpu")
    np.testing.assert_allclose(restored.predict(rgb[0].numpy(), state[0].numpy(), 0, 0, "fixture"), prediction, atol=1e-6)
    assert imported["status"] == "experimental"
    (out / "model.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="mismatch"):
        verify_model(out)


def test_unsafe_package_rejected_before_extraction(store, tmp_path):
    from lightx2v_microduck.studio.packages import import_bundle

    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as t:
        member = tarfile.TarInfo("../../outside")
        member.size = 1
        t.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        import_bundle(store, archive)
    assert not (tmp_path / "outside").exists()


def test_api_truth_boundary_and_cross_origin(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from lightx2v_microduck.studio.api import create_app

    monkeypatch.setenv("STUDIO_WORKER_TOKEN", "worker-test")
    app = create_app(tmp_path)
    client = TestClient(app)
    r = client.post("/api/skills", json=SkillDraft().model_dump())
    assert r.status_code == 200
    s = r.json()
    r = client.post(f"/api/skills/{s['id']}/runs", json={"kind": "preview"})
    assert r.status_code == 202
    run = r.json()
    tool = client.get(f"/api/runs/{run['id']}?tool=true").json()
    assert "snapshot" not in tool and "result" not in tool and "success" not in str(tool)
    assert client.get(f"/api/files/runs/{run['id']}/snapshot.json").status_code == 404
    assert client.post("/api/skills", json=SkillDraft().model_dump(), headers={"Origin": "https://unrelated.example"}).status_code == 403
    assert client.post("/api/worker/claim", json={}).status_code == 401


def test_login_cookie_is_required_when_configured(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from lightx2v_microduck.studio.api import create_app

    monkeypatch.setenv("STUDIO_TOKEN", "test-only-secret")
    client = TestClient(create_app(tmp_path))
    assert client.get("/").status_code != 401  # Login shell must load before authentication.
    assert client.get("/api/bootstrap").status_code == 401
    assert client.post("/api/session", json={"token": "bad"}).status_code == 401
    assert client.post("/api/session", json={"token": "test-only-secret"}).status_code == 200
    assert client.get("/api/bootstrap").status_code == 200


def test_geometry_identity_does_not_count_appearance_as_new_layout():
    from copy import deepcopy
    from lightx2v_microduck.studio.scene import layout_identity

    a = SkillDraft().model_dump()["scene"]
    b = deepcopy(a)
    b["objects"][0].update(color="blue", kind="box", width=0.1)
    assert layout_identity(a) == layout_identity(b)
    b["objects"][0]["x"] += 0.1
    assert layout_identity(a) != layout_identity(b)


def test_qualified_teacher_cannot_be_reused_after_task_change(store):
    s = skill(store)
    r = store.submit(s["id"], {"kind": "teacher"}, P)
    j = store.claim("cpu", ["cpu"], W)
    store.update(r["id"], j["lease"], status="complete", result={"teacher_qualified": True})
    s["task"]["timeout_s"] += 1
    store.save_skill({k: v for k, v in s.items() if k != "id"}, s["id"])
    with pytest.raises(ValueError, match="different frozen"):
        store.submit(s["id"], {"kind": "collect", "teacher_run": r["id"]}, P)


def test_training_resume_preserves_optimizer_and_final_validation(tmp_path, monkeypatch):
    import torch
    from lightx2v_microduck.studio import training
    from lightx2v_microduck.studio.models import LightweightPolicy
    from lightx2v_microduck.studio.schema import Recipe

    torch.set_num_threads(1)
    # Full architecture, synthetic tensor fixture; no pretrained download or capability claim.
    monkeypatch.setattr(training, "LightweightPolicy", lambda pretrained: LightweightPolicy(pretrained=False))
    data = fixture_data(tmp_path / "data")
    recipe = Recipe(batch_size=1, micro_batch=1, precision="fp32", max_steps=2, validate_every=1).model_dump()

    def validate(path, model, stats):
        return {"rate": int(path.name.split("-")[-1]) / 2, "diagnostic_only": True}

    full = training.train(data, tmp_path / "full", recipe, device="cpu", validate=validate)

    def interrupted(path, model, stats):
        if path.name == "step-000002":
            raise InterruptedError("cancelled during final validation")
        return validate(path, model, stats)

    with pytest.raises(InterruptedError):
        training.train(data, tmp_path / "partial", recipe, device="cpu", validate=interrupted)
    checkpoint = tmp_path / "partial/resume.pt"
    restored = training.train(data, tmp_path / "restored", recipe, device="cpu", resume=checkpoint, validate=validate)
    assert restored["step"] == 2 and restored["best_validation_rate"] == 1
    a = torch.load(full["candidate"] + "/model.pt", weights_only=True)
    b = torch.load(restored["candidate"] + "/model.pt", weights_only=True)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="recipe changed"):
        training.train(data, tmp_path / "bad-resume", {**recipe, "learning_rate": 1e-3}, device="cpu", resume=checkpoint)


def test_acceptance_registration_checks_model_cases_and_preserves_old_version(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from lightx2v_microduck.studio.api import create_app
    from lightx2v_microduck.studio.schema import digest

    monkeypatch.delenv("STUDIO_TOKEN", raising=False)
    app = create_app(tmp_path)
    store = app.state.store
    client = TestClient(app)
    spec = SkillDraft().model_dump()
    spec["task"]["groups"] = ["clear"]
    s = store.save_skill(spec)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.pt").write_bytes(b"non-executable API checksum fixture")
    write_json(model / "model.json", {"files": {"model.pt": file_hash(model / "model.pt")}})
    v = store.add_version(
        s["id"],
        {
            "name": "fixture",
            "backend": "lightweight",
            "model_path": "model",
            "manifest_sha256": file_hash(model / "model.json"),
            "status": "experimental",
            "task_contract": s["task"],
            "skill_snapshot": s,
        },
    )
    r = store.submit(s["id"], {"kind": "evaluate", "policy": "lightweight", "version": v["id"]}, P)
    job = store.claim("cpu", ["cpu"], W)
    rows = episodes()
    write_json(tmp_path / "runs" / r["id"] / "evaluation-cases.json", rows)
    report = {**summarize(rows, model_sha=file_hash(model / "model.pt"), case_sha=digest(rows)), "task_sha256": digest(s["task"])}
    store.update(r["id"], job["lease"], status="complete", result=report)
    result = client.post(f"/api/versions/{v['id']}/qualify", json={"run_id": r["id"]})
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "accepted_offline" and store.version(v["id"])["status"] == "experimental"
    (model / "model.pt").write_bytes(b"modified")
    assert client.post(f"/api/versions/{v['id']}/qualify", json={"run_id": r["id"]}).status_code == 409
    assert client.get(f"/api/runs/{r['id']}/metrics").json() == []
