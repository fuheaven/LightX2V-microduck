"""Scene portability, observer isolation and offline-video evidence regressions."""

import copy
import json

import pytest
from lightx2v_microduck.studio.assets import asset_identity, catalog, parts
from lightx2v_microduck.studio.environments import export_environment, import_environment
from lightx2v_microduck.studio.scene import collision_boxes, sample_scene, validate_scene
from lightx2v_microduck.studio.scene_jobs import apply_changes
from lightx2v_microduck.studio.schema import SceneSpec, SkillDraft, VideoOptions, digest, file_hash, write_json
from lightx2v_microduck.studio.store import Store
from lightx2v_microduck.studio.video import Review, frame_plan, freeze_source, validate_output


def furnished():
    s = SceneSpec().model_dump()
    s.update(version=2, size=4)
    s["objects"].append(dict(id="desk", kind="obstacle", color="gray", x=-1, y=0.5, yaw=30, width=0.9, depth=0.55, height=0.65, jitter=0, asset_id="table", asset_version="1.0.0", material="wood"))
    return s


def test_v1_hash_retained_and_edit_version_explicit():
    # Legacy persisted scenes have already passed request validation. Keep the
    # old default-template serialization as well (ints in unvalidated defaults).
    assert digest(SceneSpec().model_dump()) == "bef4f7809ae901a0f1ad3c5c447f4f80646de301701d83abc9eccaf0c9739f33"
    old = SceneSpec.model_validate(SceneSpec().model_dump()).model_dump()
    assert "version" not in old and "asset_id" not in old["objects"][0]
    assert digest(SceneSpec.model_validate(json.loads(json.dumps(old))).model_dump()) == digest(old)
    new = furnished()
    with pytest.raises(ValueError, match="version 2"):
        SceneSpec.model_validate({**new, "version": 1})
    assert SceneSpec.model_validate(new).model_dump()["version"] == 2


def test_environment_portability_and_tamper_rejection():
    s = validate_scene(furnished())
    package = export_environment(s)
    assert import_environment(json.loads(json.dumps(package))) == s
    changed = copy.deepcopy(package)
    changed["scene"]["objects"][0]["x"] += 0.1
    with pytest.raises(ValueError, match="checksum"):
        import_environment(changed)
    changed = copy.deepcopy(package)
    changed["assets"][0]["parts"][0][0][0] = 42
    changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="registry"):
        import_environment(changed)


def test_asset_shapes_collisions_and_sampling():
    scene = furnished()
    obj = scene["objects"][-1]
    assert len(parts(obj)) == len(collision_boxes(obj)) == 5
    p = sample_scene(scene, 3, group="clear")
    q = sample_scene(scene, 4, group="clear")
    assert len(p["objects"]) == 2 and p != q
    assert asset_identity(p) == asset_identity(q)
    assert len(catalog()["assets"]) == 5
    scene["objects"][-1]["x"] = 3
    with pytest.raises(ValueError, match="边界"):
        validate_scene(scene)


def test_llm_outputs_are_data_only_and_validate_before_apply():
    s = furnished()
    patch = {k: s[k] for k in ("objects", "spawn", "target_id", "size")}
    assert apply_changes(s, patch) == s
    with pytest.raises(ValueError):
        apply_changes(s, {**patch, "code": "arbitrary()"})
    patch = copy.deepcopy(patch)
    patch["objects"][-1]["x"] = 0
    patch["objects"][-1]["y"] = 0
    with pytest.raises(ValueError, match="重叠"):
        apply_changes(s, patch)


def test_shared_gpu_fenced_across_training_and_video(tmp_path, monkeypatch):
    import lightx2v_microduck.studio.video as video

    monkeypatch.setattr(video, "freeze_source", lambda *args: {"sha256": "frozen"})
    store = Store(tmp_path)
    try:
        s = store.save_skill(SkillDraft().model_dump())
        p = {"source_sha256": "source", "worker_image": "sim", "video_image": "video"}
        w = {"source_sha256": "source", "image": "sim", "gpu_uuid": "GPU-1", "exclusive_device_lock": True}
        train = store.submit(s["id"], {"kind": "pipeline"}, p)
        vid = store.submit(s["id"], {"kind": "video_enhance", "prompt": "wood"}, p)
        first = store.claim("trainer", ["gpu"], w)
        assert first["id"] == train["id"]
        assert store.claim("video", ["gpu", "video"], {**w, "image": "video"}) is None
        store.update(first["id"], first["lease"], status="complete")
        second = store.claim("video", ["gpu", "video"], {**w, "image": "video"})
        assert second["id"] == vid["id"]
    finally:
        store.close()


@pytest.mark.parametrize("duration", [5, 7, 10, 14.9, 15])
def test_video_frame_time_mapping_and_actual_output(duration):
    options = {"duration_s": duration, "start_s": 2, "width": 960, "height": 544}
    p = frame_plan(options, {"fps": 10, "frames": 300})
    assert (p["frames"] - 5) % 17 == 0 and 5 <= p["duration_s"] <= 15
    assert p["source_times_s"][0] == 2 and p["source_times_s"][-1] <= 2 + duration
    info = {"width": 960, "height": 544, "fps": 24, "frames": p["frames"], "duration_s": p["duration_s"]}
    assert validate_output(info, p, options)["passed"]
    assert not validate_output({**info, "width": 768}, p, options)["passed"]


def test_review_uncertain_cannot_pass_and_ranges():
    with pytest.raises(ValueError):
        Review(status="passed")
    assert Review(status="passed", **{k: "passed" for k in ("target", "layout", "passage", "occlusion", "continuity")}).status == "passed"
    with pytest.raises(ValueError):
        Review(issues=[{"start_s": 2, "end_s": 1, "note": "bad"}])
    with pytest.raises(ValueError):
        VideoOptions(source_run="a" * 16, source_file="head.mp4", width=976)


def test_source_cannot_use_generated_or_other_run(tmp_path):
    store = Store(tmp_path)
    try:
        s = store.save_skill(SkillDraft().model_dump())
        p = {"source_sha256": "s", "worker_image": "sim"}
        r = store.submit(s["id"], {"kind": "preview"}, p)
        claim = store.claim("cpu", ["cpu"], {"source_sha256": "s", "image": "sim"})
        store.update(r["id"], claim["lease"], status="complete")
        with pytest.raises(ValueError, match="original"):
            freeze_source(store, s["id"], {"source_run": r["id"], "source_file": "../../another/head.mp4"})
    finally:
        store.close()


def test_scene_api_and_observer_stream_tool_boundary(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from lightx2v_microduck.studio.api import create_app

    monkeypatch.delenv("STUDIO_TOKEN", raising=False)
    with TestClient(create_app(tmp_path)) as c:
        s = c.post("/api/skills", json=SkillDraft().model_dump()).json()
        r = c.post(f"/api/skills/{s['id']}/runs", json={"kind": "preview"}).json()
        store = c.app.state.store
        p = r["snapshot"]["provenance"]
        lease = store.claim("cpu", ["cpu"], {"source_sha256": p["source_sha256"], "image": p["worker_image"]})
        out = tmp_path / "runs" / r["id"] / "live"
        state = {"episode_id": "episode-a", "tick": 5, "geoms": [{"position": [1, 2, 3]}]}
        write_json(out / "state.json", state)
        store.update(r["id"], lease["lease"], status="complete")
        text = c.get(f"/api/runs/{r['id']}/state-stream").text
        assert "event: state" in text and "event: ended" in text
        tool = c.get(f"/api/runs/{r['id']}?tool=true").json()
        assert "geoms" not in str(tool) and "snapshot" not in tool
        env = c.post("/api/environments/export", json=furnished()).json()
        assert c.post("/api/environments/import", json=env).json()["scene"]["version"] == 2


def test_stale_layout_draft_cannot_overwrite_new_scene(tmp_path):
    store = Store(tmp_path)
    try:
        s = store.save_skill(SkillDraft().model_dump())
        old_hash = digest(s["scene"])
        changed = copy.deepcopy(s)
        changed["scene"]["spawn"]["yaw"] = 20
        store.save_skill({k: v for k, v in changed.items() if k != "id"}, s["id"])
        with pytest.raises(ValueError, match="原场景已被编辑"):
            store.apply_scene_draft(s["id"], furnished(), old_hash)
        assert store.skill(s["id"])["scene"]["spawn"]["yaw"] == 20
        # Explicitly applying against the new version succeeds atomically.
        current = store.skill(s["id"])
        assert store.apply_scene_draft(s["id"], furnished(), digest(current["scene"]))["scene"]["version"] == 2
    finally:
        store.close()


def test_video_review_is_hash_bound_and_not_skill_acceptance(tmp_path, monkeypatch):
    """Synthetic file fixture tests review bookkeeping, not video generation."""
    import io
    import zipfile

    import lightx2v_microduck.studio.video as video
    from fastapi.testclient import TestClient
    from lightx2v_microduck.studio.api import create_app

    monkeypatch.delenv("STUDIO_TOKEN", raising=False)
    monkeypatch.setattr(video, "freeze_source", lambda *args: {"sha256": "unit-test-source"})
    with TestClient(create_app(tmp_path)) as c:
        s = c.post("/api/skills", json=SkillDraft().model_dump()).json()
        r = c.post(f"/api/skills/{s['id']}/runs", json={"kind": "video_enhance", "prompt": "wood"}).json()
        p = r["snapshot"]["provenance"]
        store = c.app.state.store
        lease = store.claim(
            "video",
            ["gpu", "video"],
            {
                "source_sha256": p["source_sha256"],
                "image": p["video_image"],
                "gpu_uuid": "GPU-test",
                "exclusive_device_lock": True,
            },
        )
        out = tmp_path / "runs" / r["id"] / "video"
        out.mkdir(parents=True)
        (out / "generated.mp4").write_bytes(b"unit-test-only-not-a-model-output")
        write_json(out / "time-map.json", {"fixture": True})
        write_json(
            out / "provenance.json",
            {
                "generated_sha256": file_hash(out / "generated.mp4"),
                "effective": {"duration_s": 5},
                "automatic_checks": {"passed": True},
            },
        )
        store.update(r["id"], lease["lease"], status="complete", result={"fixture": True})
        before = store.run(r["id"])["result"]
        url = f"/api/runs/{r['id']}/video-reviews"
        assert c.post(url, json={"status": "passed"}).status_code == 422
        assert c.post(url, json={"status": "uncertain"}).json()["status"] == "uncertain"
        good = {"status": "passed", **{k: "passed" for k in ("target", "layout", "passage", "occlusion", "continuity")}}
        assert c.post(url, json=good).json()["skill_acceptance"] is False
        assert len(c.get(url).json()) == 2
        assert store.run(r["id"])["result"] == before
        package = c.get(f"/api/runs/{r['id']}/enhanced-export")
        with zipfile.ZipFile(io.BytesIO(package.content)) as z:
            assert json.loads(z.read("review.json"))["status"] == "passed"
            assert "AI_GENERATED.mp4" in z.namelist()
        (out / "generated.mp4").write_bytes(b"changed")
        assert c.get(url).status_code == 409
