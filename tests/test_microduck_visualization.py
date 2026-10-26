import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from lightx2v_microduck.visualization import (
    ACTIVE, MEDIA, BusyError, LiveJobs, load_catalog, make_server, validate_run,
)


@pytest.fixture
def server(tmp_path):
    service = make_server("127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    yield service, f"http://127.0.0.1:{service.server_port}"
    service.shutdown()
    service.jobs.close()
    service.server_close()
    thread.join(2)


def request(base, path, body=None, headers=None):
    return urlopen(Request(base + path, data=None if body is None else json.dumps(body).encode(),
                           headers={"Content-Type": "application/json", **(headers or {})}), timeout=4)


def test_gallery_preserves_cohorts_and_success_evidence():
    catalog = load_catalog()
    assert [(g["passed"], g["total"], g["mode"]) for g in catalog["groups"]] == [
        (59, 60, "realtime"), (27, 60, "offline"), (2, 21, "realtime")]
    clips = [clip for group in catalog["groups"] for clip in group["clips"]]
    assert len(clips) == 9
    assert {c["task"] for c in clips if c["passed"]} == {"find_red_ball", "find_blue_ball", "find_yellow_ball"}
    assert any(not c["passed"] for c in clips)


def test_wrong_score_and_changed_video_fail_closed(tmp_path):
    target = tmp_path / "media"
    shutil.copytree(MEDIA, target)
    catalog = json.loads((target / "catalog.json").read_text())
    catalog["groups"][0]["passed"] = 60
    (target / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="counts disagree"):
        load_catalog(target)
    catalog["groups"][0]["passed"] = 59
    (target / "catalog.json").write_text(json.dumps(catalog))
    (target / "red-ball.mp4").write_bytes(b"bad video")
    with pytest.raises(ValueError, match="checksum"):
        load_catalog(target)


def test_video_ranges_and_filesystem_boundary(server):
    _, base = server
    with request(base, "/media/red-ball.mp4", headers={"Range": "bytes=0-31"}) as response:
        assert response.status == 206
        assert response.headers["Content-Range"].startswith("bytes 0-31/")
        assert response.read() == (MEDIA / "red-ball.mp4").read_bytes()[:32]
    with request(base, "/media/red-ball.mp4", headers={"Range": "bytes=-12"}) as response:
        assert response.read() == (MEDIA / "red-ball.mp4").read_bytes()[-12:]
    for value in ("bytes=999999999-", "bytes=4-3", "bytes=0-2,4-7", "bytes=-0"):
        with pytest.raises(HTTPError) as exc:
            request(base, "/media/red-ball.mp4", headers={"Range": value})
        assert exc.value.code == 416
    for path in ("/media/../../.ssh/id_rsa", "/media/%2e%2e/catalog.json", "/api/live/bad/frame/camera", "/media/catalog.json"):
        with pytest.raises(HTTPError) as exc:
            request(base, path)
        assert exc.value.code == 404


def test_gallery_without_models_and_cross_origin_writes(server):
    _, base = server
    with request(base, "/api/health") as response:
        health = json.load(response)
        assert health["status"] == "ok" and not health["fastwam_available"]
    with request(base, "/api/live") as response:
        assert json.load(response) == {"status": "idle"}
    with pytest.raises(HTTPError) as exc:
        request(base, "/api/live", {"policy": "fastwam"})
    assert exc.value.code == 400
    with pytest.raises(HTTPError) as exc:
        request(base, "/api/live", {}, {"Origin": "https://another.example"})
    assert exc.value.code == 403


@pytest.mark.parametrize("payload", [[], {"seed": True}, {"seed": -1}, {"seconds": 10000},
                                      {"policy": "shell"}, {"scene": "../../"}, {"checkpoint": "/tmp/x"}])
def test_no_arbitrary_commands_paths_or_unbounded_sessions(payload):
    with pytest.raises(ValueError):
        validate_run(payload)


def test_process_isolation_busy_cancel_and_restart(tmp_path, monkeypatch):
    jobs = LiveJobs(tmp_path, enable_live=True)
    monkeypatch.setattr(jobs, "health", lambda: {"official_available": True})
    popen = subprocess.Popen
    def child(command, **kwargs):
        # A real independently running child: the server must reject a second
        # job, terminate its process group, and record cancellation honestly.
        return popen([command[0], "-c", "import time; time.sleep(30)"], **kwargs)
    monkeypatch.setattr(subprocess, "Popen", child)
    first = jobs.start({"policy": "official"})
    assert validate_run(first["config"]) == first["config"]
    assert jobs.processes[first["id"]].poll() is None
    with pytest.raises(BusyError):
        jobs.start({"policy": "official"})
    jobs.stop(first["id"])
    jobs.close()
    deadline = time.monotonic() + 3
    while jobs.get(first["id"])["status"] in ACTIVE and time.monotonic() < deadline:
        time.sleep(.01)
    assert jobs.get(first["id"])["status"] == "cancelled"
    restored = LiveJobs(tmp_path)
    assert restored.get(first["id"])["status"] == "cancelled"
    assert not restored.processes


def test_proxy_offline_keeps_gallery_available(tmp_path):
    service = make_server("127.0.0.1", 0, tmp_path, worker_url="http://127.0.0.1:1")
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{service.server_port}"
    try:
        with request(base, "/api/health") as response:
            assert json.load(response)["status"] == "worker_unavailable"
        with request(base, "/api/catalog") as response:
            assert len(json.load(response)["groups"]) == 3
        with request(base, "/media/red-ball.mp4", headers={"Range": "bytes=0-20"}) as response:
            assert response.status == 206
    finally:
        service.shutdown()
        service.server_close()
        thread.join(2)
