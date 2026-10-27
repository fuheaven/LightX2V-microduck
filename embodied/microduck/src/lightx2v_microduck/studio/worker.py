"""Sequential Docker worker. Web owns the queue; worker owns its child process."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import httpx
from .schema import source_identity, write_json


def runtime_receipt(image, device):
    import importlib.metadata

    packages = {}
    for name in ("torch", "torchvision", "transformers", "mujoco", "onnxruntime", "fastapi", "pydantic", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    receipt = {"python": sys.version.split()[0], "packages": packages, "image": image, "image_digest": os.environ.get("STUDIO_IMAGE_DIGEST"), "device": device, "uid": os.getuid()}
    if device == "cuda":
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True)
            receipt["gpu"] = r.stdout.strip().splitlines()
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            receipt["gpu_receipt_error"] = type(exc).__name__
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--image", default=os.environ.get("STUDIO_WORKER_IMAGE", "local-development"))
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    root = Path(args.workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # An OS lock survives HTTP/server restarts and cannot be stolen by another worker.
    lock = (root / f"worker-{args.id}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repo = Path(__file__).resolve().parents[5]
    headers = {"Authorization": "Bearer " + os.environ["STUDIO_WORKER_TOKEN"]}
    client = httpx.Client(base_url=args.server, headers=headers, timeout=20)
    stopping = False
    receipt = runtime_receipt(args.image, args.device)

    def terminate(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    while not stopping:
        try:
            res = client.post(
                "/api/worker/claim",
                json={
                    "id": args.id,
                    "capabilities": ["cpu"] + (["gpu"] if args.device == "cuda" else []),
                    "provenance": {"image": args.image, "source_sha256": source_identity(repo), "device": args.device},
                },
            )
            res.raise_for_status()
            job = res.json()
        except httpx.HTTPError:
            if args.once:
                raise
            time.sleep(5)
            continue
        if job is None:
            if args.once:
                break
            time.sleep(3)
            continue
        out = root / "runs" / job["id"]
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "snapshot.json", job["snapshot"])
        write_json(out / "runtime.json", receipt)
        env = {**os.environ, "STUDIO_DEVICE": args.device}
        log = (out / "worker.log").open("a")
        proc = subprocess.Popen(
            [sys.executable, "-m", "lightx2v_microduck.studio.pipeline", "--workspace", str(root), "--run", job["id"]], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        last_ok = time.monotonic()
        cancel_at = None
        try:
            while proc.poll() is None:
                progress = {}
                if (out / "progress.json").exists():
                    progress = json.loads((out / "progress.json").read_text())
                try:
                    res = client.post(f"/api/worker/runs/{job['id']}", json={"lease": job["lease"], "progress": progress})
                    res.raise_for_status()
                    last_ok = time.monotonic()
                    cancel = bool(res.json()["cancel"])
                except httpx.HTTPError as exc:
                    # Lost lease/server cannot leave an unowned GPU computation running.
                    cancel = time.monotonic() - last_ok > 45 or (isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 409)
                if (cancel or stopping) and cancel_at is None:
                    (out / "cancel").touch()
                    os.killpg(proc.pid, signal.SIGTERM)
                    cancel_at = time.monotonic()
                if cancel_at is not None and time.monotonic() - cancel_at > 60:
                    os.killpg(proc.pid, signal.SIGKILL)
                time.sleep(2)
            payload = (
                json.loads((out / "result.json").read_text())
                if (out / "result.json").exists()
                else {"status": "interrupted" if cancel_at else "failed", "result": {"reason": f"worker process exited {proc.returncode}"}}
            )
            res = client.post(f"/api/worker/runs/{job['id']}", json={"lease": job["lease"], **payload})
            res.raise_for_status()
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
        if args.once:
            break


if __name__ == "__main__":
    main()
