"""Run the unmodified upstream duck-body and robotd as a separate PD twin.

This path invokes pinned upstream Python source directly; it does not import
mjlab or replace Body/World with the LightX2V BAM body server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .assets import resolve_asset_root, verify_source
from .contracts import JOINT_NAMES
from .data import file_sha256, write_json


def _verified_root(asset_root=None):
    root = resolve_asset_root(asset_root)
    for name in ("microduck", "microduck_rl", "microduck-policies"):
        result = verify_source(name, root)
        if not result["ok"]:
            raise ValueError(f"pinned upstream assets failed verification: {result}")
    return root


def build_official_robotd(asset_root=None, *, jobs=8):
    """Compile pristine runtime into its own target; never overwrite BAM build."""
    root = _verified_root(asset_root)
    target = root.parent / "build/microduck-official-target"
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    subprocess.run([cargo, "build", "--locked", "-j", str(jobs), "--manifest-path", str(root / "microduck/Cargo.toml"), "--target-dir", str(target), "-p", "robotd", "-p", "robotctl"], check=True)
    binary = target / "debug/robotd"
    write_json(
        target / "official-build.json",
        {
            "source_revision": "5620aa214e6e304eec00d5c376ac81d3001fb7a3",
            "source_manifest_sha256": file_sha256(root / "microduck/Cargo.toml"),
            "source_lock_sha256": file_sha256(root / "microduck/Cargo.lock"),
            "robotd_sha256": file_sha256(binary),
            "external_policy_extension": False,
        },
    )
    return binary


def _body_environment(root):
    env = dict(os.environ)
    paths = [str(root / "microduck_rl/src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("MUJOCO_GL", "egl")
    # This path intentionally exercises the original daemon API only.
    env.pop("MICRODUCK_EXTERNAL_POLICY_SOCKET", None)
    return env


def _stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


class _BodyReader:
    """Read-only RemoteIo client for independent protocol/physics receipts."""

    def __init__(self, address):
        self.socket = socket.create_connection(address, timeout=2)
        self.stream = self.socket.makefile("rwb")
        try:
            if self.call("hello", protocol=1).get("protocol") != 1:
                raise ValueError("upstream RemoteIo protocol mismatch")
        except Exception:
            self.close()
            raise

    def call(self, op, **fields):
        self.stream.write((json.dumps({"op": op, **fields}) + "\n").encode())
        self.stream.flush()
        response = json.loads(self.stream.readline(65537))
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self):
        self.stream.close()
        self.socket.close()


def smoke_upstream_daemon(asset_root=None, output_dir="runs/upstream-daemon", *, duration_s=2.0, build=False, keyframe="SIT"):
    """Real two-process health smoke; no task-success or sim2real conclusion."""
    if not np.isfinite(duration_s) or not 0.1 <= duration_s <= 60:
        raise ValueError("health smoke duration must be 0.1..60 seconds")
    root = _verified_root(asset_root)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "upstream-daemon.json").exists():
        raise FileExistsError("upstream daemon smoke output already exists")
    target = root.parent / "build/microduck-official-target"
    binary = build_official_robotd(root) if build else target / "debug/robotd"
    receipt_path = target / "official-build.json"
    if not binary.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("build pristine robotd first with upstream_daemon build, or smoke --build")
    build_receipt = json.loads(receipt_path.read_text())
    if build_receipt["robotd_sha256"] != file_sha256(binary) or build_receipt["external_policy_extension"]:
        raise ValueError("official daemon binary differs from its pristine build receipt")
    env = _body_environment(root)
    if "ORT_DYLIB_PATH" not in env:
        import onnxruntime

        libraries = sorted((Path(onnxruntime.__file__).parent / "capi").glob("libonnxruntime.so*"))
        if not libraries:
            raise FileNotFoundError("set ORT_DYLIB_PATH to ONNX Runtime")
        env["ORT_DYLIB_PATH"] = str(libraries[0])
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    body_command = [sys.executable, "-m", "mjlab_microduck.sim.body_server", "--headless", "--host", "127.0.0.1", "--port", str(port), "--keyframe", keyframe]
    body = daemon = reader = control = None
    samples, error, startup_health, depth = [], None, None, None
    # Unix sockets have a ~108-byte address limit; run roots can be longer.
    with tempfile.TemporaryDirectory(prefix="lxofficial-") as sockets, (output / "body.log").open("w") as body_log, (output / "robotd.log").open("w") as daemon_log:
        from .daemon import ExternalPolicyClient

        try:
            body = subprocess.Popen(body_command, env=env, stdout=body_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 20
            while reader is None:
                if body.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("original upstream duck-body failed startup; see body.log")
                try:
                    reader = _BodyReader(("127.0.0.1", port))
                except OSError:
                    time.sleep(0.05)
            paths = {
                "walk": "alpha_walking.onnx",
                "stand": "alpha_stand.onnx",
                "sitstand": "alpha_sitstand.onnx",
                "ground_pick": "alpha_ground_pick.onnx",
                "kick_left": "ball_kick_left.onnx",
                "kick_right": "ball_kick_right.onnx",
                "roulade": "roulade.onnx",
            }
            params = output / "robotd.toml"
            params.write_text('[policy]\nenabled = true\nmode = "walk"\n' + "".join(f"{name} = {json.dumps(str(root / 'microduck-policies' / filename))}\n" for name, filename in paths.items()))
            control = ExternalPolicyClient(Path(sockets) / "robotd.sock")
            daemon_command = [str(binary), "--sim", f"127.0.0.1:{port}", "--socket", control.socket_path, "--params", str(params)]
            daemon = subprocess.Popen(daemon_command, env=env, stdout=daemon_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 20
            enabled = False
            while time.monotonic() < deadline:
                if daemon.poll() is not None or body.poll() is not None:
                    raise RuntimeError("upstream body/daemon exited during startup; see logs")
                try:
                    if not enabled:
                        enabled = bool(control.call("robot.enable", {"on": True}).get("accepted"))
                    startup_health = control.call("robot.health")
                    if enabled and startup_health.get("healthy"):
                        break
                except (OSError, RuntimeError):
                    pass
                time.sleep(0.05)
            else:
                raise RuntimeError(f"pristine daemon did not become healthy: {startup_health}")
            start = time.monotonic()
            while time.monotonic() - start < duration_s:
                sensor = reader.call("read")
                if len(sensor["positions"]) != 15 or len(sensor["velocities"]) != 15 or not np.isfinite(sensor["positions"] + sensor["velocities"]).all():
                    raise ValueError("upstream body emitted invalid 15-joint sensor data")
                samples.append({"wall_time_s": time.monotonic() - start, "body": sensor, "health": control.call("robot.health")})
                time.sleep(min(0.1, max(0, duration_s - (time.monotonic() - start))))
            depth = reader.call("tof")
            if depth["rows"] != 8 or depth["cols"] != 8 or len(depth["distance_mm"]) != 64:
                raise ValueError("upstream ToF does not expose its 8x8 wire contract")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if control is not None and daemon is not None and daemon.poll() is None:
                try:
                    control.call("robot.stop")
                    control.call("robot.enable", {"on": False})
                except (OSError, RuntimeError):
                    pass
            if reader is not None:
                reader.close()
            _stop(daemon)
            _stop(body)
    sim_delta = samples[-1]["body"]["sim_time"] - samples[0]["body"]["sim_time"] if len(samples) > 1 else 0.0
    wall_delta = samples[-1]["wall_time_s"] - samples[0]["wall_time_s"] if len(samples) > 1 else 0.0
    report = {
        "schema_version": 1,
        "profile": "official_daemon",
        "passed": error is None and sim_delta > 0 and all(s["health"].get("healthy") for s in samples),
        "error": error,
        "hardware_validated": False,
        "task_behavior_validated": False,
        "external_vla_validated": False,
        "runtime_revision": "5620aa214e6e304eec00d5c376ac81d3001fb7a3",
        "rl_revision": "2b581c641406a48346e696212930ea881c222c52",
        "body_source_sha256": file_sha256(root / "microduck_rl/src/mjlab_microduck/sim/body_server.py"),
        "daemon_build": build_receipt,
        "body_command": body_command,
        "joint_names": list(JOINT_NAMES),
        "keyframe": keyframe,
        "cameras_enabled": False,
        "tof": depth,
        "actuator": "upstream MJCF position actuators; gain scaled by kp/200; BAM runtime is not used",
        "unmodeled": ["BAM voltage/back-EMF/friction dynamics", "calibrated motor currents", "battery discharge", "servo temperature", "camera capture in this headless smoke"],
        "physics_dt_s": 0.005,
        "control_dt_s": 0.020,
        "samples": samples,
        "simulated_seconds": sim_delta,
        "measured_seconds": wall_delta,
        "sim_to_wall_ratio": sim_delta / wall_delta if wall_delta else None,
        "scope": "original upstream duck-body plus unmodified robotd, RemoteIo/ToF/daemon health only",
    }
    write_json(output / "upstream-daemon.json", report)
    if not report["passed"]:
        reason = error or "daemon became unhealthy or physics did not advance during the smoke"
        raise RuntimeError(f"{reason}; receipt: {output / 'upstream-daemon.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "smoke"))
    parser.add_argument("--asset-root")
    parser.add_argument("--output", default="runs/upstream-daemon")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--keyframe", default="SIT")
    args = parser.parse_args()
    if args.action == "build":
        print(build_official_robotd(args.asset_root))
    else:
        report = smoke_upstream_daemon(args.asset_root, args.output, duration_s=args.seconds, build=args.build, keyframe=args.keyframe)
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
