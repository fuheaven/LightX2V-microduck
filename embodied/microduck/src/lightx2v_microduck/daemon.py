"""Build the pinned real robotd with an opt-in external policy source.

The body server implements the official RemoteIo protocol against BAM physics.
It is a simulation transport, never a servo-bus implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .assets import RESOURCE_DIR, resolve_asset_root, sha256_file, verify_source
from .contracts import POLICY_INDICES, JointResidualChunk, Observation, pack_observation
from .execution_profiles import begin_execution_task, is_project_profile, make_execution_environment, prescribed_command, resolve_execution_recipe

LEGACY_EXECUTION_MODE = "source_tick_age_compensated"
PREFIX_EXECUTION_MODE = "prefix_playback_v1"


def prepare_robotd(asset_root=None, output=None) -> Path:
    root = resolve_asset_root(asset_root)
    checked = verify_source("microduck", root)
    if not checked["ok"]:
        raise ValueError(f"pinned runtime failed verification: {checked}")
    source = root / "microduck"
    output = Path(output or root.parent / "build" / "microduck-external-v5").resolve()
    manifest_path = output / ".lightx2v-extension.json"
    if output.exists():
        if not manifest_path.exists():
            raise ValueError(f"refusing to modify unowned runtime checkout {output}")
        previous = json.loads(manifest_path.read_text())
        for name, digest in previous["generated_files"].items():
            if sha256_file(output / name) != digest:
                raise ValueError(f"local runtime source changed: {output / name}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, output, ignore=shutil.ignore_patterns(".asset-receipt.json"))
    main = (source / "robotd/src/main.rs").read_text()
    control = (source / "robotd/src/control.rs").read_text()
    io_source = (source / "duck-control/src/io.rs").read_text()
    sim_source = (source / "duck-control/src/sim.rs").read_text()
    safety_source = (source / "duck-control/src/safety.rs").read_text()

    def replace_once(text, old, new):
        if text.count(old) != 1:
            raise ValueError(f"runtime patch anchor mismatch: {old[:70]}")
        return text.replace(old, new, 1)

    main = replace_once(main, "mod control;", "mod control;\nmod external;")
    # Optional transport metadata travels alongside the exact sensor read. It
    # does not modify Sensors, motion values, or the original 61D contract.
    capture_struct = '''#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct RemoteCapture {
    pub protocol: u32,
    pub body_tick: u64,
    pub task_tick: u64,
    pub task_epoch: u64,
    pub episode_id: String,
    pub capture_monotonic_ns: u64,
    pub command13: [f32; 13],
    pub command_program_sha256: Option<String>,
}

'''
    io_source = replace_once(io_source, "pub trait RobotIo {", capture_struct + "pub trait RobotIo {\n    fn source_capture(&self) -> Option<RemoteCapture> { None }\n")
    sim_source = replace_once(sim_source, "struct SensorFrame {", "struct SensorFrame {\n    #[serde(default)]\n    source_capture: Option<crate::io::RemoteCapture>,")
    sim_source = replace_once(sim_source, "pub struct RemoteIo {", "pub struct RemoteIo {\n    source_capture: Option<crate::io::RemoteCapture>,")
    sim_source = replace_once(sim_source, "            addr: addr.into(),", "            addr: addr.into(),\n            source_capture: None,")
    sim_source = replace_once(sim_source, "impl RobotIo for RemoteIo {", "impl RobotIo for RemoteIo {\n    fn source_capture(&self) -> Option<crate::io::RemoteCapture> { self.source_capture.clone() }\n")
    sim_source = replace_once(sim_source, "        let frame: SensorFrame = self.call(&Request::Read)?;", "        let frame: SensorFrame = self.call(&Request::Read)?;\n        self.source_capture = frame.source_capture.clone();")
    safety_source = replace_once(safety_source, "    pub fn read(&mut self) -> Result<Sensors, IoError> {", "    pub fn source_capture(&self) -> Option<crate::io::RemoteCapture> { self.io.source_capture() }\n\n    pub fn read(&mut self) -> Result<Sensors, IoError> {")
    main = replace_once(main, "                match controller.step(sensors, &command, snapshot.pose.active, dt, scale_mult) {", "                external::set_source_capture(safety.source_capture());\n                match controller.step(sensors, &command, snapshot.pose.active, dt, scale_mult) {")
    main = replace_once(main, "proto::Call::RobotStop => {\n            intents.stop();", 'proto::Call::RobotStop => {\n            external::revoke("robot.stop");\n            intents.stop();')
    main = replace_once(main, "proto::Call::RobotEnable(p) => {", 'proto::Call::RobotEnable(p) => {\n            external::revoke("robot.enable changed");')
    control = replace_once(
        control,
        "pub fn new(policy: Policy, tuning: Tuning, skills: SkillTuning) -> Self {\n        Self {",
        "pub fn new(policy: Policy, tuning: Tuning, skills: SkillTuning) -> Self {\n        crate::external::initialize();\n        Self {",
    )
    control = replace_once(control, "pub fn reset(&mut self) {\n        self.last_action", 'pub fn reset(&mut self) {\n        crate::external::revoke("controller_reset");\n        self.last_action')
    anchor = "        let action = self.policy.infer(&observation, net)?;"
    replacement = """        // The external policy contributes a raw action inside the controller.
        // Local intents and scripted skills preempt it; outer Safety still owns
        // all motor limits, fall handling, and torque. No output low-pass: this
        // profile is trained with BAM and unfiltered residuals at scale 1.
        let permitted = !self.busy() && !self.is_sitting() && !body_active
            && command.twist_magnitude() < 1e-8
            && command.head.iter().all(|value| value.abs() < 1e-8);
        if let Some(action) = crate::external::sample(observation.as_slice(), permitted) {
            self.last_action = action;
            let offsets = Observation::scatter_action(&action);
            let targets = std::array::from_fn(|joint| DEFAULT_POSITION[joint] + offsets[joint]);
            self.previous = Some(targets);
            return Ok(Step { targets, label: "external_vla".into(), gain: self.tuning.gain, busy: false });
        }
        let action = self.policy.infer(&observation, net)?;"""
    control = replace_once(control, anchor, replacement)
    (output / "robotd/src/main.rs").write_text(main)
    (output / "robotd/src/control.rs").write_text(control)
    (output / "duck-control/src/io.rs").write_text(io_source)
    (output / "duck-control/src/sim.rs").write_text(sim_source)
    (output / "duck-control/src/safety.rs").write_text(safety_source)
    shutil.copyfile(RESOURCE_DIR / "robotd/external.rs", output / "robotd/src/external.rs")
    generated = {name: sha256_file(output / name) for name in ("robotd/src/main.rs", "robotd/src/control.rs", "robotd/src/external.rs", "duck-control/src/io.rs", "duck-control/src/sim.rs", "duck-control/src/safety.rs")}
    manifest_path.write_text(json.dumps({"upstream_revision": checked["revision"], "external_protocol": 1, "source_capture_protocol": 1,
                                       "execution_receipt_protocol": 1, "read_snapshot_protocol": 1, "continuous_control_ticks": True,
                                       "supported_execution_modes": [LEGACY_EXECUTION_MODE, PREFIX_EXECUTION_MODE],
                                       "generated_files": generated, "hardware_validated": False}, indent=2) + "\n")
    return output


def build_robotd(asset_root=None, output=None):
    directory = prepare_robotd(asset_root, output)
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    subprocess.run([cargo, "build", "--locked", "-p", "robotd", "-p", "robotctl"], cwd=directory, check=True)
    return directory / "target/debug/robotd"


class ExternalPolicyClient:
    def __init__(self, socket_path, timeout=2.0):
        self.socket_path = str(socket_path)
        self.timeout = timeout
        self.episode_id = None
        self.execution_mode = LEGACY_EXECUTION_MODE
        self._request_id = 0
        self._request_lock = threading.Lock()

    def call(self, method, params=None):
        with self._request_lock:
            self._request_id += 1
            request_id = self._request_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.socket_path)
            connection.sendall((json.dumps(payload, allow_nan=False) + "\n").encode())
            with connection.makefile("rb") as stream:
                line = stream.readline(65537)
        if len(line) > 65536:
            raise ValueError("oversized robotd response")
        response = json.loads(line)
        if response.get("id") != request_id:
            raise ValueError("mismatched robotd response id")
        if "error" in response:
            raise RuntimeError(response["error"]["message"])
        return response["result"]

    def open(self, episode_id, *, execution_mode=LEGACY_EXECUTION_MODE):
        if execution_mode not in (LEGACY_EXECUTION_MODE, PREFIX_EXECUTION_MODE):
            raise ValueError("unknown external execution mode")
        params = {"episode_id": episode_id}
        if execution_mode == PREFIX_EXECUTION_MODE:
            # Old binaries may ignore unknown request keys. Check advertised
            # support before sending an open that could silently select legacy.
            if execution_mode not in self.observe().get("supported_execution_modes", []):
                raise ValueError("robotd does not support prefix_playback_v1; build the independent v5 binary")
            params["execution_mode"] = execution_mode
        result = self.call("policy.external.open", params)
        if result.get("execution_mode", LEGACY_EXECUTION_MODE) != execution_mode:
            self.call("policy.external.close", {"episode_id": episode_id})
            raise ValueError("robotd negotiated another execution mode")
        self.episode_id = episode_id
        self.execution_mode = execution_mode
        return result

    def observe(self):
        return self.call("policy.external.observation")

    def receipts(self, after_sequence=0, limit=32):
        return self.call("policy.external.receipts", {"after_sequence": after_sequence, "limit": limit})

    def submit(self, chunk: JointResidualChunk):
        if chunk.action_profile != "raw_residual":
            raise ValueError("decode/denormalize joint actions before submitting to robotd")
        params = {"episode_id": chunk.episode_id, "source_tick": chunk.source_tick, "actions": chunk.actions.tolist(), "control_dt_s": chunk.control_dt_s}
        if self.execution_mode == PREFIX_EXECUTION_MODE:
            if len(chunk.actions) != 50:
                raise ValueError("prefix_playback_v1 requires the public 50-action prediction")
            params.update(actions=chunk.actions[:5].tolist(), execution_mode=PREFIX_EXECUTION_MODE,
                          prediction_horizon=50, execution_prefix=5)
        if chunk.valid_until_tick is not None:
            params["valid_until_tick"] = chunk.valid_until_tick
        return self.call("policy.external.push_chunk", params)

    def close(self):
        if self.episode_id is not None:
            result = self.call("policy.external.close", {"episode_id": self.episode_id})
            self.episode_id = None
            return result


class BamBodyServer:
    """Official RemoteIo wire contract, with explicitly named BAM dynamics.

    Physics advances on its own thread regardless of inference or RPC clients.
    A separate rendering thread copies MjData briefly under the physics lock.
    """

    def __init__(self, env, host="127.0.0.1", port=0, cameras=True):
        if env.physics is None or env.profile != "official_bam":
            raise ValueError("BamBodyServer requires a reset official_bam environment")
        self.env = env
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.targets = env.physics.last_targets.copy()
        self.torque_on = False
        self.released = False
        self.initial = env.physics.clone_state()
        self.frame = None
        self.frame_time = 0.0
        self.frame_tick = 0
        self.fault = None
        self.steps = 0
        self.late_ticks = 0
        self.started = 0.0
        self.task_epoch = 0
        self.task_origin_body_tick = None
        self._command_key = None
        self._command = np.zeros(13, np.float32)
        self.handoff = None
        self.step_times_ns = deque(maxlen=4096)
        self.command_program_sha256 = None
        if hasattr(env, "teacher_version") or env.task.task_id == "ground_lower":
            from .qualified_teachers import training_contracts
            self.command_program_sha256 = training_contracts()[env.task.task_id]["command_program_sha256"]
        body = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while not body.stop_event.is_set():
                    line = self.rfile.readline(65537)
                    if not line or len(line) > 65536:
                        return
                    try:
                        reply = body.dispatch(json.loads(line))
                    except Exception as exc:
                        reply = {"error": str(exc)}
                    self.wfile.write((json.dumps(reply, allow_nan=False) + "\n").encode())
                    self.wfile.flush()

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server((host, port), Handler)
        self.address = self.server.server_address
        self.threads = [threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True), threading.Thread(target=self._physics_loop, daemon=True)]
        if cameras:
            self.threads.append(threading.Thread(target=self._camera_loop, daemon=True))

    def start(self):
        self.started = time.monotonic()
        for thread in self.threads:
            thread.start()
        return self

    def dispatch(self, request):
        op = request.get("op")
        with self.lock:
            physics = self.env.physics
            if op == "hello":
                if request.get("protocol") != 1 or request.get("joints", 15) != 15:
                    raise ValueError("RemoteIo protocol 1 / 15 joints required")
                return {"protocol": 1}
            if op == "read":
                gyro, gravity, positions, velocities = physics.sensors()
                self._update_command((gyro, gravity, positions, velocities))
                return {
                    "positions": positions.tolist(),
                    "velocities": velocities.tolist(),
                    "currents_ma": [0.0] * 15,
                    "imu": {"gyro": gyro.tolist(), "gravity": gravity.tolist(), "quat": physics.data.xquat[physics.trunk].tolist()},
                    "source_capture": {"protocol": 1, "body_tick": self.steps, "task_tick": self.env.tick,
                                       "task_epoch": self.task_epoch, "episode_id": self.env.episode_id,
                                       "capture_monotonic_ns": time.monotonic_ns(), "command13": self._command.tolist(),
                                       "command_program_sha256": self.command_program_sha256},
                }
            if op == "write":
                values = np.asarray(request["targets"], dtype=np.float64)
                if values.shape != (15,) or not np.isfinite(values).all():
                    raise ValueError("15 finite named-order targets required")
                self.targets = values[list(POLICY_INDICES)].copy()
                return {}
            if op == "gain":
                gain = float(request["kp"])
                if not np.isfinite(gain) or not 0 <= gain <= 16383:
                    raise ValueError("invalid servo firmware gain")
                physics.bam_model.actuator.kp = gain
                return {}
            if op == "torque":
                self.torque_on = bool(request["on"])
                if self.torque_on:
                    self.released = True
                    self.targets = physics.data.qpos[physics.qpos_indices].copy()
                return {}
            if op == "slow":
                return {"volts": 7.4, "temps_c": [32.0] * 15}
            if op == "tof":
                distances, metadata = physics.read_tof(self.steps)
                if distances is None:
                    raise ValueError("ToF unavailable")
                return {"rows": 8, "cols": 8, "distance_mm": (distances * 1000).astype(int).ravel().tolist(), "status": np.asarray(metadata["statuses"]).ravel().tolist()}
            raise ValueError(f"unknown RemoteIo op {op}")

    def _update_command(self, sensors=None):
        key = (self.task_epoch, self.env.tick)
        if key != self._command_key:
            gyro, gravity, positions, velocities = sensors or self.env.physics.sensors()
            state = pack_observation(gyro, gravity, positions, velocities, self.env.last_action, np.zeros(13))
            self._command = np.asarray(prescribed_command(self.env, state), np.float32)
            self._command_key = key

    def begin_task(self):
        with self.lock:
            # Runtime bring-up is real motion. Record its resulting pose/history
            # explicitly rather than relabeling it as the offline seed's pose.
            self.handoff = {"body_tick": self.steps, "capture_monotonic_ns": time.monotonic_ns(),
                            "qpos": self.env.physics.data.qpos.tolist(), "qvel": self.env.physics.data.qvel.tolist(),
                            "targets": self.targets.tolist(), "startup_fallen": bool(self.env._ever_fallen),
                            "offline_initial_pose_equal": bool(np.array_equal(self.env.physics.data.qpos, self.initial["data"].qpos)
                                                               and np.array_equal(self.env.physics.data.qvel, self.initial["data"].qvel)),
                            "offline_initial_qpos_max_abs_difference": float(np.max(np.abs(self.env.physics.data.qpos - self.initial["data"].qpos))),
                            "offline_controller_history_equal": None, "startup_physics_paused_for_model": False}
            if hasattr(self.env, "_qualified_initialization_failed"):
                self.env._qualified_initialization_failed |= self.handoff["startup_fallen"]
            if hasattr(self.env, "_ground_initialization_failed"):
                self.env._ground_initialization_failed |= self.handoff["startup_fallen"]
            begin_execution_task(self.env)
            self.task_epoch += 1
            self.task_origin_body_tick = self.steps
            self._command_key = None
            self._update_command()

    def _physics_loop(self):
        deadline = time.monotonic()
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    physics = self.env.physics
                    if not self.released:
                        # Match the official twin's held-before-enable boundary.
                        physics.restore_state(self.initial)
                    elif self.torque_on:
                        physics.step(self.targets)
                    else:
                        physics.data.ctrl[physics.actuator_indices] = 0.0
                        for _ in range(4):
                            physics.mj.mj_step(physics.model, physics.data)
                    self.steps += 1
                    self.step_times_ns.append(time.monotonic_ns())
                    if self.released:
                        self.env.tick += 1
                        self.env._rgb_tick = -1
                        self._update_command()
                        self.env._record_metrics()
                        self.env._update_task()
                deadline += 0.02
                remaining = deadline - time.monotonic()
                if remaining < -0.02:
                    self.late_ticks += 1
                if remaining < -0.25:
                    deadline = time.monotonic()
                self.stop_event.wait(max(0.0, remaining))
        except Exception as exc:
            self.fault = repr(exc)
            self.stop_event.set()

    def _camera_loop(self):
        renderer = None
        try:
            physics = self.env.physics
            data = physics.mj.MjData(physics.model)
            renderer = physics.mj.Renderer(physics.model, height=physics.image_size, width=physics.image_size)
            while not self.stop_event.is_set():
                with self.lock:
                    physics.mj.mj_copyData(data, physics.model, physics.data)
                    tick = self.steps
                    captured = time.monotonic()
                renderer.update_scene(data, camera=physics.camera)
                frame = np.rot90(renderer.render(), k=-1).copy()
                with self.lock:
                    self.frame, self.frame_time, self.frame_tick = frame, captured, tick
                self.stop_event.wait(1.0 / 15)
        except Exception as exc:
            self.fault = f"camera: {exc!r}"
            self.stop_event.set()
        finally:
            if renderer is not None:
                renderer.close()

    def camera_snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy(), self.frame_tick, self.frame_time

    def status(self):
        elapsed = time.monotonic() - self.started
        from .execution_receipts import distribution
        boundary = (self.handoff or {}).get("capture_monotonic_ns", 0)
        with self.lock:
            step_times = [value for value in self.step_times_ns if value >= boundary]
        periods = np.diff(step_times).astype(np.float64) / 1e6
        return {
            "backend": "robotd_bam",
            "control_ticks": self.steps,
            "wall_time_s": elapsed,
            "achieved_hz": self.steps / max(elapsed, 1e-9),
            "late_ticks": self.late_ticks,
            "task_epoch": self.task_epoch, "task_origin_body_tick": self.task_origin_body_tick,
            "task_tick": self.env.tick, "handoff": self.handoff,
            "task_step_monotonic_ns": step_times, "task_control_period_ms": distribution(periods),
            "fault": self.fault,
            "torque_on": self.torque_on,
            "hardware_validated": False,
            "unmodeled": ["motor currents", "battery discharge", "servo temperature", "camera ISP"],
        }

    def close(self):
        self.stop_event.set()
        self.server.shutdown()
        self.server.server_close()
        for thread in self.threads:
            thread.join(timeout=3.0)


class RobotdBamBackend:
    """A real robotd process consuming a BAM body and externally supplied chunks.

    ``step`` submits work; it does not advance or stop physics. Observations
    carry daemon ticks, while camera metadata retains the separate body clock.
    """

    def __init__(self, asset_root=None, *, scene="room", render=True, binary=None, initial_conditions="nominal",
                 execution_mode=LEGACY_EXECUTION_MODE, body_factory=None):
        if execution_mode not in (LEGACY_EXECUTION_MODE, PREFIX_EXECUTION_MODE):
            raise ValueError("unknown external execution mode")
        self.execution_mode = execution_mode
        # Isolated instrumentation probes can supply a body implementation;
        # normal simulator and model-service execution keep the official body.
        self._body_factory = body_factory or BamBodyServer
        self.asset_root = resolve_asset_root(asset_root)
        self.scene, self.render_enabled = scene, render
        self.initial_conditions = initial_conditions
        v5_binary = self.asset_root.parent / "build/microduck-external-v5/target/debug/robotd"
        v4_binary = self.asset_root.parent / "build/microduck-external-v4/target/debug/robotd"
        v3_binary = self.asset_root.parent / "build/microduck-external-v3/target/debug/robotd"
        new_binary = self.asset_root.parent / "build/microduck-external-v2/target/debug/robotd"
        build = "microduck-external-v4" if v4_binary.is_file() else "microduck-external-v3" if v3_binary.is_file() else "microduck-external-v2" if is_project_profile(initial_conditions) or new_binary.is_file() else "microduck-external"
        if v5_binary.is_file() and not v4_binary.is_file():
            build = "microduck-external-v5"
        if execution_mode == PREFIX_EXECUTION_MODE:
            build = "microduck-external-v5"
        self.binary = Path(binary) if binary else self.asset_root.parent / f"build/{build}/target/debug/robotd"
        self.process = self.body = self.env = self._temporary = self._log = None
        self.clock_samples = []
        self.clock_offset = None
        self.execution_receipts = []
        self.receipt_cursor = self.missing_receipts = self.unrecorded_control_ticks = 0
        self.receipt_protocol = None

    def reset(self, seed=0, task="find_red_ball", options=None):
        self.close()
        self.execution_recipe = resolve_execution_recipe(self.initial_conditions, task)
        self.requires_source_capture = is_project_profile(self.execution_recipe)
        self.clock_samples, self.clock_offset = [], None
        self.execution_receipts = []
        self.receipt_cursor = self.missing_receipts = self.unrecorded_control_ticks = 0
        self.receipt_protocol = None
        if not self.binary.is_file():
            raise FileNotFoundError(f"build the pinned daemon first: microduck robotd build ({self.binary})")
        self._temporary = tempfile.TemporaryDirectory(prefix="lxduck-")
        directory = Path(self._temporary.name)
        self.env = make_execution_environment(task, recipe=self.execution_recipe, asset_root=self.asset_root,
                                              profile="official_bam", scene=self.scene, render=False)
        self.env.preset_clock = "program"
        self.env.reset(seed, task, options)
        self.episode_id, self.task = self.env.episode_id, self.env.task
        self.body = self._body_factory(self.env, cameras=self.render_enabled).start()
        policies = self.asset_root / "microduck-policies"
        slots = {
            "walk": "alpha_walking.onnx",
            "stand": "alpha_stand.onnx",
            "sitstand": "alpha_sitstand.onnx",
            "ground_pick": "alpha_ground_pick.onnx",
            "kick_left": "ball_kick_left.onnx",
            "kick_right": "ball_kick_right.onnx",
            "roulade": "roulade.onnx",
        }
        if self.task.embodiment == "rollers":
            slots["walk"], slots["ground_pick"] = "roller.onnx", "roller_crouch.onnx"
        config = "[policy]\nenabled = true\nmode = " + json.dumps("roller" if self.task.embodiment == "rollers" else "walk") + "\n"
        config += "".join(f"{key} = {json.dumps(str(policies / filename))}\n" for key, filename in slots.items())
        params = directory / "robotd.toml"
        params.write_text(config)
        self.control = ExternalPolicyClient(directory / "robotd.sock")
        self.client = ExternalPolicyClient(directory / "external.sock")
        variables = dict(os.environ, MICRODUCK_EXTERNAL_POLICY_SOCKET=self.client.socket_path)
        if "ORT_DYLIB_PATH" not in variables:
            import onnxruntime

            libraries = sorted((Path(onnxruntime.__file__).parent / "capi").glob("libonnxruntime.so*"))
            if not libraries:
                raise FileNotFoundError("set ORT_DYLIB_PATH to the ONNX Runtime shared library")
            variables["ORT_DYLIB_PATH"] = str(libraries[0])
        self._log = (directory / "robotd.log").open("w+")
        self.process = subprocess.Popen(
            [str(self.binary), "--sim", f"{self.body.address[0]}:{self.body.address[1]}", "--socket", self.control.socket_path, "--params", str(params)],
            env=variables,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            enabled = False
            while time.monotonic() < deadline:
                if self.process.poll() is not None or self.body.fault:
                    raise RuntimeError("robotd/BAM exited during startup")
                try:
                    if not enabled:
                        enabled_result = self.control.call("robot.enable", {"on": True})
                        if not enabled_result.get("accepted", False):
                            raise RuntimeError(f"enable refused: {enabled_result}")
                        enabled = True
                    snapshot = self.client.observe()
                    if len(snapshot.get("state", [])) == 61 and snapshot["observation_age_s"] < 0.1:
                        if self.requires_source_capture and (snapshot.get("source_capture_protocol") != 1 or not snapshot.get("source_capture")):
                            raise ValueError("qualified robotd requires source_capture protocol 1; build the independent v2 binary")
                        if snapshot.get("source_capture_protocol") == 1 and snapshot.get("external_control_permitted") is not True:
                            # A fresh observation can come from seated-boot rise
                            # or another local skill. Wait for actual ownership.
                            time.sleep(.02)
                            continue
                        if self.render_enabled:
                            frame, _, captured = self.body.camera_snapshot()
                            if frame is None or time.monotonic() - captured >= 0.2:
                                time.sleep(0.02)
                                continue
                        self.body.begin_task()
                        # Wait for a newly read sensor packet that explicitly
                        # names the new task epoch; do not relabel the old read.
                        if snapshot.get("source_capture_protocol") == 1:
                            while time.monotonic() < deadline:
                                snapshot = self.client.observe()
                                capture = snapshot.get("source_capture") or {}
                                if capture.get("task_epoch") == self.body.task_epoch:
                                    break
                                time.sleep(.002)
                            else:
                                raise TimeoutError("no sensor capture from the new task epoch")
                        opened = self.client.open(self.episode_id, execution_mode=self.execution_mode)
                        self.receipt_protocol = opened.get("execution_receipt_protocol")
                        self.receipt_cursor = max(0, opened.get("receipt_sequence", 0) - 1)
                        self.unrecorded_control_baseline = opened.get("unrecorded_control_ticks", snapshot.get("unrecorded_control_ticks", 0))
                        self.task_origin_tick = opened["tick"]
                        self.started = time.monotonic()
                        return self.observe()
                except (OSError, RuntimeError):
                    pass
                time.sleep(0.02)
            raise TimeoutError("robotd did not publish a fresh policy observation")
        except Exception as exc:
            self._log.flush()
            log = (directory / "robotd.log").read_text()[-6000:]
            self.close()
            raise RuntimeError(f"{exc}\n{log}") from exc

    def observe(self):
        snapshot = self.client.observe()
        if snapshot["episode_id"] != self.episode_id or snapshot["observation_age_s"] >= 0.2:
            raise RuntimeError(f"external control was revoked or stale: {snapshot['reason']}")
        rgb, camera_tick, capture_time = self.body.camera_snapshot()
        camera_age = time.monotonic() - capture_time if capture_time else None
        if self.render_enabled and (rgb is None or camera_age is None or camera_age >= 0.2):
            raise RuntimeError("BAM camera frame is absent or stale (200ms limit)")
        with self.body.lock:
            tof, tof_metadata = self.env.physics.read_tof(self.body.steps)
            metadata = self.env.physics.camera_metadata()
        metadata.update(body_capture_tick=camera_tick, capture_monotonic_s=capture_time, frame_age_s=camera_age, clock_domain="body", calibrated=False)
        state = np.asarray(snapshot["state"], dtype=np.float32).copy()
        capture = snapshot.get("source_capture")
        if capture is not None:
            clock = self._validate_capture(snapshot, capture)
            metadata.update(source_capture=capture, clock_mapping=clock, task_elapsed_tick=capture["task_tick"],
                            image_source_body_tick=camera_tick, image_and_proprioception_same_tick=camera_tick == capture["body_tick"])
            state[48:61] = np.asarray(capture["command13"], np.float32)
        elif self.requires_source_capture:
            raise ValueError("qualified robotd observation is missing its source sensor identity")
        elif self.task.suite == "author_presets":
            from .tasks import task_command
            state[48:61] = task_command(self.task, max(0, snapshot["tick"] - self.task_origin_tick))
            metadata["task_elapsed_tick"] = max(0, snapshot["tick"] - self.task_origin_tick)
        return Observation(
            self.episode_id, snapshot["tick"], snapshot["tick"] * 0.02, state, rgb, self.task.instruction, self.task.task_id, None,
            capture["capture_monotonic_ns"] if capture is not None else time.monotonic_ns(), tof, metadata, tof_metadata
        )

    def _validate_capture(self, snapshot, capture):
        if (capture.get("protocol") != 1 or capture.get("episode_id") != self.episode_id
                or capture.get("task_epoch") != self.body.task_epoch):
            raise ValueError("sensor capture belongs to another episode/task epoch or protocol")
        if any(isinstance(capture.get(key), bool) or not isinstance(capture.get(key), int) or capture[key] < 0
               for key in ("body_tick", "task_tick", "capture_monotonic_ns")):
            raise ValueError("invalid source capture clock")
        if capture["task_tick"] != capture["body_tick"] - self.body.task_origin_body_tick:
            raise ValueError("source task/body tick mapping changed within the episode")
        age_s = (time.monotonic_ns() - capture["capture_monotonic_ns"]) / 1e9
        if not 0 <= age_s < .2:
            raise ValueError("source proprioception is stale or its host clock is incompatible")
        command = np.asarray(capture.get("command13"), np.float32)
        if command.shape != (13,) or not np.isfinite(command).all():
            raise ValueError("source capture command must be finite command13")
        if self.requires_source_capture and capture.get("command_program_sha256") != self.body.command_program_sha256:
            raise ValueError("source capture used another task command program")
        offset = snapshot["tick"] - capture["body_tick"]
        if self.clock_offset is None:
            self.clock_offset = offset
        clock = {"daemon_tick": snapshot["tick"], "source_body_tick": capture["body_tick"], "source_task_tick": capture["task_tick"],
                 "body_tick_at_query": self.body.steps, "daemon_minus_body_ticks": offset,
                 "offset_drift_ticks": offset - self.clock_offset, "source_age_s": age_s,
                 "identity_source": "RemoteIo.source_capture", "same_tick_claimed": False}
        if not self.clock_samples or self.clock_samples[-1]["daemon_tick"] != snapshot["tick"]:
            self.clock_samples.append(clock)
        return clock

    @property
    def tick(self):
        return self.client.observe()["tick"]

    @property
    def terminated(self):
        return self.env.terminated

    def step(self, action):
        if not isinstance(action, JointResidualChunk):
            raise TypeError("external robotd mode accepts JointResidualChunk; local intents use the official API")
        self.client.submit(action)
        return self.observe()

    def drain_execution_receipts(self):
        if self.receipt_protocol != 1:
            return {"supported": False, "complete": False, "reason": "robotd lacks execution receipt protocol 1"}
        latest = self.receipt_cursor
        for _ in range(16):
            page = self.client.receipts(self.receipt_cursor)
            if page.get("protocol") != 1:
                raise ValueError("robotd receipt protocol changed")
            self.missing_receipts += page["missing_events"]
            self.unrecorded_control_ticks = max(self.unrecorded_control_ticks,
                                               page["unrecorded_control_ticks"] - self.unrecorded_control_baseline)
            self.execution_receipts.extend(page["events"])
            self.receipt_cursor, latest = page["next_sequence"], page["latest_sequence"]
            if self.receipt_cursor >= latest:
                break
        return {"supported": True, "complete": not self.missing_receipts and not self.unrecorded_control_ticks and self.receipt_cursor >= latest,
                "missing_events": self.missing_receipts, "unrecorded_control_ticks": self.unrecorded_control_ticks,
                "cursor": self.receipt_cursor, "latest_sequence": latest}

    def cancel(self):
        """Revoke the local external session while physical time keeps advancing."""
        return self.client.close()

    def render(self):
        frame, _, _ = self.body.camera_snapshot()
        return frame

    def evaluator_state(self):
        with self.body.lock:
            result = self.env.evaluator_state()
        external = self.client.observe()
        result.update(backend=self.body.status(), external=external, health=self.control.call("robot.health"),
                      execution_mode=self.execution_mode,
                      execution_recipe=self.execution_recipe, clock_mapping={"source_capture_required": self.requires_source_capture,
                      "samples": self.clock_samples, "maximum_offset_drift_ticks": max((abs(row["offset_drift_ticks"]) for row in self.clock_samples), default=None),
                      "daemon_and_body_are_distinct_clocks": True}, runtime_handoff=self.body.handoff)
        result["system_final_success"] = result["success"]
        result["success"] = bool(result["success"] and external["fallback_ticks"] == 0)
        return result

    def clone_state(self):
        raise RuntimeError("real-time robotd cannot clone its control thread; branch DAgger in official_bam")

    def restore_state(self, snapshot):
        raise RuntimeError("real-time robotd cannot rewind; start a new episode")

    def close(self):
        if self.process is not None:
            try:
                self.client.close()
                self.control.call("robot.stop")
            except (OSError, RuntimeError, ValueError):
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None
        if self.body is not None:
            self.body.close()
            self.body = None
        if self.env is not None:
            self.env.close()
            self.env = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
