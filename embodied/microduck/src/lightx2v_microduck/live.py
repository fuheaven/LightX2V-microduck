"""Wall-clock MuJoCo viewing, with atomic latest-frame delivery to the UI.

Only the simulation thread accesses MuJoCo/OpenGL. Slow TinyVLA inference uses
immutable sensor snapshots on a worker and the existing bounded action buffer.
The browser never controls the simulation clock or blocks the physics loop.
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .contracts import JointResidualChunk
from .evaluation import validate_action_source
from .execution import ActionBuffer


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def image_uri(rgb):
    if rgb is None:
        return None
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class LiveController:
    def __init__(self, env, output, *, async_inference=False, publish_every=2):
        self.env = env
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.async_inference = async_inference
        self.publish_every = max(1, int(publish_every))
        self.stopped = threading.Event()
        self.requests = queue.Queue(maxsize=1)
        self.replies = queue.Queue(maxsize=1)
        self.buffer = ActionBuffer()
        self.max_lag_s = 0.0
        self.overrun_ticks = 0
        self.sequence = 0
        self.last_packet = {}
        self.inference_ms = []
        self.error = None
        self.worker = None
        self.last_step_wall_s = 0.0
        self.last_requested_tick = -1
        self.startup_inference_ms = None

    @staticmethod
    def _latest(target, value):
        try:
            target.put_nowait(value)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(value)

    def cancelled(self):
        return (self.output / "cancel.request").exists()

    def start(self, policy, observation):
        self.policy = policy
        self.origin_tick = observation.tick
        self.buffer.reset(observation.episode_id)
        # Shader setup happens before the wall-clock episode starts.
        self.env.render_observer()
        manifest = getattr(policy, "manifest", None)
        if self.async_inference and manifest is not None and manifest.metadata.get("engine") == "lightx2v-native-tinyvla":
            # Compile/load caches before the episode's clock is started. Then
            # reset the model session and obtain one fresh initial action. An
            # already-running world is never paused by this initialization.
            policy.predict(observation)
            policy.reset(observation.episode_id)
            self.env._rgb_tick = -1
            fresh = self.env.observe()
            before = time.monotonic()
            initial = policy.predict(fresh)
            self.startup_inference_ms = (time.monotonic() - before) * 1000
            validate_action_source(initial, fresh)
            if (time.monotonic_ns() - fresh.monotonic_ns) / 1e9 <= .2:
                self.buffer.submit(initial, current_tick=fresh.tick)
                self.last_requested_tick = fresh.tick
            else:
                self.buffer.rejected += 1
                self.buffer.rejection_reason = "stale_startup_action"
                policy.reset(observation.episode_id)
        self.started = time.monotonic()
        if self.async_inference:
            self.worker = threading.Thread(target=self._infer, name="microduck-live-policy", daemon=True)
            self.worker.start()
        self.publish(observation, "running")

    def _infer(self):
        while not self.stopped.is_set():
            try:
                observation = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            before = time.monotonic()
            try:
                action = self.policy.predict(observation)
                if not isinstance(action, JointResidualChunk):
                    raise TypeError("asynchronous policy must return joint residual chunks")
                validate_action_source(action, observation)
                result = (action, observation.monotonic_ns, (time.monotonic() - before) * 1000, None)
            except Exception as exc:
                result = (None, observation.monotonic_ns, (time.monotonic() - before) * 1000, f"{type(exc).__name__}: {exc}")
            if not self.stopped.is_set():
                self._latest(self.replies, result)

    def predict(self, observation):
        if not self.async_inference:
            return self.policy.predict(observation)
        if observation.tick % 5 == 0 and observation.tick > self.last_requested_tick:
            self._latest(self.requests, observation)
            self.last_requested_tick = observation.tick
        try:
            action, captured_ns, elapsed_ms, error = self.replies.get_nowait()
        except queue.Empty:
            pass
        else:
            self.inference_ms.append(elapsed_ms)
            if error:
                self.error = error
            elif (time.monotonic_ns() - captured_ns) / 1e9 > 0.2:
                self.buffer.rejected += 1
                self.buffer.rejection_reason = "stale_wall_observation"
            else:
                self.buffer.submit(action, current_tick=observation.tick)
        decision = self.buffer.consume(observation.tick)
        raw = decision.action
        if raw is None:
            state = observation.state.copy()
            state[48:] = 0
            raw = self.env.bank.infer("roller" if self.env.task.embodiment == "rollers" else "stand", state)
        return JointResidualChunk(
            raw[None], observation.episode_id, observation.tick, observation.sim_time_s,
            diagnostics={"execution_source": decision.source, "reason": decision.reason,
                         "original_source_tick": decision.source_tick, "action_index": decision.action_index,
                         "inference_error": self.error},
        )

    def after_step(self, observation):
        deadline = self.started + (observation.tick - self.origin_tick) * 0.02
        while not self.cancelled():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.01))
        lag = max(0.0, time.monotonic() - deadline)
        self.last_step_wall_s = time.monotonic() - self.started
        self.max_lag_s = max(self.max_lag_s, lag)
        self.overrun_ticks += int(lag > 0.02)
        if observation.tick % self.publish_every == 0 or self.env.terminated:
            self.publish(observation, "running")

    def publish(self, observation, status):
        elapsed = max(self.last_step_wall_s if status in ("complete", "cancelled") else time.monotonic() - self.started, 1e-9)
        sim_s = (observation.tick - self.origin_tick) * 0.02
        self.sequence += 1
        diagnostics = getattr(self.policy, 'last_diagnostics', {})
        perception = diagnostics.get('perception', {})
        overlay = None
        if observation.rgb is not None and perception:
            frame = Image.fromarray(getattr(self.policy, 'last_perception_rgb', observation.rgb))
            draw = ImageDraw.Draw(frame)
            draw.text((4, 4), f"Detection frame tick {perception.get('frame_tick', '?')}", fill='white')
            for track in perception.get('tracks', []):
                if track.get('age_ticks', 0) == 0:
                    draw.rectangle(track['bbox'], outline='lime', width=2)
                    draw.text(tuple(track['bbox'][:2]), f"ID {track['track_id']}", fill='lime')
            overlay = image_uri(np.asarray(frame))
        self.last_packet = {
            "sequence": self.sequence, "status": status, "tick": observation.tick, "sim_time_s": sim_s,
            "wall_time_s": elapsed, "real_time_factor": sim_s / elapsed,
            "max_lag_s": self.max_lag_s, "overrun_ticks": self.overrun_ticks,
            "task": self.env.task.task_id, "scene": self.env.scene, "seed": self.env.seed,
            "policy": self.policy.manifest.policy_id,
            "frames": {"camera": image_uri(observation.rgb), "observer": image_uri(self.env.render_observer()), "perception": overlay},
            "diagnostics": diagnostics,
            "execution": self.buffer.status() if self.async_inference else {"source": "selected_policy"},
            "startup_inference_ms": self.startup_inference_ms,
        }
        atomic_json(self.output / "live.json", self.last_packet)

    def finish(self, report):
        self.stopped.set()
        report["mode"] = "live"
        report["live_execution"] = {
            "target_control_hz": 50, "publish_hz": 50 / self.publish_every,
            "control_wall_time_s": self.last_step_wall_s,
            "max_lag_s": self.max_lag_s, "overrun_ticks": self.overrun_ticks,
            "async_inference": self.async_inference,
            "execution": self.buffer.status() if self.async_inference else None,
            "model_inference_ms": self.inference_ms, "model_error": self.error,
            "startup_inference_ms": self.startup_inference_ms,
            "physics_waits_for_model": False,
        }
        if self.async_inference:
            report["executor_latency_ms"] = report["inference_ms"]
            report["inference_ms"] = {
                "count": len(self.inference_ms),
                "p50": float(np.percentile(self.inference_ms, 50)) if self.inference_ms else None,
                "p95": float(np.percentile(self.inference_ms, 95)) if self.inference_ms else None,
            }
        if self.async_inference and (self.buffer.fallback_ticks or self.error):
            report.update(success=False, primary_success=False if report["primary_success"] is not None else None)
            report["termination_reason"] = "intervention" if report["termination_reason"] == "success" else report["termination_reason"]
        status = "cancelled" if report["termination_reason"] == "cancelled_by_user" else "complete"
        self.publish(self.env.observe(), status)
        self.last_packet.update(success=report["success"], termination_reason=report["termination_reason"])
        atomic_json(self.output / "live.json", self.last_packet)

    def close(self):
        self.stopped.set()
        if self.worker:
            self.worker.join(timeout=2.0)
