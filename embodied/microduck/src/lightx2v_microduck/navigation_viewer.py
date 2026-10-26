"""Lossy observer rendering on an independent MuJoCo snapshot and GL thread.

The viewer never steps physics, runs forward dynamics, changes policy inputs or
waits for a browser. Busy viewers drop display frames; the control loop continues.
"""
from __future__ import annotations

import copy
import io
import queue
import threading
import time
from pathlib import Path

from .visualization import atomic_json


class NavigationViewer:
    def __init__(self, output, *, fps=5, size=448):
        self.output = Path(output)
        self.interval, self.size = 1.0 / fps, size
        self.free, self.requests = queue.Queue(1), queue.Queue(1)
        self.ready, self.stopped = threading.Event(), threading.Event()
        self.next_frame = 0.
        self.thread = None
        self.error = None
        self.frames, self.dropped = 0, 0
        self.last_camera = None
        self.last_camera_tick = None
        self.render_device = None

    def start(self, env):
        self.physics = env.physics
        self.model = copy.copy(self.physics.model)
        # Observer-only quality/size: the policy's camera/model is unchanged.
        self.model.vis.global_.offwidth = max(self.size, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(self.size, self.model.vis.global_.offheight)
        self.model.vis.quality.offsamples = 0
        self.data = self.physics.mj.MjData(self.model)
        self.physics.mj.mj_copyData(self.data, self.model, self.physics.data)
        self.thread = threading.Thread(target=self._run, name="microduck-live-display", daemon=True)
        self.thread.start()
        if not self.ready.wait(30):
            raise TimeoutError("live observer renderer initialization timed out")
        if self.error:
            raise RuntimeError(self.error)
        self.free.put_nowait(True)

    def begin(self):
        atomic_json(self.output / "stage.json", {"status": "running", "message": "正在运行新的 MuJoCo 仿真。"})

    def cancelled(self):
        return (self.output / "cancel.request").exists()

    def camera(self, observation):
        if observation.rgb is not None:
            self.last_camera = observation.rgb
            self.last_camera_tick = observation.camera_tick

    def publish(self, env, *, wall_s, execution, latency_ms=None, fallback_ticks=0):
        now = time.monotonic()
        if now < self.next_frame:
            return
        self.next_frame = now + self.interval
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.free.get_nowait()
        except queue.Empty:
            self.dropped += 1
            return
        self.physics.mj.mj_copyData(self.data, self.model, self.physics.data)
        packet = dict(tick=env.tick, sim_s=env.tick * .02, wall_s=wall_s,
                      real_time_factor=env.tick * .02 / max(wall_s, 1e-9),
                      camera_tick=self.last_camera_tick, execution=dict(execution),
                      latency_ms=latency_ms, fallback_ticks=fallback_ticks,
                      captured_unix=time.time(), source="live_mujoco_snapshot",
                      physics_waits_for_model=False, observer_is_policy_input=False)
        self.requests.put_nowait((packet, self.last_camera))

    def _jpeg(self, name, rgb):
        from PIL import Image
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="JPEG", quality=86)
        temporary = self.output / (name + ".jpg.tmp")
        temporary.write_bytes(stream.getvalue())
        temporary.replace(self.output / (name + ".jpg"))

    def _run(self):
        renderer = None
        try:
            mj = self.physics.mj
            renderer = mj.Renderer(self.model, height=self.size, width=self.size)
            camera = mj.MjvCamera()
            mj.mjv_defaultCamera(camera)
            camera.type = mj.mjtCamera.mjCAMERA_FREE
            camera.distance, camera.azimuth, camera.elevation = 1.5, 135, -28
            renderer.update_scene(self.data, camera=camera)
            renderer.render()
            from OpenGL import GL
            self.render_device = GL.glGetString(GL.GL_RENDERER).decode("utf-8", errors="replace")
            self.ready.set()
            while not self.stopped.is_set():
                try:
                    packet, head = self.requests.get(timeout=.1)
                except queue.Empty:
                    continue
                camera.lookat[:] = self.data.xpos[self.physics.trunk]
                camera.lookat[2] = max(.1, camera.lookat[2]) + .05
                renderer.update_scene(self.data, camera=camera)
                self._jpeg("observer", renderer.render())
                if head is not None:
                    self._jpeg("camera", head)
                self.frames += 1
                packet.update(sequence=self.frames, published_unix=time.time(), display_dropped_frames=self.dropped)
                atomic_json(self.output / "live.json", packet)
                self.free.put_nowait(True)
        except Exception as exc:
            self.error = f"live renderer: {type(exc).__name__}: {exc}"
        finally:
            self.ready.set()
            if renderer is not None:
                renderer.close()

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=5)

    def metrics(self):
        return dict(frames=self.frames, dropped_frames=self.dropped, error=self.error,
                    worker_stopped=self.thread is None or not self.thread.is_alive(),
                    observer_source="mj_copyData_without_forward", observer_size=self.size,
                    head_camera_source="original_policy_camera", display_is_policy_input=False,
                    render_device=self.render_device)
