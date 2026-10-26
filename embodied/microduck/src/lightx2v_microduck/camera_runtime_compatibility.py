"""Verify the reviewed camera API additions without accepting evaluator changes.

The fixed replacements below describe the camera configuration migration only.
They are not generated from candidate source at runtime. Every other byte must
match the pinned development environment, including task and physics logic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def verify_camera_api_patch(name, development, runtime):
    """Accept exact original bytes or the one reviewed camera-only migration."""
    if development == runtime:
        return 'unchanged'
    if name not in CAMERA_API_PATCHES:
        raise ValueError('unreviewed runtime source migration: ' + name)
    transformed = development
    for before, after in CAMERA_API_PATCHES[name]:
        if transformed.count(before) != 1:
            raise ValueError('development source does not match camera API migration: ' + name)
        transformed = transformed.replace(before, after, 1)
    if transformed != runtime:
        raise ValueError('runtime has changes beyond the reviewed camera API: ' + name)
    return 'exact_camera_api_additions'


def verify_camera_runtime_compatibility(development, runtime, *, development_root):
    """Read pinned source bytes; do not trust a declaration of equivalence."""
    prefix = 'embodied/microduck/src/lightx2v_microduck/'
    roots = [Path(development_root), Path(runtime['source'])]
    if any(not root.is_absolute() for root in roots):
        raise ValueError('source compatibility requires absolute roots')
    rows = {}
    for name in ('env.py', 'physics.py', 'tasks.py', 'contracts.py',
                 'scenes.py', 'policies.py', 'assets.py'):
        contents, digests = [], []
        for root, manifest in zip(roots, (development, runtime)):
            data = (root / (prefix + name)).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != manifest.get('files', {}).get(prefix + name):
                raise ValueError('source bytes differ from pinned manifest: ' + name)
            contents.append(data.decode('utf-8'))
            digests.append(digest)
        kind = verify_camera_api_patch(name, *contents)
        rows[name] = dict(development_sha256=digests[0], runtime_sha256=digests[1], kind=kind)
    return dict(schema='microduck_camera_api_compatibility_v1', status='verified',
        default_camera_only=True, files=rows,
        development_evaluator_sha256=rows['env.py']['development_sha256'],
        runtime_evaluator_sha256=rows['env.py']['runtime_sha256'],
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        basis='Exact reviewed camera-only additions; all other bytes and task/physics dependencies unchanged.')

# Reviewed delta from the pinned pre-camera runtime to the camera-aware runtime.
CAMERA_API_PATCHES = {'env.py': [('from .contracts import HOME_POLICY, JointResidualChunk, Observation, SkillIntent, TaskSpec, '
             'pack_observation\n'
             'from .physics import PROFILES, Physics\n'
             'from .policies import PolicyBank\n'
             'from .scenes import CURRICULUM_ROOMS, build_scene, sample_ball_positions\n',
             'from .contracts import HOME_POLICY, JointResidualChunk, Observation, SkillIntent, TaskSpec, '
             'pack_observation\n'
             'from .physics import PROFILES, Physics\n'
             'from .rendering import validate_camera_offsamples\n'
             'from .policies import PolicyBank\n'
             'from .scenes import CURRICULUM_ROOMS, build_scene, sample_ball_positions\n'),
            ('\n'
             'class MicroDuckEnv:\n'
             '    def __init__(self, profile="official_bam", scene="room", asset_root=None, render=True, *, '
             'image_size=256, action_delay_ticks=None, daemon_backend=None):\n'
             '        if profile not in PROFILES:\n'
             '            raise ValueError(f"unknown profile {profile!r}")\n',
             '\n'
             'class MicroDuckEnv:\n'
             '    def __init__(self, profile="official_bam", scene="room", asset_root=None, render=True, *, '
             'image_size=256, action_delay_ticks=None, daemon_backend=None, camera_offsamples=None):\n'
             '        if profile not in PROFILES:\n'
             '            raise ValueError(f"unknown profile {profile!r}")\n'),
            ('            raise ValueError("official_daemon requires an actual robotd/duck-body '
             'daemon_backend; start duck-sim and use the daemon client")\n'
             '        self.profile, self.scene = profile, str(scene)\n'
             '        self.asset_root = resolve_asset_root(asset_root)\n'
             '        self.render_enabled = bool(render)\n',
             '            raise ValueError("official_daemon requires an actual robotd/duck-body '
             'daemon_backend; start duck-sim and use the daemon client")\n'
             '        self.profile, self.scene = profile, str(scene)\n'
             '        self.camera_offsamples = validate_camera_offsamples(camera_offsamples)\n'
             '        if daemon_backend is not None and camera_offsamples is not None:\n'
             '            raise ValueError("camera_offsamples applies only to the local MuJoCo renderer")\n'
             '        self.asset_root = resolve_asset_root(asset_root)\n'
             '        self.render_enabled = bool(render)\n'),
            ('        spec, self.description = build_scene(self.asset_root, self.profile, self.scene, '
             'self.task, obstacle_layout=self.seed if seeded_room else self.seed % 2)\n'
             '        tof_source = self.asset_root / "microduck_rl/src/mjlab_microduck/sim/tof.py"\n'
             '        self.physics = Physics(spec, profile=self.profile, image_size=self.image_size, '
             'robot_prefix=self.description.robot_prefix, tof_source=tof_source if tof_source.is_file() else '
             'None)\n'
             '        ball_positions = sample_ball_positions(self.description, self.rng)\n'
             '        program = PROGRAMS.get(self.task.task_id)\n',
             '        spec, self.description = build_scene(self.asset_root, self.profile, self.scene, '
             'self.task, obstacle_layout=self.seed if seeded_room else self.seed % 2)\n'
             '        tof_source = self.asset_root / "microduck_rl/src/mjlab_microduck/sim/tof.py"\n'
             '        self.physics = Physics(spec, profile=self.profile, image_size=self.image_size, '
             'robot_prefix=self.description.robot_prefix, tof_source=tof_source if tof_source.is_file() else '
             'None, camera_offsamples=self.camera_offsamples)\n'
             '        ball_positions = sample_ball_positions(self.description, self.rng)\n'
             '        program = PROGRAMS.get(self.task.task_id)\n'),
            ('    def sim_time_s(self):\n'
             '        return self.tick * 0.02 if self.physics is None else float(self.physics.data.time)\n'
             '\n'
             '    @property\n',
             '    def sim_time_s(self):\n'
             '        return self.tick * 0.02 if self.physics is None else float(self.physics.data.time)\n'
             '\n'
             '    def camera_configuration(self):\n'
             '        if self.daemon_backend is not None:\n'
             '            return None\n'
             '        if self.physics is None:\n'
             '            raise RuntimeError("call reset before reading the camera configuration")\n'
             '        return self.physics.camera_configuration()\n'
             '\n'
             '    @property\n'),
            ('            "signature": self._signature,\n'
             '            "action_delay_ticks": self.action_delay_ticks,\n'
             '            "physics": self.physics.clone_state(),\n'
             '            "env": {name: copy.deepcopy(getattr(self, name)) for name in names},\n',
             '            "signature": self._signature,\n'
             '            "action_delay_ticks": self.action_delay_ticks,\n'
             '            "camera_offsamples": self.camera_offsamples,\n'
             '            "physics": self.physics.clone_state(),\n'
             '            "env": {name: copy.deepcopy(getattr(self, name)) for name in names},\n'),
            ('        if snapshot["profile"] != self.profile or snapshot["scene"] != self.scene or '
             'snapshot["action_delay_ticks"] != self.action_delay_ticks:\n'
             '            raise ValueError("snapshot profile/scene/delay differs from destination '
             'environment")\n'
             '        if self.physics is None or self._signature != snapshot["signature"]:\n'
             '            state = snapshot["env"]\n',
             '        if snapshot["profile"] != self.profile or snapshot["scene"] != self.scene or '
             'snapshot["action_delay_ticks"] != self.action_delay_ticks:\n'
             '            raise ValueError("snapshot profile/scene/delay differs from destination '
             'environment")\n'
             '        if snapshot.get("camera_offsamples") != self.camera_offsamples:\n'
             '            raise ValueError("snapshot camera configuration differs from destination '
             'environment")\n'
             '        if self.physics is None or self._signature != snapshot["signature"]:\n'
             '            state = snapshot["env"]\n')],
 'physics.py': [('import importlib.util\nimport json\nfrom dataclasses import dataclass\n\n',
                 'import importlib.util\nimport json\nimport os\nfrom dataclasses import dataclass\n\n'),
                ('\n'
                 'from .contracts import HOME_POLICY, JOINT_NAMES, PHYSICS_DT_S, POLICY_JOINT_NAMES\n'
                 '\n'
                 'BAM_REVISION = "62bd8ce12154340be97e06f7f41a0ca8f116d967"\n',
                 '\n'
                 'from .contracts import HOME_POLICY, JOINT_NAMES, PHYSICS_DT_S, POLICY_JOINT_NAMES\n'
                 'from .rendering import validate_camera_offsamples\n'
                 '\n'
                 'BAM_REVISION = "62bd8ce12154340be97e06f7f41a0ca8f116d967"\n'),
                ('\n'
                 'class Physics:\n'
                 '    def __init__(self, spec, *, profile="official_bam", robot_prefix="", image_size=256, '
                 'tof_source=None):\n'
                 '        import mujoco\n'
                 '\n',
                 '\n'
                 'class Physics:\n'
                 '    def __init__(self, spec, *, profile="official_bam", robot_prefix="", image_size=256, '
                 'tof_source=None, camera_offsamples=None):\n'
                 '        import mujoco\n'
                 '\n'),
                ('        self.robot_prefix = robot_prefix\n'
                 '        self.image_size = int(image_size)\n'
                 '        self.renderer = None\n'
                 '        self.observer_renderer = None\n',
                 '        self.robot_prefix = robot_prefix\n'
                 '        self.image_size = int(image_size)\n'
                 '        self.camera_offsamples = validate_camera_offsamples(camera_offsamples)\n'
                 '        self.renderer = None\n'
                 '        self.observer_renderer = None\n'),
                ('                joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)\n'
                 '        self.model = spec.compile()\n'
                 '        self.model.opt.timestep = PHYSICS_DT_S\n'
                 '        self.data = mujoco.MjData(self.model)\n',
                 '                joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)\n'
                 '        self.model = spec.compile()\n'
                 '        if self.camera_offsamples is not None:\n'
                 '            self.model.vis.quality.offsamples = self.camera_offsamples\n'
                 '        self.model.opt.timestep = PHYSICS_DT_S\n'
                 '        self.data = mujoco.MjData(self.model)\n'),
                ('        if self.renderer is None:\n'
                 '            self.renderer = self.mj.Renderer(self.model, height=self.image_size, '
                 'width=self.image_size)\n'
                 '        return self.renderer\n'
                 '\n'
                 '    def render(self):\n',
                 '        if self.renderer is None:\n'
                 '            self.renderer = self.mj.Renderer(self.model, height=self.image_size, '
                 'width=self.image_size)\n'
                 '            if (self.camera_offsamples is not None\n'
                 '                    and self.renderer._mjr_context.offSamples != self.camera_offsamples):\n'
                 '                self.renderer.close()\n'
                 '                self.renderer = None\n'
                 '                raise RuntimeError("renderer did not apply the requested '
                 'camera_offsamples")\n'
                 '        return self.renderer\n'
                 '\n'
                 '    def camera_configuration(self):\n'
                 '        """Report the actual context after capture; do not initialize a renderer."""\n'
                 '        return dict(schema="microduck_camera_rendering_v1", '
                 'requested_offsamples=self.camera_offsamples,\n'
                 '                    model_offsamples=int(self.model.vis.quality.offsamples),\n'
                 '                    context_offsamples=(None if self.renderer is None else '
                 'int(self.renderer._mjr_context.offSamples)),\n'
                 '                    image_size=self.image_size, backend=os.environ.get("MUJOCO_GL", '
                 '"auto"),\n'
                 '                    mujoco_version=self.mj.__version__, orientation=self.profile.camera)\n'
                 '\n'
                 '    def render(self):\n')]}
