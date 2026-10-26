"""Measure initial target visibility for evaluation scenario coverage.

Evaluator-only segmentation is compared with an isolated rendering of the same
sphere, camera and physical pose. Missing target pixels alone cannot distinguish
occlusion from an out-of-view target. This module never supplies policy inputs,
advances physics, chooses test layouts or certifies navigation success.
"""

import hashlib

import numpy as np


def classify_visibility(visible, isolated, *, outside_planes=(), clipped_planes=(),
                        unresolved_blocked_pixels=0):
    visible, isolated = np.asarray(visible), np.asarray(isolated)
    if (visible.dtype != np.bool_ or isolated.dtype != np.bool_
            or visible.ndim != 2 or visible.shape != isolated.shape):
        raise ValueError('visibility evidence requires equally sized boolean masks')
    if np.any(visible & ~isolated):
        raise ValueError('actual target pixels differ from the isolated target projection')
    observed, projected = int(visible.sum()), int(isolated.sum())
    blocked = projected - observed
    if type(unresolved_blocked_pixels) is not int or not 0 <= unresolved_blocked_pixels <= blocked:
        raise ValueError('invalid unresolved occlusion pixel count')
    if outside_planes:
        if projected:
            raise ValueError('rendered target contradicts the camera frustum evidence')
        classification = 'outside_camera_frustum'
    elif not projected:
        classification = 'unresolved_no_rasterized_target'
    elif unresolved_blocked_pixels:
        classification = 'unresolved_render_difference'
    elif observed == projected:
        classification = 'unoccluded'
    elif not observed:
        classification = 'fully_occluded'
    else:
        classification = 'partially_occluded'
    return dict(classification=classification, visible_pixels=observed,
        isolated_projected_pixels=projected, occluded_pixels=blocked,
        visible_fraction=None if not projected else observed/projected,
        outside_frustum_planes=list(outside_planes), clipped_frustum_planes=list(clipped_planes),
        unresolved_blocked_pixels=unresolved_blocked_pixels,
        # Raster visibility does not establish recognition by the learned model.
        model_recognition_verified=False)


def sphere_frustum(local_xyz, radius, *, fovy_degrees, aspect, near, far):
    """Sphere/plane test in the MuJoCo camera's right/up/backward coordinates."""
    position = np.asarray(local_xyz, dtype=float)
    parameters = np.asarray([radius, fovy_degrees, aspect, near, far], dtype=float)
    if (position.shape != (3,) or not np.isfinite(position).all()
            or not np.isfinite(parameters).all() or radius <= 0 or aspect <= 0
            or not 0 < fovy_degrees < 180 or not 0 < near < far):
        raise ValueError('invalid sphere or perspective camera')
    x, y, z = position
    depth = -z
    ty = np.tan(np.deg2rad(fovy_degrees)/2)
    tx = ty*aspect
    distances = dict(left=(x+depth*tx)/np.sqrt(1+tx*tx),
        right=(-x+depth*tx)/np.sqrt(1+tx*tx),
        bottom=(y+depth*ty)/np.sqrt(1+ty*ty),
        top=(-y+depth*ty)/np.sqrt(1+ty*ty), near=depth-near, far=far-depth)
    return dict(outside_planes=[k for k,v in distances.items() if v < -radius],
        clipped_planes=[k for k,v in distances.items() if v < radius],
        plane_signed_distances_m={k:float(v) for k,v in distances.items()},
        sphere_camera_xyz_m=position.tolist(), sphere_radius_m=float(radius))


def measure_initial_targets(env, expected_rgb):
    """Inspect an independently reconstructed tick-zero pose, without rollout.

    The caller must bind the recorded pose, assets and source version. Rendering
    groups change only inside a separate renderer and are restored even on error.
    No result is returned unless the original head RGB matches before and after.
    """
    if type(env.tick) is not int or env.tick != 0 or env.evaluator_state()['evaluation_elapsed_s'] != 0:
        raise ValueError('scenario coverage must be measured at the initial task pose')
    result = measure_targets_at_pose(env.physics, expected_rgb, recorded_tick=0)
    result.update(schema='microduck_initial_visibility_measurement_v1', task_tick=0)
    return result


class TargetVisibilityProbe:
    """Reuse a private diagnostic rendering context for one physical model.

    One probe belongs to one replay/thread; close it before resetting that
    environment. It never reuses the robot's observation renderer.
    """

    def __init__(self, physics):
        self.physics = physics
        self.model = physics.model
        self.image_size = physics.image_size
        self.renderer = physics.mj.Renderer(self.model, height=self.image_size, width=self.image_size)
        self.renderer.enable_segmentation_rendering()

    def measure(self, expected_rgb, *, recorded_tick, colors=('red', 'blue', 'yellow')):
        return measure_targets_at_pose(self.physics, expected_rgb, recorded_tick=recorded_tick,
                                       colors=colors, _probe=self)

    def measure_observation(self, env, observation, *, colors=('red', 'blue', 'yellow')):
        """Inspect the already-captured policy frame without recapturing RGB.

        The caller must compare this observation with its recording. Every
        future policy frame still requires the same strict replay comparison.
        Additional RGB captures can differ slightly from the original capture
        and must not replace the observation's pixels.
        """
        if (env.physics is not self.physics or observation.episode_id != env.episode_id
                or observation.tick != env.tick or observation.camera_tick != env.tick
                or env._rgb_tick != env.tick):
            raise ValueError('visibility requires the current captured policy observation')
        if (not isinstance(observation.rgb, np.ndarray) or observation.rgb.dtype != np.uint8
                or observation.rgb.shape != (self.image_size, self.image_size, 3)):
            raise ValueError('visibility requires a captured RGB image')
        return measure_targets_at_pose(self.physics, observation.rgb,
            recorded_tick=env.tick, colors=colors, _probe=self, _captured_env=env)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def measure_targets_at_pose(physics, expected_rgb, *, recorded_tick, colors=('red', 'blue', 'yellow'),
                            _probe=None, _captured_env=None):
    """Inspect a recorded camera pose, preserving the caller's physical state.

    Matching RGB binds the supplied pose to the recording. It does not certify
    how that pose was reached; trajectory replay remains separate evidence.
    The recorded tick is explicit and cannot be interpreted as task tick zero.
    """
    if type(recorded_tick) is not int or recorded_tick < 0:
        raise ValueError('recorded camera tick must be a nonnegative integer')
    if (not isinstance(colors, (tuple, list)) or not colors or len(set(colors)) != len(colors)
            or any(c not in ('red', 'blue', 'yellow') for c in colors)):
        raise ValueError('select one or more unique target colors')
    if _probe is not None and (_probe.physics is not physics or _probe.renderer is None
            or _probe.model is not physics.model or _probe.image_size != physics.image_size):
        raise ValueError('diagnostic renderer is closed or belongs to another model/camera size')
    mj, model, data, camera = physics.mj, physics.model, physics.data, physics.camera
    if getattr(model, 'cam_orthographic', np.zeros(model.ncam))[camera]:
        raise ValueError('perspective camera required')
    # This path uses the current MicroDuck fovy camera, not calibrated intrinsics.
    if hasattr(model, 'cam_sensorsize') and np.any(model.cam_sensorsize[camera]):
        raise ValueError('intrinsic-calibrated camera requires a separate projection implementation')
    expected_rgb = np.asarray(expected_rgb)
    if _captured_env is not None:
        if (_captured_env.physics is not physics or _captured_env.tick != recorded_tick
                or _captured_env._rgb_tick != recorded_tick):
            raise ValueError('captured observation does not match the current physical tick')
        rgb_at_pose = lambda: _captured_env._rgb_cache.copy()
    else:
        rgb_at_pose = physics.render
    np.testing.assert_array_equal(rgb_at_pose(), expected_rgb)
    qpos, qvel, groups, clock = data.qpos.copy(), data.qvel.copy(), model.geom_group.copy(), float(data.time)
    camera_position, camera_rotation = data.cam_xpos.copy(), data.cam_xmat.copy()
    geom_position, geom_rotation = data.geom_xpos.copy(), data.geom_xmat.copy()
    renderer = (mj.Renderer(model, height=physics.image_size, width=physics.image_size)
                if _probe is None else _probe.renderer)
    if _probe is None:
        renderer.enable_segmentation_rendering()
    results = {}
    geom_type = int(mj.mjtObj.mjOBJ_GEOM)
    try:
        renderer.update_scene(data, camera=camera)
        original = renderer.render().copy()
        for color in colors:
            geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'lx_ball_'+color+'_geom')
            if geom < 0 or model.geom_type[geom] != mj.mjtGeom.mjGEOM_SPHERE:
                raise ValueError('target sphere is absent')
            radius = float(model.geom_size[geom, 0])
            rotation = data.cam_xmat[camera].reshape(3,3)
            position = rotation.T @ (data.geom_xpos[geom]-data.cam_xpos[camera])
            projection = sphere_frustum(position, radius, fovy_degrees=float(model.cam_fovy[camera]),
                aspect=1., near=float(model.vis.map.znear*model.stat.extent),
                far=float(model.vis.map.zfar*model.stat.extent))
            option = mj.MjvOption()
            option.geomgroup[:] = 0
            option.geomgroup[0] = 1
            try:
                model.geom_group[:] = 5
                model.geom_group[geom] = 0
                renderer.update_scene(data, camera=camera, scene_option=option)
                isolated_segmentation = renderer.render().copy()
            finally:
                model.geom_group[:] = groups
            visible = (original[:,:,0] == geom) & (original[:,:,1] == geom_type)
            isolated = ((isolated_segmentation[:,:,0] == geom)
                        & (isolated_segmentation[:,:,1] == geom_type))
            blocked = isolated & ~visible
            unresolved = int((blocked & ((original[:,:,1] != geom_type) | (original[:,:,0] < 0))).sum())
            classification = classify_visibility(visible, isolated,
                outside_planes=projection['outside_planes'], clipped_planes=projection['clipped_planes'],
                unresolved_blocked_pixels=unresolved)
            blockers = {}
            for other in np.unique(original[:,:,0][blocked & (original[:,:,1] == geom_type)]):
                if other >= 0:
                    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(other)) or 'unnamed_'+str(other)
                    blockers[name] = int((blocked & (original[:,:,0] == other)
                                         & (original[:,:,1] == geom_type)).sum())
            results[color] = dict(classification, projection=projection, blocking_geometries=blockers,
                visible_mask_sha256=hashlib.sha256(visible.tobytes()).hexdigest(),
                isolated_mask_sha256=hashlib.sha256(isolated.tobytes()).hexdigest())
    finally:
        model.geom_group[:] = groups
        if _probe is None:
            renderer.close()
    np.testing.assert_array_equal(model.geom_group, groups)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.cam_xpos, camera_position)
    np.testing.assert_array_equal(data.cam_xmat, camera_rotation)
    np.testing.assert_array_equal(data.geom_xpos, geom_position)
    np.testing.assert_array_equal(data.geom_xmat, geom_rotation)
    if data.time != clock:
        raise ValueError('visibility inspection advanced simulation')
    np.testing.assert_array_equal(rgb_at_pose(), expected_rgb)
    return dict(schema='microduck_recorded_pose_visibility_measurement_v1', recorded_camera_tick=recorded_tick,
        head_rgb_sha256=hashlib.sha256(expected_rgb.tobytes()).hexdigest(), targets=results,
        original_rgb_exact_before_and_after=True, physical_state_unchanged=True,
        rgb_binding='captured_policy_observation' if _captured_env is not None else 'repeated_physics_render',
        evaluator_only=True, model_inference=False, formal_acceptance=False)
