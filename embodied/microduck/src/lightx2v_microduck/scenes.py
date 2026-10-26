"""Scene composition around an unchanged, revision-pinned MicroDuck MJCF."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .assets import resolve_asset_root
from .contracts import TaskSpec

DEFAULT_ASSET_ROOT = resolve_asset_root()
COLORS = {"red": (0.90, 0.05, 0.05, 1.0), "blue": (0.05, 0.15, 0.95, 1.0), "yellow": (0.95, 0.80, 0.05, 1.0)}
# Training/development curricula keep the full generator's target distribution.
# They never alias the final four-obstacle randomized-room benchmark.
CURRICULUM_ROOMS = {"room_curriculum_zero": 0, "room_curriculum_one": 1, "room_curriculum_two": 2}


@dataclass(frozen=True)
class Obstacle:
    name: str
    xy: tuple[float, float]
    half_size: tuple[float, float]
    height: float = 0.20


@dataclass(frozen=True)
class SceneDescription:
    name: str
    source_xml: str
    embodiment: str
    obstacles: tuple[Obstacle, ...] = ()
    room_bounds: tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5)
    target_colors: tuple[str, ...] = ()
    robot_prefix: str = ""
    robot_source_xml: str = ""
    original_demo_root: str | None = None
    layout_id: str = ""
    geometry_hash: str = ""
    texture_id: str = "official_groundplane_v1"
    ball_layout: tuple[tuple[str, float, float, float], ...] = ()

    def as_dict(self):
        return asdict(self)


def source_directory(asset_root: Path | str | None, profile: str) -> Path:
    root = resolve_asset_root(asset_root)
    # Accept an explicit upstream RL checkout as well as the fetched catalog.
    if (root / "src/mjlab_microduck/robot/microduck").is_dir():
        return root
    name = "tinyvla-author-rl" if profile == "tinyvla_author_compat" else "microduck_rl"
    path = root / name
    if not (path / "src/mjlab_microduck/robot/microduck").is_dir():
        raise FileNotFoundError(f"missing pinned {name} snapshot at {path}; run microduck assets fetch first")
    return path


def build_scene(asset_root, profile: str, scene: str | Path, task: TaskSpec, *, obstacle_layout: int = 0):
    """Return MjSpec and task-level metadata; no dynamics approximation is made."""
    import mujoco

    upstream = source_directory(asset_root, profile)
    robot_dir = upstream / "src/mjlab_microduck/robot/microduck"
    if task.embodiment == "rollers":
        filename = "scene_rollers.xml"
    elif task.task_id in ("kick_left", "kick_right"):
        filename = "scene_ball.xml"
    else:
        filename = "scene.xml"
    scene = "official" if str(scene) == "flat" else scene
    requested = Path(scene)
    builtins = ("official", "room", "room_clear", "empty", "room_obstacles", "room_randomized", "water_court", "ball_apex", "community_follow", "slalom_cones", "ball_tracking", "duck_detection", "climb_desk", *CURRICULUM_ROOMS)
    if str(scene) == "water_court":
        source = resolve_asset_root(asset_root) / "microduck-ai-world/simulation/src/mjlab_microduck/robot/microduck/scene_aaa.xml"
        if not source.is_file():
            raise FileNotFoundError("Water Court assets missing; fetch microduck-ai-world first")
    elif str(scene) not in builtins:
        if not requested.is_file():
            raise FileNotFoundError(f"scene {scene!r} is not a built-in or existing MJCF")
        source = requested.resolve()
    else:
        source = robot_dir / filename
    spec = mujoco.MjSpec.from_file(str(source))
    robot_prefix = ""
    canonical_scene = robot_dir / filename
    robot_file = next(i.attrib["file"] for i in ET.parse(canonical_scene).getroot().findall("include") if i.attrib["file"].startswith("robot_"))
    canonical_robot = robot_dir / robot_file
    original_demo = None
    if source != canonical_scene:
        # A community world is a scene extension, never a replacement robot.
        # MjSpec.delete removes attached joints/actuators/sensors too. Prefixing
        # the official body and its assets prevents old mesh/default collisions.
        community_robots = [b for b in spec.bodies if b.name == "trunk_base"]
        for body in community_robots:
            spec.delete(body)
        child = mujoco.MjSpec.from_file(str(canonical_robot))
        robot_prefix = "lx_official_"
        spec.attach(child, prefix=robot_prefix, frame=spec.worldbody.add_frame())
        if task.task_id in ("kick_left", "kick_right"):
            for body in [body for body in spec.bodies if body.name == "ball"]:
                spec.delete(body)
            spec.attach(mujoco.MjSpec.from_file(str(robot_dir / "ball.xml")), prefix="lx_kick_", frame=spec.worldbody.add_frame())
            # MjSpec uses a slash namespace for an empty prefix; retain the
            # official ball/freejoint bindings used by reset and event scoring.
            for body in spec.bodies:
                if body.name == "lx_kick_ball":
                    body.name = "ball"
                    for joint in body.joints:
                        joint.name = "ball_free"
                    for geom in body.geoms:
                        geom.name = "ball_geom"
        if str(scene) == "water_court":
            original_demo = str(resolve_asset_root(asset_root) / "microduck-ai-world")
    # Added free joints change nq. Reset is applied by joint name after compile,
    # so source keyframes are not resized or accidentally replayed positionally.
    for key in list(spec.keys):
        spec.delete(key)
    obstacles = ()
    ball_layout = ()
    if str(scene) == 'climb_desk':
        if profile != 'official_bam':
            raise ValueError('climb_desk requires the official_bam profile')
        from .climb import add_scene
        add_scene(spec, asset_root)
    if str(scene) == "slalom_cones":
        from .tricks import CONES, SPACING
        obstacles = tuple(Obstacle(f"cone_{k}", ((k + 1.5) * SPACING, 0.), (.03, .03), .06) for k in range(CONES))
        for cone in obstacles:
            spec.worldbody.add_geom(name=f"lx_{cone.name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    size=(.03, .03, 0.), pos=(*cone.xy, .03), rgba=(1., .45, 0., 1.),
                                    friction=(.6, .005, .0001))
    if str(scene) in ("room", "room_clear", "room_obstacles", "room_randomized", *CURRICULUM_ROOMS):
        for name, pos, size in (
            ("north", (0.0, 1.5, 0.15), (1.5, 0.025, 0.15)),
            ("south", (0.0, -1.5, 0.15), (1.5, 0.025, 0.15)),
            ("east", (1.5, 0.0, 0.15), (0.025, 1.5, 0.15)),
            ("west", (-1.5, 0.0, 0.15), (0.025, 1.5, 0.15)),
        ):
            spec.worldbody.add_geom(name=f"lx_room_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=(0.55, 0.59, 0.64, 1.0))
        if str(scene) == "room_randomized" or str(scene) in CURRICULUM_ROOMS:
            obstacles, ball_layout = randomized_room_layout(obstacle_layout)
            if str(scene) in CURRICULUM_ROOMS:
                # Select a deterministic prefix after generating all four
                # boxes and targets. This preserves targets/color assignment
                # across curriculum levels and the original full scene.
                obstacles = obstacles[:CURRICULUM_ROOMS[str(scene)]]
        elif str(scene) != "room_clear" and (task.suite == "visual_goal" or str(scene) == "room_obstacles"):
            sign = -1 if obstacle_layout % 2 else 1
            obstacles = (
                Obstacle("box_a", (0.50, 0.05 * sign), (0.095, 0.16)),
                Obstacle("box_b", (0.86, -0.65 * sign), (0.10, 0.10)),
            )
        for obstacle in obstacles:
            spec.worldbody.add_geom(
                name=f"lx_{obstacle.name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(*obstacle.xy, obstacle.height / 2), size=(*obstacle.half_size, obstacle.height / 2), rgba=(0.35, 0.35, 0.38, 1.0)
            )
    targets = tuple(COLORS) if task.suite == "visual_goal" else ()
    if str(scene) == "ball_tracking":
        targets = ("red", "red_distractor", "blue")
        ball_layout = (("red", .85, .25, .035), ("red_distractor", .85, -.25, .035), ("blue", 1.1, .0, .035))
    if str(scene) == "duck_detection":
        # A fixed-root official duck target, not an autonomous second robot.
        child = mujoco.MjSpec.from_file(str(canonical_robot))
        root = next(b for b in child.bodies if b.name == "trunk_base")
        for joint in list(root.joints):
            child.delete(joint)
        root.pos = (0., 0., 0.)
        spec.attach(child, prefix="target_", frame=spec.worldbody.add_frame(pos=(.75, 0., .125), quat=(0., 0., 0., 1.)))
    for color in targets:
        body = spec.worldbody.add_body(name=f"lx_ball_{color}", pos=(0.8, 0.2, 0.035))
        body.add_freejoint(name=f"lx_ball_{color}_free")
        body.add_geom(name=f"lx_ball_{color}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.035, 0, 0), mass=0.015, rgba=COLORS[color.split('_')[0]], friction=(0.8, 0.01, 0.001), condim=3)
    if task.suite == "community_basketball":
        body = spec.worldbody.add_body(name="ball", pos=(0.0, 0.0, 0.12))
        body.add_freejoint(name="ball_free")
        collision = body.add_geom(
            name="ball_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.12, 0.0, 0.0), mass=0.62, friction=(1.2, 0.01, 0.001), priority=1, condim=4, rgba=(0.91, 0.42, 0.12, 1.0)
        )
        body.add_site(name="ball_center", size=(0.004, 0.004, 0.004), group=3)
        visual_root = resolve_asset_root(asset_root).parent / "community-policies/basketball"
        if all((visual_root / filename).is_file() for filename in ("basketball.obj", "basketball.png")):
            visual_lock = json.loads((Path(__file__).parent / "resources/community-policies.json").read_text())["policies"]["basketball"]["files"]
            for filename in ("basketball.obj", "basketball.png"):
                if hashlib.sha256((visual_root / filename).read_bytes()).hexdigest() != visual_lock[filename]["sha256"]:
                    raise ValueError(f"basketball visual asset checksum mismatch: {filename}")
            texture = spec.add_texture(name="lx_basketball_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(visual_root / "basketball.png"))
            material = spec.add_material(name="lx_basketball_mat", specular=0.15, shininess=0.15)
            textures = list(material.textures)
            textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
            material.textures = textures
            mesh = spec.add_mesh(name="lx_basketball_mesh", file=str(visual_root / "basketball.obj"))
            body.add_geom(name="ball_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, material=material.name, contype=0, conaffinity=0, mass=0.0)
            collision.rgba, collision.group = (1.0, 1.0, 1.0, 0.0), 3
    if task.suite == "community_follow":
        # Explicit stationary mocap target for the first follow-v4 oracle test;
        # its physics and occlusion are real, its detector remains privileged.
        body = spec.worldbody.add_body(name="follow_person", mocap=True, pos=(1.4, 0.4, 0.8))
        body.add_geom(name="follow_person_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(0.20, 0.6, 0.0), rgba=(0.2, 0.6, 0.3, 1.0))
    geometry = {"scene": str(scene), "obstacles": [asdict(o) for o in obstacles], "balls": ball_layout}
    if str(scene) == 'climb_desk':
        from .community import load_catalog
        geometry['climb_asset_sha256'] = load_catalog()['climb-scene']['generated_files']['compiled_scene.json']['sha256']
    geometry_hash = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    layout_id = hashlib.sha256(json.dumps({"geometry": geometry_hash, "seed": obstacle_layout}, sort_keys=True).encode()).hexdigest()
    return spec, SceneDescription(
        str(scene),
        str(source),
        task.embodiment,
        obstacles,
        target_colors=targets,
        robot_prefix=robot_prefix,
        robot_source_xml=str(canonical_robot),
        original_demo_root=original_demo,
        layout_id=layout_id,
        geometry_hash=geometry_hash,
        texture_id="community_original" if original_demo else "official_groundplane_v1",
        ball_layout=ball_layout,
    )


def randomized_room_layout(seed: int):
    """Separated obstacle islands leave at least .28 m between boxes.

    The spawn has .30 m clearance and walls have .30 m clearance. Targets
    include one behind the initial camera and one geometrically occluded by a
    box when possible. This is a seeded layout generator, not success evidence.
    """
    rng = np.random.default_rng(int(seed))
    obstacles = []
    for _ in range(500):
        if len(obstacles) >= 4:
            break
        xy, half = rng.uniform(-1.0, 1.0, 2), rng.uniform(0.08, 0.15, 2)
        if np.linalg.norm(np.maximum(np.abs(xy) - half, 0.0)) < 0.30:
            continue
        if any(np.linalg.norm(np.maximum(np.abs(xy - o.xy) - half - o.half_size, 0.0)) < 0.28 for o in obstacles):
            continue
        obstacles.append(Obstacle(f"random_{len(obstacles)}", tuple(xy), tuple(half), float(rng.uniform(0.16, 0.28))))
    points = []

    def valid(xy):
        return (
            np.all(np.abs(xy) < 1.25)
            and np.linalg.norm(xy) > 0.50
            and all(np.linalg.norm(np.maximum(np.abs(xy - o.xy) - o.half_size, 0.0)) > 0.16 for o in obstacles)
            and all(np.linalg.norm(xy - old) > 0.25 for old in points)
        )

    for _ in range(100):
        point = np.array((-rng.uniform(0.55, 1.2), rng.uniform(-1.2, 1.2)))
        if valid(point):
            points.append(point)
            break
    candidates = []
    for obstacle in obstacles:
        direction = np.asarray(obstacle.xy) / np.linalg.norm(obstacle.xy)
        extent = min(np.asarray(obstacle.half_size) / np.maximum(np.abs(direction), 1e-8))
        candidates.append(np.asarray(obstacle.xy) + direction * (extent + 0.23))
    candidates.extend(rng.uniform(-1.24, 1.24, (500, 2)))
    for point in candidates:
        if valid(point):
            points.append(point)
        if len(points) == 3:
            break
    if len(points) != 3:
        raise RuntimeError("failed to generate three separated reachable ball positions")
    colors = rng.permutation(tuple(COLORS))
    return tuple(obstacles), tuple((str(color), float(point[0]), float(point[1]), 0.035) for color, point in zip(colors, points))


def sample_ball_positions(description: SceneDescription, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Distinct visible objects; color assignment and geometry vary with seed."""
    if description.ball_layout:
        return {color: np.asarray((x, y, z), dtype=np.float64) for color, x, y, z in description.ball_layout if color in description.target_colors}
    anchors = np.asarray(((1.05, 0.55), (0.9, -0.40), (0.45, 0.85)), dtype=np.float64)
    anchors = anchors[rng.permutation(len(anchors))]
    positions = {}
    for color, anchor in zip(description.target_colors, anchors):
        xy = anchor + rng.uniform(-0.07, 0.07, 2)
        positions[color] = np.asarray((*xy, 0.035), dtype=np.float64)
    return positions
