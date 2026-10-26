"""Bounded language grammar and reproducible course distributions; no eval/codegen."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields

import numpy as np

REVISION = "school_bam_v1"
UPSTREAM = "b18978c3246a20b4476f6d6ce1f4940c7e4675a1"
COURSES = {
    "goto": ("到达标记", "走到前方 0.8 米的标记并停下"),
    "explore": ("探索", "向外探索 1 米"),
    "spin": ("原地转圈", "原地向左转 1 圈"),
    "circle": ("绕圈", "沿半径 0.5 米的圆向左绕 1 圈"),
    "ball": ("推球", "把球向前推 0.3 米"),
    "rgb": ("RGB 寻球", "找到红色球，靠近并停下"),
}


@dataclass(frozen=True)
class Course:
    family: str
    distance_m: float = 0.8
    radius_m: float = 0.5
    turns: float = 1.0
    direction: int = 1
    color: str = "red"
    waypoints: int = 1
    duration_s: float = 30.0

    def __post_init__(self):
        if self.family not in COURSES or self.color not in ("red", "blue", "yellow"):
            raise ValueError("unknown course or color")
        for key, low, high in (("distance_m", 0.1, 1.5), ("radius_m", 0.2, 1.0), ("turns", 0.5, 3.0), ("duration_s", 10.0, 60.0)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be in [{low}, {high}]")
        if type(self.direction) is not int or self.direction not in (-1, 1) or type(self.waypoints) is not int or self.waypoints not in (1, 2):
            raise ValueError("direction must be -1/1; waypoints must be 1/2")

    @property
    def privileged(self):
        return self.family != "rgb"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("invalid course fields")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("invalid course") from exc


def parse_lesson(text):
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 300:
        raise ValueError("请输入 1–300 字的课程目标")
    source = text.strip()
    # Decimal dots are significant; normalize punctuation without dropping them.
    phrase = re.sub(r"[\s，,。!！]", "", source.lower()).rstrip(".")
    number = r"([0-9]+(?:\.[0-9]+)?)"
    candidates = [
        (rf"(?:走到|到达)(?:前方)?{number}米的?标记(?:并停下)?", lambda m: Course("goto", distance_m=float(m[1]))),
        (r"依次到达两个标记并停下", lambda m: Course("goto", waypoints=2)),
        (rf"(?:向外)?探索{number}米", lambda m: Course("explore", distance_m=float(m[1]))),
        (rf"原地向(左|右)转{number}圈", lambda m: Course("spin", direction=1 if m[1] == "左" else -1, turns=float(m[2]))),
        (rf"沿半径{number}米的圆向(左|右)绕{number}圈", lambda m: Course("circle", radius_m=float(m[1]), direction=1 if m[2] == "左" else -1, turns=float(m[3]), duration_s=45)),
        (rf"(?:把)?球向前推{number}米", lambda m: Course("ball", distance_m=float(m[1]))),
        (r"(?:找到|寻找)(红|蓝|黄)色?球(?:并|然后)?(?:靠近|接近)(?:并)?停下", lambda m: Course("rgb", color={"红": "red", "蓝": "blue", "黄": "yellow"}[m[1]], duration_s=40)),
        (rf"goto{number}m(?:andstop)?", lambda m: Course("goto", distance_m=float(m[1]))),
        (rf"explore{number}m", lambda m: Course("explore", distance_m=float(m[1]))),
        (rf"spin(left|right){number}turns?", lambda m: Course("spin", direction=1 if m[1] == "left" else -1, turns=float(m[2]))),
        (rf"circle(left|right)radius{number}mfor{number}turns?", lambda m: Course("circle", radius_m=float(m[2]), direction=1 if m[1] == "left" else -1, turns=float(m[3]), duration_s=45)),
        (rf"push(?:the)?ball{number}m", lambda m: Course("ball", distance_m=float(m[1]))),
        (r"find(?:the)?(red|blue|yellow)ball(?:and)?(?:approachand)?stop", lambda m: Course("rgb", color=m[1], duration_s=40)),
    ]
    for pattern, factory in candidates:
        match = re.fullmatch(pattern, phrase)
        if match:
            course = factory(match)
            return {
                "instruction": source,
                "parser": "registered_grammar_v1",
                "course": course.to_dict(),
                "sensors": ["proprioception", "simulator_truth"] if course.privileged else ["rgb", "proprioception"],
                "label": COURSES[course.family][0],
            }
    raise ValueError("不支持该目标。请选择课程示例再修改数字或颜色；首版不支持自由生成新任务、避障、跳跃或组合指令。")


def split_seeds(seed, split, count):
    if type(seed) is not int or not 0 <= seed < 1000000 or split not in ("train", "validation", "test", "demo") or not 1 <= count <= 100:
        raise ValueError("invalid split/seed/count")
    base = seed * 1000 + ("train", "validation", "test", "demo").index(split) * 100
    return list(range(base, base + count))


def scenario(course, seed):
    rng = np.random.default_rng(int(seed))
    origin = rng.uniform(-0.25, 0.25, 2)
    yaw = float(rng.uniform(-np.pi, np.pi))
    angle = yaw + float(rng.uniform(-0.65, 0.65))
    distance = course.distance_m * float(rng.uniform(0.8, 1.2))
    waypoints = [origin + distance * np.array([np.cos(angle), np.sin(angle)])]
    if course.waypoints == 2:
        waypoints.append(waypoints[0] + 0.5 * np.array([np.cos(angle + 1.2), np.sin(angle + 1.2)]))
    center = origin + course.radius_m * np.array([-np.sin(yaw), np.cos(yaw)]) * course.direction
    balls = {}
    for k, color in enumerate(("red", "blue", "yellow")):
        # Randomized target direction, including outside the initial camera view.
        a = yaw + float(rng.uniform(-np.pi, np.pi))
        d = float(rng.uniform(0.50, 0.85))
        balls[color] = [*(origin + d * np.array([np.cos(a), np.sin(a)])), 0.035]
    # Reject overlapping balls deterministically.
    for k, color in enumerate(balls):
        if any(np.linalg.norm(np.array(balls[color][:2]) - balls[c][:2]) < 0.18 for c in list(balls)[:k]):
            a = yaw + k * 2 * np.pi / 3
            balls[color] = [*(origin + 0.8 * np.array([np.cos(a), np.sin(a)])), 0.035]
    if course.family == "ball":
        balls["red"] = [*(origin + 0.23 * np.array([np.cos(yaw), np.sin(yaw)])), 0.035]
        balls["blue"], balls["yellow"] = [4.0, 4.0, 0.035], [-4.0, 4.0, 0.035]
    elif course.family != "rgb":
        balls = {c: [4.0 + k, 4.0, 0.035] for k, c in enumerate(balls)}
    result = {"seed": int(seed), "origin": origin.tolist(), "yaw": yaw, "waypoints": [p.tolist() for p in waypoints], "center": center.tolist(), "balls": balls}
    result["layout_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def catalog():
    return {
        "revision": REVISION,
        "courses": [{"id": key, "label": value[0], "example": value[1], "privileged": key != "rgb"} for key, value in COURSES.items()],
        "default_budget_s": 600,
        "max_budget_s": 1800,
        "parser": "registered_grammar_v1",
    }
