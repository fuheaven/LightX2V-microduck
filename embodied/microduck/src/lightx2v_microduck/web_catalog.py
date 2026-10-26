"""Curated experiment choices shared by the launcher and HTTP validation.

These are supported execution combinations, not claims of learned capability.
The CLI remains available for experiments outside the curated web workflows.
"""

from __future__ import annotations

from pathlib import Path

from .tasks import PROGRAMS, TASKS

VISUAL_TASKS = [name for name, task in TASKS.items() if task.suite == "visual_goal" and name.startswith("find_")]
VISUAL_POLICIES = ["lightx2v_tinyvla", "pixel_rgb", "pixel", "privileged"]
TASK_LABELS = dict(
    walk_forward="向前走并停下", circle_left="向左转圈", figure_eight="走八字", zigzag="之字行走",
    sway_circle="摇摆后转圈", kick_left="左脚踢球", kick_right="右脚踢球", ground_touch="鸭嘴触地",
    sit_walk="坐下、起身、行走", roll_circle="翻滚后转圈", roller_glide="滚轮滑行",
    roller_crouch="滚轮蹲伏滑行", roller_slalom="滚轮蛇形滑行", find_red_ball="寻找红球",
    find_blue_ball="寻找蓝球", find_yellow_ball="寻找黄球", community_follow="接近并跟随人形目标",
    ball_apex="在球上保持平衡", ground_lower="降低身体并恢复 · 独立动作",
    run_roll_recover="助跑、前滚翻、恢复站立", cone_slalom="滚轮穿越八个路桩",
    track_red_ball="记住最初左侧的红球并接近", detect_duck="检测另一只鸭子 · 固定目标",
    climb_desk="爬上桌面并恢复站立",
)
SCENES = dict(flat="平地", room="标准房间", room_clear="无障碍房间", room_randomized="随机布局房间",
              water_court="Water Court 庭院", community_follow="人形目标场景", ball_apex="球上起始场景", slalom_cones="八桩跑道 · 有碰撞", ball_tracking="同色球干扰场景", duck_detection="官方鸭子视觉目标", climb_desk="社区 27 级梯子与 0.66m 桌面")
POLICIES = {
    "community:official-velstand": {"label": "官方 VelStand · 9 月新版", "description": "官方新增行走与站立一体策略。独立固定版本作对照；不替换 TinyVLA 原教师与验收基准。"},
    "community:duckbatch-128": {"label": "Duckbatch · 26K 参数学生", "description": "社区从官方 VelStand 蒸馏的小型 ONNX；本地 BAM 迁移实验，不是 TinyVLA，也不是自主导航。"},
    "community:climb-pair": {"label": "Climber + Get-up · 双策略", "description": "爬梯与恢复两份 ONNX，保留 raw action 历史；接触桌面后切换滤波与增益。切换条件使用真值。"},
    "community:skill-planner": {"label": "VLM 技能规划 · quackd 思路", "planner": True,
                                "context_modes": ["feedback", "current_frame"],
                                "description": "外部 VLM 看图选择技能，本地官方 ONNX 执行；需要已启动的模型服务。没有默认模型，不用脚本冒充 VLM。"},
    "community:rgb-tracking": {"label": "RGB 检测 + 时序跟踪", "description": "独立实现的图像跟踪器；选择最初左侧红球，丢失身份后不读取真值补回。"},
    "community:duck-detector": {"label": "官方 duck-detector v2", "description": "真实 RGB → ONNX 检测框与跟踪 ID。固定鸭子目标；检测能力单独评分，不是导航。"},
    "community:tricks": {"label": "microduck-tricks · 社区权重", "description": "已训练的翻滚/滚轮 ONNX。当前测试 BAM 迁移；绕桩使用真值路径控制器。"},
    "community:tricks-official": {"label": "对照：官方运动权重", "description": "同一任务、时序与路径控制器，换用官方 ONNX。"},
    "official": {"label": "官方 ONNX + 动作程序", "description": "执行预设命令时序；恢复控制会读取仿真真值，并计入干预。"},
    "official_project": {"label": "官方 ONNX · 已验收教师", "description": "执行本项目固定版本的动作程序，读取关节与 IMU 状态；教师成绩与 TinyVLA 学生成绩分开。"},
    "lightx2v_tinyvla": {"label": "TinyVLA · LightX2V", "description": "视觉、语言和状态 → 关节动作。后端已接入，行为能力需用训练模型验证。", "model": True},
    "pixel_rgb": {"label": "对照：RGB 规则策略", "description": "颜色检测与搜索状态机，调用官方行走策略；不使用 VLA/VLM。"},
    "pixel": {"label": "对照：RGB + ToF 规则策略", "description": "颜色检测、测距与避障状态机；使用额外 ToF，单列辅助结果。"},
    "privileged": {"label": "教师：仿真真值规划", "description": "读取目标位置和环境真值，用于示范与教师对照，不计为纯视觉自主能力。"},
    "community:follow-v4": {"label": "follow-v4 · 社区跟随策略", "description": "ONNX 跟随策略；当前适配使用几何检测与真值里程计，不是 VLM。"},
    "community:rough-walk-g": {"label": "rough-walk-g · 社区行走策略", "description": "社区 ONNX 运动策略；这里测试平地行走，不代表已完成复杂地形能力复现。"},
    "community:basketball": {"label": "basketball · LSTM 平衡策略", "description": "有状态的关节控制策略，从球上初始化；不是语言导航或 VLM。"},
}
CATEGORIES = [
    {"id": "presets", "label": "官方策略与预设动作", "description": "行走、姿态、踢球与滚轮动作"},
    {"id": "visual", "label": "视觉目标任务", "description": "寻球、避障与停止实验；尚无 TinyVLA 自主能力验收"},
    {"id": "community", "label": "社区实验", "description": "跟随、球上平衡、行走策略与 AI World 场景"},
]
WORKFLOWS = {
    "velstand": {"category": "community", "label": "官方新版 · 行走与站立", "tasks": ["walk_forward", "circle_left", "figure_eight", "zigzag"],
                 "scenes": ["flat"], "policies": ["community:official-velstand", "community:duckbatch-128", "official"],
                 "description": "官方 9 月 VelStand 与社区小型学生的独立对照。此处不改变已验收 TinyVLA 的教师、任务或成绩。"},
    "duckbatch": {"category": "community", "label": "Duckbatch · 轻量策略对照", "tasks": ["walk_forward", "circle_left", "figure_eight", "zigzag"],
                  "scenes": ["flat"], "policies": ["community:duckbatch-128", "community:official-velstand"],
                  "description": "26,254 参数学生与 197,774 参数教师。这里使用本平台 BAM；作者原平台的成绩不作为本地能力验收。"},
    "project_tasks": {
        "category": "presets", "label": "项目训练动作 · 12 项 + 近地动作",
        "tasks": [name for name in PROGRAMS if name != "ground_touch"] + ["ground_lower"],
        "scenes": ["flat"], "policies": ["official_project", "lightx2v_tinyvla"], "realtime": True,
        "acceptance_scope": "project_tasks_v2", "initial_conditions": "qualified_initial_v2",
        "description": "当前训练使用的固定动作版本。近地动作 ground_lower 单独验收，不代表鸭嘴触地。教师已通过检查，TinyVLA 学生是否达标以登记模型成绩为准。",
    },
    "climb_desk": {"category": "community", "label": "Desk climb · 爬梯与桌面恢复", "tasks": ['climb_desk'],
                   "scenes": ['climb_desk'], "policies": ['community:climb-pair'],
                   "description": "固定作者 27 级梯子几何和双策略，运行 60 秒。当前为 BAM 固定延迟移植实验；失败后继续恢复，不传送复位。"},
    "skill_planner": {"category": "community", "label": "VLM 大脑 · 技能规划对照", "tasks": VISUAL_TASKS,
                      "scenes": ["room", "room_clear", "water_court"], "policies": ["community:skill-planner", "pixel_rgb"],
                      "description": "图像和指令 → VLM → 技能 → 官方 RL。物体运动和任务得分都来自 MuJoCo；实时模式不会等待远程模型。"},
    "rgb_tracking": {"category": "community", "label": "像素跟踪 · 同色球干扰", "tasks": ["track_red_ball"],
                     "scenes": ["ball_tracking"], "policies": ["community:rgb-tracking"],
                     "description": "两个红球中记住最初左侧目标。用真实像素关联身份，再调用官方行走策略。"},
    "duck_detector": {"category": "community", "label": "官方视觉检测 · 识别鸭子", "tasks": ["detect_duck"],
                      "scenes": ["duck_detection"], "policies": ["community:duck-detector"],
                      "description": "官方完整本体作为固定目标。检测器只读 RGB，评测器用可见掩码检查检测框。"},
    "run_roll": {"category": "community", "label": "microduck-tricks · 助跑翻滚", "tasks": ["run_roll_recover"],
                 "scenes": ["flat"], "policies": ["community:tricks", "community:tricks-official"],
                 "description": "站稳、助跑、翻滚 2 秒、恢复 2.5 秒。检查真实入场速度、翻转角度和恢复姿态。"},
    "cone_slalom": {"category": "community", "label": "microduck-tricks · 滚轮绕桩", "tasks": ["cone_slalom"],
                    "scenes": ["slalom_cones"], "policies": ["community:tricks", "community:tricks-official"],
                    "description": "八个物理路桩，间距 0.7 米。真值控制器给出航向，检查穿桩顺序、碰撞和跌倒；属于运动控制对照。"},
    "presets": {
        "realtime": True,
        "category": "presets", "label": "预设动作", "tasks": list(PROGRAMS),
        "scenes": ["flat"], "policies": ["official", "lightx2v_tinyvla"],
        "description": "默认由官方运动策略执行。13 项任务的编排来自 TinyVLA 作者，选择任务不代表模型已学会该行为。",
    },
    "visual": {
        "category": "visual", "label": "视觉寻球", "tasks": VISUAL_TASKS,
        "scenes": ["room", "room_clear", "room_randomized", "water_court"], "policies": VISUAL_POLICIES,
        "realtime": True,
        "description": "寻找颜色目标、绕开障碍、接近并停下。TinyVLA 接口已接入，此类自主能力尚未验收；可先体验规则策略或教师对照。",
    },
    "follow": {
        "category": "community", "label": "microduck-lab · 跟随", "tasks": ["community_follow"],
        "scenes": ["community_follow"], "policies": ["community:follow-v4"],
        "description": "使用社区 follow-v4 策略和专用人形目标场景。当前跟随输入含仿真真值。",
    },
    "rough": {
        "category": "community", "label": "rough-walk · 行走对照", "tasks": ["walk_forward", "circle_left", "figure_eight", "zigzag"],
        "scenes": ["flat"], "policies": ["community:rough-walk-g"],
        "description": "在相同平地任务上检查社区行走策略；暂未提供其复杂地形训练场景。",
    },
    "basketball": {
        "category": "community", "label": "basketball · 球上平衡", "tasks": ["ball_apex"],
        "scenes": ["ball_apex"], "policies": ["community:basketball"],
        "description": "鸭子从球上开始，测试 LSTM 策略保持平衡的能力。",
    },
    "ai_world": {
        "category": "community", "label": "AI World · Water Court 场景", "tasks": VISUAL_TASKS,
        "scenes": ["water_court"], "policies": ["pixel", "pixel_rgb", "lightx2v_tinyvla", "privileged"],
        "realtime": True,
        "description": "已接入 Water Court 场景，原项目的 VLM 大脑尚未接入。当前用规则策略或 TinyVLA 执行寻球任务。",
    },
}


WORKFLOWS = {name: WORKFLOWS[name] for name in (
    'presets', 'project_tasks', 'visual', 'run_roll', 'cone_slalom', 'duck_detector', 'rgb_tracking',
    'skill_planner', 'velstand', 'duckbatch', 'climb_desk', 'follow', 'rough', 'basketball', 'ai_world')}


def experiment_catalog():
    from .model_registry import catalog_models
    from .ground_lower import TASK as ground_lower
    tasks = {**TASKS, "ground_lower": ground_lower}
    return {
        "models": catalog_models(),
        "categories": CATEGORIES, "workflows": WORKFLOWS, "scenes": SCENES, "policies": POLICIES,
        "tasks": {name: {"label": TASK_LABELS[name], "instruction": task.instruction,
                         "embodiment": task.embodiment} for name, task in tasks.items()},
    }


def validate_launch(payload):
    """Reject incompatible combinations before a subprocess or model is started."""
    if not isinstance(payload, dict):
        raise ValueError("expected an experiment object")
    result = dict(payload)
    result.pop("acceptance_scope", None)
    registered = result.get("model_id")
    if registered and registered != "local":
        if result.get("policy") != "lightx2v_tinyvla":
            raise ValueError("model version is only supported by TinyVLA")
        from .model_registry import resolve
        result.update(resolve(registered, result.get("task")))
        result.pop("model_path", None)
        result.pop("backbone_path", None)
    else:
        registered = None
    mode = result.get("mode", "offline")
    policy = result.get("policy")
    if mode not in ("offline", "live", "robotd_bam"):
        raise ValueError("mode must be offline, live or robotd_bam")
    if mode == "robotd_bam" and (policy != "lightx2v_tinyvla" or (not registered and (not result.get("model_path") or not result.get("backbone_path")))):
        raise ValueError("robotd_bam requires lightx2v_tinyvla and local model/backbone artifact paths")
    # Older clients can omit the new workflow field; they still obey the same
    # task/scene/policy compatibility rules as the new UI.
    candidates = [name for name, flow in WORKFLOWS.items()
                  if result.get("task") in flow["tasks"] and result.get("scene") in flow["scenes"] and policy in flow["policies"]]
    workflow = result.get("workflow")
    if workflow is None and candidates:
        workflow = candidates[0]
    if workflow not in candidates or ("category" in result and result["category"] != WORKFLOWS[workflow]["category"]):
        raise ValueError("incompatible task, scene or policy for this workflow; choose its listed options")
    if mode == "robotd_bam" and not WORKFLOWS[workflow].get("realtime"):
        raise ValueError("robotd_bam is available for visual goals; preset command timing is offline only")
    seed = result.get("seed", 0)
    if isinstance(seed, bool) or not str(seed).isascii() or not str(seed).isdigit():
        raise ValueError("seed must be a non-negative integer")
    result.update(workflow=workflow, category=WORKFLOWS[workflow]["category"], seed=int(seed), mode=mode)
    project = WORKFLOWS[workflow].get("acceptance_scope") == "project_tasks_v2"
    if registered and (result.get("acceptance_scope") == "project_tasks_v2") != project:
        raise ValueError("模型动作版本与实验方案不匹配；请选择对应的项目训练动作或原预设动作")
    if project:
        result["initial_conditions"] = "qualified_initial_v2"
    elif not registered:
        # Hidden fields left by a previous workflow cannot change task semantics.
        result.pop("initial_conditions", None)
    if policy == 'community:skill-planner':
        from urllib.parse import urlparse
        if urlparse(str(result.get('endpoint', ''))).scheme not in ('http', 'https') or not str(result.get('planner_model', '')).strip():
            raise ValueError('请填写 VLM 服务地址与模型名称；本平台没有内置替代模型')
        if result.get('planner_context', 'feedback') not in ('current_frame', 'feedback'):
            raise ValueError('unknown planner context mode')
    elif not registered:
        result.pop('endpoint', None)
        result.pop('planner_model', None)
        result.pop('planner_context', None)
    if policy == "lightx2v_tinyvla":
        for key in (() if registered else ("model_path", "backbone_path")):
            value = result.get(key)
            if not isinstance(value, str) or not value.strip() or not Path(value.strip()).is_dir():
                raise ValueError(f"{key} must be an existing local directory on the server")
            result[key] = value.strip()
        if not registered:
            import json
            manifest_path = Path(result["model_path"]) / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            if (manifest.get("acceptance_scope") == "project_tasks_v2") != project:
                raise ValueError("本机模型动作版本与实验方案不匹配")
            if project:
                from .qualified_eval import validate_qualified_manifest
                validate_qualified_manifest(manifest, [result["task"]])
        if result.get("device", "cpu") not in ("cpu", "cuda") or result.get("precision", "fp32") not in ("fp32", "bf16"):
            raise ValueError("unsupported device or precision")
    else:
        for key in ("model_path", "backbone_path", "device", "precision"):
            result.pop(key, None)
    return result


def record_category(record):
    """Classify old records from actual execution, never from the launch form."""
    policy = record.get("policy", {}).get("policy_id", "")
    task = record.get("task")
    if record.get("scene") in ("water_court", "ball_tracking") or (task in TASKS and TASKS[task].suite.startswith("community_")) or any(name in policy for name in ("follow-v4", "rough-walk", "basketball", "tricks_", "vlm_skill_planner", "official-velstand", "duckbatch-128")):
        return "community"
    if task in VISUAL_TASKS:
        return "visual"
    return "presets" if task in PROGRAMS or task == "ground_lower" else "other"
