"""Capability gallery built from verified reports, never from menu availability."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

from . import training_dashboard as training

COMMUNITY_REVIEW = Path(__file__).parent / "resources" / "community-review.json"
DESCRIPTIONS = {
    "walk_forward": "向前行走，在任务结束时稳定停下。",
    "circle_left": "保持站立，沿左转轨迹行走并结束动作。",
    "figure_eight": "连续切换左右转向，完成八字行走。",
    "zigzag": "交替转向，完成之字轨迹并停下。",
    "ground_lower": "降低身体接近地面，再恢复站立；不是鸭嘴触地验收。",
    "roller_crouch": "在滚轮本体上蹲伏、恢复并结束滑行。",
    "roller_glide": "在滚轮本体上滑行，并稳定停下。",
    "roller_slalom": "按预设时序切换滚轮动作；不是视觉绕桩。",
    "sway_circle": "先摇摆，再转圈。",
    "kick_left": "在规定起始位置用左脚踢球。",
    "kick_right": "在规定起始位置用右脚踢球。",
    "sit_walk": "坐下、起身，再向前行走。",
    "roll_circle": "前滚翻后恢复，再转圈。",
}


def _verified_report(directory, reference, artifacts):
    item = artifacts.get(reference.get("artifact_id"))
    if not item or not item.get("available") or item.get("path") != reference.get("source"):
        return None, None
    try:
        report = training.read_json(training.artifact_path(directory, item))
        if not isinstance(report, dict) or not training.valid_digest(report.get("checkpoint_sha256")):
            return None, None
        if report["checkpoint_sha256"] != reference.get("checkpoint_sha256"):
            return None, None
        return report, item
    except (OSError, ValueError, KeyError):
        return None, None


def _recordings(task, report, report_item, artifacts):
    """Bind each clip to its report's task, episode and outcome, including failures."""
    root = PurePosixPath(report_item["path"]).parent / "episodes"
    episodes = {row["seed"]: row for row in report.get("episodes", []) if row.get("task") == task}
    clips = []
    for item in artifacts.values():
        if item.get("kind") != "video" or not item.get("available"):
            continue
        path = PurePosixPath(item["path"])
        if path.parent.parent != root or path.name not in ("observer.mp4", "camera.mp4"):
            continue
        prefix = task + "-"
        if not path.parent.name.startswith(prefix):
            continue
        try:
            seed = int(path.parent.name[len(prefix):])
        except ValueError:
            continue
        episode = episodes.get(seed)
        if episode is None:
            continue
        clips.append({"id": item["id"], "url": item["url"], "seed": seed,
                      "view": "observer" if path.name == "observer.mp4" else "camera",
                      "success": episode.get("success") is True,
                      "failed_checks": episode.get("task_result", {}).get("failed_checks", []),
                      "sha256": item.get("sha256")})
    return sorted(clips, key=lambda x: (not x["success"], x["view"] != "observer", x["seed"]))


def skill_cards(directory, snapshot):
    registered = snapshot.get("registered_acceptance") or {}
    campaign = registered.get("campaign") or {}
    artifacts = {item["id"]: item for item in snapshot.get("artifacts", [])}
    cards = []
    for task, label in training.TASK_LABELS.items():
        card = {"id": task, "label": "低头近地并起身" if task == "ground_lower" else label, "description": DESCRIPTIONS[task],
                "embodiment": "rollers" if task.startswith("roller_") else "legs",
                "status": "unassessed", "score": None, "videos": [], "report_url": None,
                "model_sha256": None, "realtime_accepted": False, "hardware_validated": False}
        references = [r for r in campaign.get("reports", [])
                      if r.get("mode") == "offline" and task in r.get("tasks", {})]
        if references:
            report, item = _verified_report(Path(directory), references[-1], artifacts)
            if report is None:
                card["status"] = "evidence_unavailable"
            else:
                score = report.get("tasks", {}).get(task, {})
                final = (report.get("purpose") == "final_test" and report.get("independent_final_test") is True
                         and report.get("complete") is True and report.get("evaluator_role") == "student")
                if final:
                    accepted = training.scoped_acceptance(report, score, task, "offline")
                    card.update(status="accepted" if accepted else "failed", score={
                        key: score.get(key) for key in ("successes", "episodes", "strict_rate", "wilson_95", "failed_checks", "takeovers")},
                        report_url=item["url"], model_sha256=report["checkpoint_sha256"],
                        videos=_recordings(task, report, item, artifacts),
                        uses_prescribed_command13=report.get("uses_prescribed_command13") is True,
                        physics_waits_for_model=report.get("physics_waits_for_model") is True)
        cards.append(card)
    return cards


def community_cards(asset_root=None, review_path=None):
    from .community import verify_community_policy
    from .web_catalog import WORKFLOWS
    review = json.loads(Path(review_path or COMMUNITY_REVIEW).read_text())
    cards = []
    for original in review["projects"]:
        card = dict(original)
        workflow = WORKFLOWS.get(card.get("workflow"))
        card["launch_url"] = None
        card["asset_errors"] = []
        if workflow:
            for policy in card.get("required_assets", []):
                result = verify_community_policy(policy, asset_root)
                if not result["ok"]:
                    card["asset_errors"].append({"policy": policy, "errors": result["errors"]})
            if not card["asset_errors"]:
                card["launch_url"] = "/experiments?" + urlencode({"workflow": card["workflow"],
                    "task": card.get("task", workflow["tasks"][0]), "mode": "live"})
        card["launch_ready"] = bool(card["launch_url"]) and not card.get("requires_service", False)
        cards.append(card)
    return {"checked_at": review["checked_at"], "discovery": review["discovery"], "projects": cards}


def gallery(directory, asset_root=None):
    snapshot = training.dashboard(directory)
    cards = skill_cards(directory, snapshot)
    community = community_cards(asset_root)
    registered = snapshot.get("registered_acceptance") or {}
    campaign = registered.get("campaign") or {}
    return {"schema_version": 1, "skills": cards, "community": community,
            "summary": {"accepted": sum(c["status"] == "accepted" for c in cards),
                        "tested": sum(c["status"] in ("accepted", "failed") for c in cards),
                        "total": len(cards), "realtime_accepted": 0, "hardware_validated": False},
            "campaign": {"id": campaign.get("id"), "completed_at": campaign.get("completed_at")},
            "freshness": snapshot.get("freshness"), "snapshot_available": snapshot.get("available", False),
            "scope": "本项目 13 项预设行为（含 ground_lower 替代项），成绩绑定登记模型与独立测试报告。"}
