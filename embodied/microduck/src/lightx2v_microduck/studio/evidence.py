"""Evidence gates are shared by API, worker and exports. UI cannot award success."""

import math
from collections import Counter, defaultdict


def interval(successes, total):
    if not total:
        return [0.0, 1.0]
    z = 1.95996398454
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [max(0, center - half), min(1, center + half)]


def summarize(episodes, *, teacher=False, groups=(), mode="offline", model_sha=None, case_sha=None):
    slices = defaultdict(list)
    for e in episodes:
        slices["group:" + e["group"]].append(e)
        slices["category:" + e["category"]].append(e)
        slices["color:" + e["color"]].append(e)

    def score(rows):
        n = len(rows)
        wins = sum(bool(r["success"]) for r in rows)
        return {"passed": wins, "total": n, "rate": wins / n if n else 0.0, "confidence95": interval(wins, n)}

    totals = score(episodes)
    subgroups = {k: score(v) for k, v in slices.items()}
    unique = len({e["layout_id"] for e in episodes})
    qualified = (
        bool(groups) and unique == len(episodes) and all(subgroups.get("group:" + g, {}).get("total", 0) >= 100 and subgroups["group:" + g]["rate"] >= 0.95 for g in groups) if teacher else False
    )
    realtime_ok = mode != "realtime" or all(0.9 <= e.get("rtf", 0) <= 1.1 for e in episodes)
    accepted = (
        not teacher
        and len(episodes) >= 200
        and unique == len(episodes)
        and totals["rate"] >= 0.8
        and realtime_ok
        and all(e.get("split") == "test" for e in episodes)
        and bool(model_sha)
        and bool(case_sha)
        and all(subgroups.get("group:" + g, {}).get("total", 0) > 0 for g in groups)
    )
    return {
        **totals,
        "schema": "studio_evaluation_v1",
        "groups": subgroups,
        "failures": dict(Counter(e["reason"] for e in episodes if not e["success"])),
        "teacher_qualified": qualified,
        "accepted": accepted,
        "mode": mode,
        "model_sha256": model_sha,
        "cases_sha256": case_sha,
        "unique_layouts": unique,
        "realtime_speed_passed": realtime_ok,
        "episodes": episodes,
        "hardware_validated": False,
        "evidence": "teacher" if teacher else "independent_test" if accepted else "development_or_insufficient_evidence",
    }


def compare(reports, prices=None):
    prices = prices or {}
    rows = []
    identities = {(r.get("cases_sha256"), r.get("mode"), r.get("execution")) for r in reports}
    comparable = len(identities) == 1 and all(r.get("cases_sha256") for r in reports)
    for report in reports:
        resource = report.get("resources", {})
        successes = report.get("passed", 0)
        gpu = resource.get("gpu_seconds", 0) / 3600
        cost = None if prices.get("gpu_hour") is None else gpu * prices["gpu_hour"] + resource.get("api_cost", 0)
        if resource.get("external_inference_unmetered"):
            if prices.get("api_input_million") is None or prices.get("api_output_million") is None:
                cost = None
            elif cost is not None:
                cost += (resource.get("input_tokens", 0) * prices["api_input_million"] + resource.get("output_tokens", 0) * prices["api_output_million"]) / 1e6
        rows.append(
            {
                "model": report.get("model_sha256"),
                "success_rate": report.get("rate"),
                "confidence95": report.get("confidence95"),
                "p95_ms": report.get("p95_ms"),
                "gpu_hours": gpu,
                "cost_per_success": cost / successes if cost is not None and successes else None,
                "accepted": report.get("accepted", False),
            }
        )
    recommendation = None
    if comparable and len(rows) == 2 and all(r["accepted"] for r in rows):
        baseline, candidate = rows
        reliable = candidate["confidence95"][0] >= baseline["confidence95"][1]
        faster = bool(candidate["p95_ms"] and baseline["p95_ms"] and candidate["p95_ms"] <= 0.8 * baseline["p95_ms"])
        cheaper = bool(candidate["cost_per_success"] and baseline["cost_per_success"] and candidate["cost_per_success"] <= 0.8 * baseline["cost_per_success"])
        if reliable and (faster or cheaper):
            recommendation = candidate["model"]
    savings = (rows[0]["cost_per_success"] - rows[1]["cost_per_success"]) if len(rows) == 2 and all(r["cost_per_success"] is not None for r in rows) else None
    return {
        "comparable": comparable,
        "rows": rows,
        "recommended_model": recommendation,
        "break_even_successes": math.ceil(prices["upfront_cost"] / savings) if savings and savings > 0 and prices.get("upfront_cost") is not None else None,
        "note": "Without matched cases and sufficient reliability evidence, results remain experimental.",
    }
