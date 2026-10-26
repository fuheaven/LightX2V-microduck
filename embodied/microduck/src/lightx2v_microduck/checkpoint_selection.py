"""Dependency-free ranking shared by host scheduling and model evaluation."""
from math import isfinite
from statistics import mean


SELECTION_PROTOCOL = "strict_then_terminal_stability_then_predicates_v1"


def checkpoint_selection_score(report):
    """Rank fixed-validation candidates without changing skill acceptance.

    Complete-task success remains the first criterion. At the same strict
    success rate, prefer terminal balance and then fewer failed predicates.
    This gives a zero-success curriculum a learning signal without treating
    partial progress as a passed skill. Macro averaging prevents task counts
    from changing their relative weight. Old best-validation receipts already
    contain the task summaries needed to compute this score.
    """
    strict = float(report["mean_strict_rate"])
    if not isfinite(strict) or not 0 <= strict <= 1:
        raise ValueError("checkpoint selection requires a finite strict rate")
    summaries = list(report["tasks"].values())
    if not summaries:
        raise ValueError("checkpoint selection requires per-task results")
    stable_rates, failed_per_episode = [], []
    for task in summaries:
        count = task["episodes"]
        failures = task["failed_checks"]
        if count < 1 or any(not isfinite(value) or not 0 <= value <= count for value in failures.values()):
            raise ValueError("invalid validation predicate counts")
        stable_rates.append(1 - failures.get("terminal_stable", 0) / count)
        failed_per_episode.append(sum(failures.values()) / count)
    return (strict, float(mean(stable_rates)), -float(mean(failed_per_episode)))
