"""Evaluation with explicit ground-truth denominators and visible abstentions."""

from collections import Counter
from typing import Any


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize every case, retaining escalations and errors in denominators."""
    counts = Counter(record["outcome"] for record in records)
    acceptable = [r for r in records if r["truth"] == "acceptable"]
    defective = [r for r in records if r["truth"] == "defective"]
    false_accepts = sum(r["outcome"] == "ACCEPT" for r in defective)
    false_rejects = sum(r["outcome"] == "REJECT" for r in acceptable)
    correct = sum((r["truth"] == "acceptable" and r["outcome"] == "ACCEPT") or
                  (r["truth"] == "defective" and r["outcome"] == "REJECT") for r in records)
    n = len(records)
    return {
        "cases": n, "acceptable_cases": len(acceptable), "defective_cases": len(defective),
        "outcomes": dict(counts), "false_accepts": false_accepts, "false_rejects": false_rejects,
        "false_accept_rate": false_accepts / len(defective) if defective else None,
        "false_reject_rate": false_rejects / len(acceptable) if acceptable else None,
        "correct_definitive_decisions": correct,
        "correct_definitive_fraction": correct/n if n else None,
        "escalation_fraction": counts["ESCALATE"]/n if n else None,
        "error_fraction": counts["ERROR"]/n if n else None,
        "mean_views": sum(r["views"] for r in records)/n if n else None,
        "latency_scope": "backend-specific; fixture timings are not robot cycle times",
    }
