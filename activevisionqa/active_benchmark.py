"""Fresh-workcell comparisons of fixed, sequence and next-best-view policies."""

import argparse
import html
import json
import os
from pathlib import Path
import subprocess

from .cli import run_name
from .config import Config
from .dataset import write_json
from .ocr import LineRecognizer
from .validate_simulation import stop_owned_process
from .validation import audit_run

POLICIES = {"fixed_view": ("sequence", 1), "sequence": ("sequence", 2), "next_best_view": ("next_best_view", 2)}
SCENARIOS = ("all_occluded", "top_occluded", "covered")


def expected_case(scenario: str, policy: str) -> tuple[str, int]:
    """Predeclared smoke expectations; failures are retained, never substituted."""
    if policy == "fixed_view":
        return "ESCALATE", 1
    if policy == "sequence":
        return "ESCALATE", 2
    return ("ESCALATE", 1) if scenario == "covered" else ("ACCEPT", 2)


def summarize_episode(directory: Path) -> dict:
    """Derive action/motion statistics from recorded events, not intended goals."""
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    motions = [event["motion"] for event in events if event["stage"] == "MOTION_RESULT" and event["motion"]["ok"]]
    decisions = [event for event in events if event["stage"] == "VIEW_SELECTION"]
    previews = [row["preview"] for event in decisions for row in event["candidates"] if "preview" in row]
    first = motions[0]["checks"]["planned_joint_trajectory"] if motions else None
    return {"selected_views": [event["selected"] for event in decisions],
            "selection_reasons": [event["reason"] for event in decisions],
            "motion_previews": len(previews), "rejected_previews": sum(not row["ok"] for row in previews),
            "successful_preview_planning_wall_seconds": sum(row["checks"].get("planning_wall_seconds", 0.) for row in previews if row["ok"]),
            "executed_planned_duration_s": sum(m["checks"].get("planned_duration_s", 0.) for m in motions),
            "executed_planned_joint_path_length_rad": sum(m["checks"].get("planned_joint_path_length_rad", 0.) for m in motions),
            "initial_joint_state": dict(zip(first["joint_names"], first["points"][0]["positions_rad"])) if first else None}


def audit_decisions(directory: Path) -> dict:
    """Verify executed choices follow feasible ranked previews and feedback."""
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    errors = []
    selections = [e for e in events if e["stage"] == "VIEW_SELECTION"]
    for selection in selections:
        future = [e for e in events if e["stage"] == "MOTION_REQUEST" and e["sequence"] > selection["sequence"]]
        if selection["selected"] is None:
            if future:
                errors.append("motion_after_escalated_selection")
            continue
        scored = [row for row in selection["candidates"] if row.get("eligible") and "score" in row]
        scored.sort(key=lambda row: (-row["score"], row["viewpoint"]))
        if not scored or scored[0]["viewpoint"] != selection["selected"]:
            errors.append("executed_choice_not_highest_ranked_feasible")
        if not future or future[0]["viewpoint"]["name"] != selection["selected"]:
            errors.append("selected_view_does_not_match_motion_request")
        for row in selection["candidates"]:
            preview = row.get("preview")
            if preview and preview["ok"] and (preview["phase"] != "plan" or preview["checks"].get("robot_motion_verified")):
                errors.append("preview_record_claims_execution")
    requests = [e["viewpoint"]["name"] for e in events if e["stage"] == "MOTION_REQUEST"]
    if len(set(requests)) != len(requests):
        errors.append("candidate_revisited")
    if any(e.get("unselected_candidate_outcomes") != "unknown" for e in events if e["stage"] == "ACTION_FEEDBACK"):
        errors.append("unselected_outcomes_fabricated")
    return {"passed": not errors, "errors": errors, "selection_events": len(selections)}


def matching_starts(records: list[dict]) -> bool:
    """Require the same measured planning start across fresh-workcell episodes."""
    starts = [row.get("episode", {}).get("initial_joint_state") for row in records]
    if not starts or any(state is None for state in starts):
        return False
    first = starts[0]
    return all(set(state) == set(first) and max(abs(state[name]-first[name]) for name in first) < .02 for state in starts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/active_inspection.json"))
    parser.add_argument("--ocr-model", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    if len(config.camera_positions) < 3 or config.marking_kind != "text":
        parser.error("Active comparison needs a text config with at least three candidates")
    if len(set(args.scenarios)) != len(args.scenarios) or len(set(args.policies)) != len(args.policies):
        parser.error("Choose unique scenarios and policies")
    LineRecognizer.load(args.ocr_model)
    output = (args.output or Path("outputs") / ("active_vision_"+run_name())).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "worlds", "runs"):
        (output / name).mkdir()
    config_path = output / "config.json"
    write_json(config_path, config.to_dict())
    project = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["AVQA_ROS_DOMAIN_ID"] = env.get("AVQA_ACTIVE_ROS_DOMAIN_ID", "75")
    env["AVQA_GZ_PARTITION"] = "activevisionqa_policy_"+env["AVQA_ROS_DOMAIN_ID"]
    records = []
    for scenario in args.scenarios:
        for policy in args.policies:
            name = scenario+"_"+policy
            kind, budget = POLICIES[policy]
            directory = output / "runs" / name
            print(f"Starting fresh workcell: {name}", flush=True)
            launch = ["bash", str(project / "scripts/run_simulation.sh"), f"gui:={'true' if args.gui else 'false'}",
                      "scenario:="+scenario, "config:="+str(config_path), "output_root:="+str(output / "worlds")]
            with (output / "logs" / (name+"_simulation.log")).open("x") as log:
                process = subprocess.Popen(launch, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    command = ["bash", str(project / "scripts/inspect_robot.sh"), "--config", str(config_path),
                               "--scenario", scenario, "--policy", kind, "--max-views", str(budget),
                               "--ocr-model", str(args.ocr_model.resolve()), "--output", str(directory)]
                    with (output / "logs" / (name+"_inspection.log")).open("x") as log:
                        try:
                            code = subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240).returncode
                        except subprocess.TimeoutExpired:
                            code = 124
                    expected, views = expected_case(scenario, policy)
                    result = json.loads((directory / "result.json").read_text()) if (directory / "result.json").exists() else {"outcome": "ERROR", "reason": "no_result"}
                    audit = audit_run(directory, expected, code, views)
                    decision_audit = audit_decisions(directory) if (directory / "events.jsonl").exists() else {"passed": False, "errors": ["no_events"]}
                    episode = summarize_episode(directory) if result.get("motion_attempts") else {}
                    matched = audit["passed"] and decision_audit["passed"]
                    if policy == "next_best_view" and not decision_audit.get("selection_events"):
                        matched = False
                    record = {"scenario": scenario, "policy": policy, "expected": expected, "expected_views": views,
                              "exit_code": code, "matched": matched, "result": result, "audit": audit,
                              "decision_audit": decision_audit, "episode": episode,
                              "run_directory": directory.relative_to(output).as_posix()}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    _save_report(output, records, complete=False)
                finally:
                    stop_owned_process(process)
    report = _save_report(output, records, complete=True)
    return 0 if all(row["matched"] for row in records) and report["matching_initial_states"] else 2


def _save_report(output: Path, records: list[dict], complete: bool) -> dict:
    report = {"scope": "predeclared known-box active-vision smoke comparison; not a generalization or safety claim",
              "complete": complete, "matched": sum(row["matched"] for row in records), "total": len(records),
              "fresh_world_per_policy": True, "matching_initial_states": matching_starts(records), "cases": records}
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for row in records:
        result, episode = row["result"], row["episode"]
        path = row["run_directory"]+"/report.html"
        rows.append(f"<tr><td>{esc(row['scenario'])}</td><td>{esc(row['policy'])}</td>"
                    f"<td>{esc(result['outcome'])}</td><td>{result.get('views', 0)}</td>"
                    f"<td>{result.get('elapsed_seconds', 0):.3f}</td><td>{episode.get('motion_previews', 0)}</td>"
                    f"<td>{esc(episode.get('selected_views', []))}</td><td><a href='{esc(path)}'>Evidence</a></td></tr>")
    text = "<!doctype html><html lang='en'><meta charset='utf-8'><title>ActiveVisionQA active-view comparison</title>"
    text += "<style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:1rem}td,th{padding:.6rem;text-align:left;border-bottom:1px solid #ddd}table{width:100%}</style>"
    text += f"<h1>Active-view comparison: {report['matched']}/{len(records)} expected cases</h1>"
    text += "<p>Known static box geometry; geometric utilities and checked motion previews. No PyTorch/VLM policy, learned value function or calibrated information-gain claim.</p>"
    text += "<table><tr><th>Scene</th><th>Policy</th><th>Outcome</th><th>Views</th><th>Inspection s</th><th>Previews</th><th>Selected alternatives</th><th>Run</th></tr>"
    text += "".join(rows)+"</table><p>Each episode starts in a fresh workcell. Times include preview planning and execution but exclude startup/readiness. "
    text += "These individual controlled runs do not establish a statistically reliable cycle-time improvement.</p></html>"
    (output / "comparison.html").write_text(text, encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
