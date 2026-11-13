"""Plan, collect, audit and visually review synthetic text camera datasets."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from .dataset import SCENARIOS, audit_dataset, load_plan, make_plan, render_gallery, write_json, write_review_template


def collect(plan_path: Path, directory: Path, limit: int | None, gui: bool = False) -> int:
    """Launch isolated workcells and stop only owned simulator process groups."""
    from .validate_simulation import stop_owned_process
    plan = load_plan(plan_path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", plan)
    for name in ("configs", "captures", "logs", "worlds"):
        (directory / name).mkdir()
    project = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment["AVQA_ROS_DOMAIN_ID"] = environment.get("AVQA_DATASET_ROS_DOMAIN_ID", "69")
    environment["AVQA_GZ_PARTITION"] = "activevisionqa_dataset_"+environment["AVQA_ROS_DOMAIN_ID"]
    cases = plan["cases"][:limit] if limit else plan["cases"]
    statuses = []
    with (directory / "samples.jsonl").open("x") as manifest:
        for number, case in enumerate(cases, 1):
            case_id = case["case_id"]
            config_path = directory / "configs" / (case_id+".json")
            write_json(config_path, case["config"])
            command = ["bash", str(project / "scripts/run_simulation.sh"), f"gui:={'true' if gui else 'false'}",
                       "scenario:="+case["scenario"], "config:="+str(config_path), "output_root:="+str(directory / "worlds")]
            print(f"Collecting {number}/{len(cases)}: {case_id} ({case['split']})", flush=True)
            with (directory / "logs" / (case_id+"_simulation.log")).open("x") as log:
                process = subprocess.Popen(command, cwd=project, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    capture = ["bash", str(project / "scripts/capture_text_case.sh"), "--plan", str(directory / "plan.json"),
                               "--case-id", case_id, "--output", str(directory / "captures" / case_id)]
                    # Keep full child diagnostics without flooding the collection summary.
                    with (directory / "logs" / (case_id+"_capture.log")).open("x") as capture_log:
                        try:
                            result = subprocess.run(capture, cwd=project, env=environment, stdout=capture_log,
                                                    stderr=subprocess.STDOUT, timeout=240)
                            code = result.returncode
                        except subprocess.TimeoutExpired:
                            code = 124
                    case_output = directory / "captures" / case_id
                    sample_file = case_output / "samples.json"
                    rows = json.loads(sample_file.read_text()) if sample_file.exists() else []
                    for row in rows:
                        for field in ("image", "rectified_image"):
                            row[field] = (Path("captures") / case_id / row[field]).as_posix()
                        row["events"] = (Path("captures") / case_id / "events.jsonl").as_posix()
                        manifest.write(json.dumps(row, allow_nan=False)+"\n")
                    manifest.flush()
                    statuses.append({"case_id": case_id, "exit_code": code, "images": len(rows),
                                     "complete": code == 0 and len(rows) == case["config"]["max_views"]})
                    print(json.dumps(statuses[-1]), flush=True)
                finally:
                    stop_owned_process(process)
                if code != 0:
                    print(f"Stopping collection after failed case; retained logs: {directory / 'logs'}", flush=True)
                    break
    audit = audit_dataset(directory, require_complete=len(cases) == len(plan["cases"]))
    write_json(directory / "audit.json", audit)
    write_json(directory / "collection.json", {"selected_cases": len(cases), "statuses": statuses,
               "complete": len(statuses) == len(cases) and all(s["complete"] for s in statuses),
               "full_plan_selected": len(cases) == len(plan["cases"]), "model_trained": False})
    render_gallery(directory)
    write_review_template(directory)
    print(json.dumps({"output": str(directory), "audit": audit}, indent=2), flush=True)
    return 0 if audit["passed"] and len(statuses) == len(cases) and all(s["complete"] for s in statuses) else 2


def main(argv=None) -> int:
    """Keep data collection separate from inspection decisions and training."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--seed", type=int, default=123)
    plan.add_argument("--parts-per-split", type=int, default=1)
    plan.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["clear", "occluded"])
    plan.add_argument("--output", type=Path, required=True)
    capture = sub.add_parser("capture-case", help=argparse.SUPPRESS)
    capture.add_argument("--plan", type=Path, required=True)
    capture.add_argument("--case-id", required=True)
    capture.add_argument("--output", type=Path, required=True)
    collection = sub.add_parser("collect")
    collection.add_argument("--plan", type=Path, required=True)
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--limit", type=int)
    collection.add_argument("--gui", action="store_true", help="Show the owned Gazebo window while collecting")
    audit = sub.add_parser("audit")
    audit.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = make_plan(args.seed, args.parts_per_split, tuple(args.scenarios))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({"plan": str(args.output), "cases": len(result["cases"]), "expected_images": result["expected_images"]}))
        return 0
    if args.command == "capture-case":
        from .dataset_capture import capture_case
        cases = {c["case_id"]: c for c in load_plan(args.plan)["cases"]}
        if args.case_id not in cases:
            parser.error("Unknown planned case")
        return capture_case(cases[args.case_id], args.output)
    if args.command == "collect":
        return collect(args.plan, args.output, args.limit, args.gui)
    result = audit_dataset(args.directory)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
