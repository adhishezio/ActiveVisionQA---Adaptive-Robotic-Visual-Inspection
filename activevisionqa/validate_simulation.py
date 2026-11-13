"""Run real Gazebo/MoveIt cases and retain independent evidence for each."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .cli import run_name
from .config import Config
from .validation import audit_run


def stop_owned_process(process: subprocess.Popen) -> None:
    """Stop only the process group created by this validation run."""
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-unreachable", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ocr-model", type=Path)
    parser.add_argument("--ocr-backend", choices=["svm", "template"], default="svm")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=["occluded", "wrong_id", "all_occluded"],
                        default=["occluded", "wrong_id", "all_occluded"])
    args = parser.parse_args(argv)
    base_config = Config.load(args.config)
    if base_config.marking_kind == "text" and args.ocr_backend == "svm" and args.ocr_model is None:
        parser.error("Text SVM validation requires --ocr-model")
    if base_config.marking_kind == "text" and args.ocr_backend == "svm":
        from .ocr import LineRecognizer
        LineRecognizer.load(args.ocr_model)
    project = Path(__file__).resolve().parent.parent
    directory = (args.output or project / "outputs" / ("simulation_validation_"+run_name())).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["AVQA_ROS_DOMAIN_ID"] = environment.get("AVQA_ROS_DOMAIN_ID", "68")
    cases = [
        ("occluded", [("fixed_view", 1, "ESCALATE"), ("active", 2, "ACCEPT")]),
        ("wrong_id", [("active", 2, "REJECT")]),
        ("all_occluded", [("active", 2, "ESCALATE")]),
    ]
    cases = [case for case in cases if case[0] in args.scenarios]
    if args.include_unreachable:
        cases.append(("clear", [("unreachable", 1, "ESCALATE")]))
    records = []
    for scenario, policies in cases:
        config_path = args.config.resolve() if args.config else None
        if policies[0][0] == "unreachable":
            config_path = directory / "unreachable_config.json"
            config = replace(base_config, max_views=1, camera_positions=((.84, .79, .99),))
            config_path.write_text(json.dumps(config.to_dict(), indent=2)+"\n")
        log_path = directory / f"{scenario}_{policies[0][0]}_simulation.log"
        command = ["bash", str(project / "scripts/run_simulation.sh"), f"gui:={'true' if args.gui else 'false'}", f"scenario:={scenario}", f"output_root:={directory}"]
        if config_path:
            command.append(f"config:={config_path}")
        with log_path.open("w") as simulation_log:
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=simulation_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                time.sleep(2)
                for policy, max_views, expected in policies:
                    run_dir = directory / f"{scenario}_{policy}"
                    inspect = ["bash", str(project / "scripts/inspect_robot.sh"), "--scenario", scenario,
                               "--max-views", str(max_views), "--output", str(run_dir)]
                    if config_path:
                        inspect += ["--config", str(config_path)]
                    if args.ocr_model:
                        inspect += ["--ocr-model", str(args.ocr_model.resolve())]
                    inspect += ["--ocr-backend", args.ocr_backend]
                    with (directory / f"{scenario}_{policy}_inspection.log").open("w") as log:
                        try:
                            completed = subprocess.run(inspect, cwd=project, env=environment, stdout=log,
                                                       stderr=subprocess.STDOUT, timeout=240)
                            exit_code = completed.returncode
                        except subprocess.TimeoutExpired:
                            exit_code = 124
                    result_path = run_dir / "result.json"
                    result = json.loads(result_path.read_text()) if result_path.exists() else {"outcome": "ERROR", "reason": "result_not_written"}
                    expected_views = 0 if policy == "unreachable" else (2 if scenario == "all_occluded" or (scenario == "occluded" and policy == "active") else 1)
                    audit = audit_run(run_dir, expected, exit_code, expected_views)
                    record = {"scenario": scenario, "policy": policy, "expected": expected,
                              "matched": audit["passed"], "audit": audit, "exit_code": exit_code,
                              "result": result, "run_directory": str(run_dir)}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    (directory / "validation.json").write_text(json.dumps({"scope": "Gazebo/MoveIt scenario smoke suite", "cases": records}, indent=2)+"\n")
            finally:
                stop_owned_process(process)
    summary = {"scope": "Gazebo/MoveIt scenario smoke suite", "cases": records,
               "matched": sum(item["matched"] for item in records), "total": len(records)}
    (directory / "validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"output": str(directory), "matched": summary["matched"], "total": summary["total"]}), flush=True)
    return 0 if all(item["matched"] for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
