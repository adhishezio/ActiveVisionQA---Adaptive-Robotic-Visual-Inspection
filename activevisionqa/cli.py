"""CLI for deterministic fixtures, paired comparisons and ROS inspection."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .config import Config
from .assets import WORLD_SCENARIOS
from .evaluation import summarize
from .fixtures import FixtureBackend, SCENARIOS
from .perception import MarkerInspector
from .policy import InspectionRunner
from .recording import RunRecorder
from .report import render_report


def run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


def run_fixture(config: Config, scenario: str, seed: int, output: Path) -> dict:
    """Run a policy test fixture and write its complete evidence bundle."""
    backend = FixtureBackend(config, scenario, seed)
    truth = next(item.truth for item in SCENARIOS if item.name == scenario)
    metadata = {"backend": backend.mode, "scenario": scenario, "seed": seed, "truth": truth,
                "config": config.to_dict(), "robot_motion_verified": False}
    with RunRecorder(output, metadata) as recorder:
        result = InspectionRunner(config, backend, MarkerInspector(config), recorder).run()
    render_report(output)
    return {**result.to_dict(), "scenario": scenario, "seed": seed, "truth": truth, "run_directory": str(output)}


def main(argv: list[str] | None = None) -> int:
    """Run one fixture, a paired fixture benchmark, or actual ROS integration."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="Policy/image fixture; does not validate robot motion")
    fixture.add_argument("--scenario", choices=[s.name for s in SCENARIOS], default="occluded")
    fixture.add_argument("--seed", type=int, default=42)
    fixture.add_argument("--max-views", type=int, choices=[1, 2], default=2)
    fixture.add_argument("--output", type=Path)
    benchmark = sub.add_parser("benchmark", help="Compare fixed-view/active policies on identical fixtures")
    benchmark.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    benchmark.add_argument("--output", type=Path)
    robot = sub.add_parser("robot", help="Inspect using an already running Gazebo/MoveIt workcell")
    robot.add_argument("--output", type=Path)
    robot.add_argument("--scenario", default="occluded", choices=WORLD_SCENARIOS)
    robot.add_argument("--max-views", type=int, choices=range(1, 6))
    robot.add_argument("--policy", choices=["sequence", "next_best_view"])
    robot.add_argument("--ocr-model", type=Path, help="Trained OCR bundle directory for text workcells")
    robot.add_argument("--ocr-backend", choices=["svm", "template"], default="svm")
    for command in (fixture, benchmark, robot):
        command.add_argument("--config", type=Path, dest="command_config")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.config and args.command_config:
        parser.error("Supply --config only once")
    config = Config.load(args.command_config or args.config)
    if args.command == "robot":
        config = replace(config, max_views=args.max_views if args.max_views is not None else config.max_views,
                         selection_policy=args.policy or config.selection_policy)
    elif config.selection_policy != "sequence":
        parser.error("Next-best-view selection requires the robot backend and its planning scene")
    if config.marking_kind == "text":
        if args.command != "robot":
            parser.error("Text inspection uses the robot command; marker fixtures remain ArUco-only")
        if args.ocr_backend == "svm" and args.ocr_model is None:
            parser.error("Text SVM inspection requires --ocr-model; see scripts/ocr.sh train")
    output = args.output or Path("outputs") / (args.command + "_" + run_name())
    if args.command == "fixture":
        result = run_fixture(replace(config, max_views=args.max_views), args.scenario, args.seed, output)
        print(json.dumps(result, indent=2))
        return 0 if result["outcome"] not in ("ERROR",) else 2
    if args.command == "robot":
        from .ros_backend import run_robot
        return run_robot(config, output, args.scenario, args.ocr_model, args.ocr_backend)
    output.mkdir(parents=True, exist_ok=False)
    records = {"fixed_view": [], "active": []}
    for policy, max_views in (("fixed_view", 1), ("active", 2)):
        for scenario in SCENARIOS:
            for seed in args.seeds:
                run_output = output / f"{policy}_{scenario.name}_{seed}"
                records[policy].append(run_fixture(replace(config, max_views=max_views), scenario.name, seed, run_output))
    report = {"scope": "offline fixtures; no robot/simulator improvement claim",
              "split": "predefined_diagnostic_suite", "seeds": args.seeds,
              "scenarios": [s.__dict__ for s in SCENARIOS],
              "policies": {name: summarize(rows) for name, rows in records.items()}, "records": records}
    (output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(output), "policies": report["policies"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
