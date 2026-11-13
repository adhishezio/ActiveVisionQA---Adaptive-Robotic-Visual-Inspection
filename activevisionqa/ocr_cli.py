"""Generate, fit, evaluate and run the CPU OCR baseline."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2

from .ocr import LineRecognizer
from .ocr_data import generate_words
from .ocr_evaluation import evaluate
from .ocr_training import train_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=217)
    generate.add_argument("--counts", type=int, nargs=3, default=[80, 20, 20], metavar=("TRAIN", "VALIDATION", "TEST"))
    train = sub.add_parser("train")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--dataset", type=Path)
    train.add_argument("--annotations", type=Path)
    train.add_argument("--seed", type=int, default=991)
    train.add_argument("--per-character", type=int, default=80)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--dataset", type=Path, required=True)
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--split", choices=["train", "validation", "test", "all"], default="validation")
    evaluation.add_argument("--annotations", type=Path)
    evaluation.add_argument("--validation-report", type=Path)
    evaluation.add_argument("--development-smoke", action="store_true")
    predict = sub.add_parser("predict")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_words(args.output, args.seed, tuple(args.counts))
    elif args.command == "train":
        result = train_model(args.output, args.dataset, args.annotations, args.seed, args.per_character)
        result = {k: result[k] for k in ("algorithm", "training_features", "generated_glyphs", "support_vectors", "training_seconds", "hardware", "scope")}
    elif args.command == "evaluate":
        report = evaluate(args.dataset, args.model, args.output, args.split, args.annotations,
                          args.validation_report, args.development_smoke)
        result = {"source": report["source"], "split": report["split"], "development_smoke": report["development_smoke"],
                  "baselines": {name: summary["overall"] for name, summary in report["baselines"].items()}}
    else:
        recognizer = LineRecognizer.load(args.model) if args.model else LineRecognizer()
        image = cv2.imread(str(args.image))
        if image is None:
            parser.error("Cannot read image")
        result = asdict(recognizer.recognize(image))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
