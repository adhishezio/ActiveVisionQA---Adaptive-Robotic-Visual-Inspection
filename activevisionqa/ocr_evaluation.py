"""OCR metrics with explicit denominators, abstentions and model/data lineage."""

from collections import Counter
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import time

from .dataset import file_sha256, write_json
from .ocr import LineRecognizer
from .ocr_data import read_image, read_rows
from .text_inspection import text_decision


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance with insertions, deletions and substitutions."""
    previous = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def summarize_ocr(records: list[dict]) -> dict:
    """Count abstentions as unresolved transcription, retaining all cases."""
    n = len(records)
    decided = [r for r in records if r["prediction"]["text"] is not None]
    exact = sum(r["prediction"]["text"] == r["surface_text"] for r in records)
    edits = sum(edit_distance(r["prediction"]["text"] or "", r["surface_text"]) for r in records)
    characters = sum(len(r["surface_text"]) for r in records)
    wrong = [r for r in records if r["surface_text"] != r["expected_text"]]
    correct = [r for r in records if r["surface_text"] == r["expected_text"]]
    false_accepts = sum(r["decision"] == "ACCEPT" for r in wrong)
    false_rejects = sum(r["decision"] == "REJECT" for r in correct)
    return {"samples": n, "definitive_transcriptions": len(decided), "exact_strings": exact,
            "exact_string_accuracy_all": exact/n if n else None,
            "exact_string_accuracy_decided": exact/len(decided) if decided else None,
            "character_edits": edits, "ground_truth_characters": characters,
            "character_error_rate_all": edits/characters if characters else None,
            "abstentions": n-len(decided), "abstention_fraction": (n-len(decided))/n if n else None,
            "wrong_id_cases": len(wrong), "correct_id_cases": len(correct),
            "false_accepts": false_accepts, "false_accept_rate": false_accepts/len(wrong) if wrong else None,
            "false_rejects": false_rejects, "false_reject_rate": false_rejects/len(correct) if correct else None,
            "decisions": dict(Counter(r["decision"] for r in records)),
            "mean_recognition_ms": float(sum(r["recognition_ms"] for r in records)/n) if n else None,
            "latency_scope": "CPU per-crop recognition; excludes camera acquisition and robot motion"}


def evaluate(dataset: Path, model: Path, output: Path, split: str = "validation", annotations: Path | None = None,
             validation_report: Path | None = None, development_smoke: bool = False) -> dict:
    """Compare template and SVM on one declared split with frozen model gates."""
    if split not in ("train", "validation", "test", "all") or (split == "all" and not development_smoke):
        raise ValueError("All-split evaluation requires explicit development-smoke mode")
    if output.exists():
        raise FileExistsError(output)
    rows, source = read_rows(dataset, annotations)
    selected = [row for row in rows if split == "all" or row["split"] == split]
    if not selected:
        raise ValueError("No samples in selected OCR split")
    if source == "gazebo_robot_camera" and not development_smoke and any(r["readability"] == "unreviewed" for r in selected):
        raise ValueError("Formal camera evaluation requires readability review; use development-smoke for pipeline diagnostics")
    recognizer = LineRecognizer.load(model)
    model_hash = file_sha256(model / "model.json")
    manifest_name = "manifest.jsonl" if source == "offline_synthetic_text" else "samples.jsonl"
    manifest_hash = file_sha256(dataset / manifest_name)
    annotation_path = annotations or dataset / "readability_review.csv"
    annotation_hash = file_sha256(annotation_path) if annotation_path.exists() else None
    if split == "test" and not development_smoke:
        if validation_report is None:
            raise ValueError("Test evaluation requires a preceding --validation-report with frozen model/data hashes")
        prior = json.loads(validation_report.read_text())
        if (prior.get("split") != "validation" or prior.get("model_sha256") != model_hash
                or prior.get("manifest_sha256") != manifest_hash or prior.get("development_smoke")):
            raise ValueError("Validation report does not match this frozen model and dataset")
        if prior.get("annotations_sha256") != annotation_hash:
            raise ValueError("Readability annotations changed after validation")
    fitted = set((recognizer.metadata.get("word_training") or {}).get("surface_strings", []))
    if split in ("validation", "test") and any(r["surface_text"] in fitted or r["expected_text"] in fitted for r in selected):
        raise ValueError("Evaluation word identity overlaps model training words")
    baselines = {"template": LineRecognizer(recognizer.settings, references=recognizer.references), "svm": recognizer}
    records = {name: [] for name in baselines}
    for row in selected:
        image = read_image(dataset, row)
        for name, reader in baselines.items():
            start = time.monotonic()
            prediction = reader.recognize(image)
            elapsed = (time.monotonic()-start)*1000
            decision = text_decision(prediction, row["expected_text"])
            records[name].append({key: row[key] for key in ("sample_id", "part_id", "split", "surface_text", "expected_text", "condition", "readability", "image")})
            records[name][-1].update(prediction=asdict(prediction), decision=decision.outcome.value,
                                     recognition_ms=elapsed)
    report = {"schema_version": 1, "source": source, "split": split, "development_smoke": development_smoke,
              "evaluation_role": "development_smoke" if development_smoke else ("training_diagnostic" if split == "train" else "held_out_word_split"),
              "scope": "same-font synthetic/camera diagnostics; no industrial generalization claim",
              "model_sha256": model_hash, "manifest_sha256": manifest_hash,
              "annotations_sha256": annotation_hash,
              "validation_report_sha256": file_sha256(validation_report) if validation_report else None,
              "settings": asdict(recognizer.settings), "readability_counts": dict(Counter(r["readability"] for r in selected)),
              "baselines": {name: {"overall": summarize_ocr(values),
                            "by_condition": {condition: summarize_ocr([r for r in values if r["condition"] == condition])
                                             for condition in sorted({r["condition"] for r in values})}}
                            for name, values in records.items()}}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    with (output / "predictions.jsonl").open("x") as stream:
        for name, values in records.items():
            for row in values:
                stream.write(json.dumps({"baseline": name, **row}, allow_nan=False)+"\n")
    _render_report(output, dataset, report, records)
    return report


def _render_report(output: Path, dataset: Path, report: dict, records: dict) -> None:
    esc = lambda value: html.escape(str(value), quote=True)
    summaries, examples = [], []
    for name, values in records.items():
        summary = report["baselines"][name]["overall"]
        summaries.append(f"<tr><td>{esc(name)}</td><td>{summary['exact_strings']}/{summary['samples']}</td>"
                         f"<td>{summary['character_error_rate_all']:.3f}</td><td>{summary['abstentions']}</td>"
                         f"<td>{summary['false_accepts']}/{summary['wrong_id_cases']}</td>"
                         f"<td>{summary['false_rejects']}/{summary['correct_id_cases']}</td></tr>")
    for i, row in enumerate(records["svm"]):
        baseline = records["template"][i]
        image = os.path.relpath(dataset / row["image"], output)
        examples.append(f"<article><h3>{esc(row['sample_id'])}</h3><img src='{esc(image)}' alt='Evaluated text crop'>"
                        f"<p>Ground truth: {esc(row['surface_text'])}; expected: {esc(row['expected_text'])}.</p>"
                        f"<p>SVM: {esc(row['prediction']['text'] or 'ABSTAIN')} ({esc(row['prediction']['reason'])}); "
                        f"template: {esc(baseline['prediction']['text'] or 'ABSTAIN')}.</p>"
                        f"<p>Readability annotation: {esc(row['readability'])}.</p></article>")
    text = "<!doctype html><html lang='en'><meta charset='utf-8'><title>ActiveVisionQA OCR evaluation</title>"
    text += "<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:1rem;background:#eef2f6}"
    text += "article,section{background:white;padding:1rem;margin:1rem 0}img{max-width:100%}td,th{text-align:left;padding:.6rem}table{width:100%}</style>"
    text += f"<h1>OCR evaluation · {esc(report['split'])}</h1><p>Source: {esc(report['source'])}; "
    text += f"development smoke: {esc(report['development_smoke'])}. Same-font synthetic scope; no industrial accuracy claim.</p>"
    text += "<section><table><tr><th>Baseline</th><th>Exact/all</th><th>CER/all</th><th>Abstain</th><th>False accept/wrong ID</th><th>False reject/correct ID</th></tr>"
    text += "".join(summaries)+"</table><p>Abstentions count as unresolved full-string deletions in CER/all. "
    text += "Reference distances are not calibrated probabilities. CPU timings exclude robot movement.</p></section>"
    text += "".join(examples)+"</html>"
    (output / "report.html").write_text(text, encoding="utf-8")
