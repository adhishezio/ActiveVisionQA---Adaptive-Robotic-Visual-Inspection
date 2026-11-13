"""A local evidence report; all values come from an actual recorded run."""

import html
import json
from pathlib import Path


def render_report(directory: str | Path) -> Path:
    """Render camera evidence and ordered events without external resources."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    metadata = json.loads((directory / "run.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    rows = []
    decisions = []
    for event in events:
        evidence = event.get("evidence", {})
        motion = event.get("motion", {})
        viewpoint = event.get("viewpoint", "")
        if isinstance(viewpoint, dict):
            viewpoint = viewpoint.get("name", "")
        reason = event.get("reason", evidence.get("reason", motion.get("reason", "")))
        rows.append(f"<tr><td>{event['sequence']}</td><td>{esc(event['stage'])}</td><td>{esc(viewpoint)}</td><td>{esc(reason)}</td></tr>")
        if event["stage"] == "VIEW_SELECTION":
            candidates = []
            def fmt(value):
                return "—" if value is None else f"{float(value):.3f}"
            for candidate in event["candidates"]:
                checks = candidate.get("preview", {}).get("checks", {})
                candidates.append(f"<tr><td>{esc(candidate['viewpoint'])}</td>"
                                  f"<td>{fmt(candidate.get('predicted_visible_fraction'))}</td>"
                                  f"<td>{fmt(candidate.get('projected_min_label_edge_px'))}</td>"
                                  f"<td>{fmt(checks.get('planned_duration_s'))}</td>"
                                  f"<td>{fmt(candidate.get('score'))}</td><td>{esc(candidate['reason'])}</td></tr>")
            decisions.append(f"<section><h2>Next-view selection: {esc(event.get('selected') or 'escalate')}</h2>"
                             f"<p>Observation: {esc(event.get('evidence_reason'))}. Geometry source: {esc(event.get('context_source'))}.</p>"
                             "<table><tr><th>Candidate</th><th>Predicted visible fraction</th><th>Label edge px</th><th>Planned seconds</th><th>Utility</th><th>Status</th></tr>"
                             +"".join(candidates)+"</table><p>Visibility is a known-box estimate, not measured readability. "
                             "Motion previews do not execute the robot; the chosen goal is replanned before execution.</p></section>")
        if event.get("image"):
            crop = f"<img src='{esc(event['roi_image'])}' alt='Rectified inspection ROI'>" if event.get("roi_image") else ""
            cards.append(f"<figure><img src='{esc(event['image'])}' alt='Recorded camera evidence'>"
                         f"{crop}<figcaption>{esc(viewpoint)} · {esc(evidence.get('outcome', ''))} · {esc(reason)}"
                         f"<p>Recognized text: {esc(evidence.get('text') or '—')}</p>"
                         f"<pre>{esc(json.dumps(evidence.get('metrics', {}), indent=2))}</pre>"
                         f"Simulation timestamp: {esc(event.get('image_stamp'))}</figcaption></figure>")
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>ActiveVisionQA inspection evidence</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f5f7fa;color:#182638}}
header,figure,section{{background:white;padding:1.25rem;border-radius:10px;margin-bottom:1rem}}
.views{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}}
figure{{margin:0}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}
td,th{{text-align:left;padding:.5rem;border-bottom:1px solid #dae0e8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style><header><h1>ActiveVisionQA · {esc(result['outcome'])}</h1>
<p>{esc(result['reason'])}</p><p>Backend: <strong>{esc(result['backend'])}</strong> ·
Views: {result['views']} · Motion attempts: {result['motion_attempts']} ·
Elapsed: {result['elapsed_seconds']:.3f} s</p>
<p>Scenario: {esc(metadata.get('scenario'))}. This report covers the recorded case only.</p></header>
<div class="views">{''.join(cards)}</div>{''.join(decisions)}<section><h2>Recorded events</h2><table>
<tr><th>#</th><th>Stage</th><th>View</th><th>Reason</th></tr>{''.join(rows)}</table></section>
<section><h2>Artifacts</h2><p><a href="run.json">Configuration</a> · <a href="result.json">Result</a> ·
<a href="events.jsonl">Full events and planned trajectories</a></p></section></html>'''
    output = directory / "report.html"
    output.write_text(text, encoding="utf-8")
    return output
