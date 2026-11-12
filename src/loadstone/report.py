"""Reports: deterministic tables, charts and plain text from stored measurements.

Every chart is derived from files in the study directory, so a number on a chart
can always be traced back to a JSONL row or a summary.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .util import read_json

HEADER_LABELS = (
    "suite",
    "generator_version",
    "source_commit",
    "comparison_regime",
    "data_split",
    "context_unit",
    "precision",
    "protocol_sha256",
)


def _study(study_dir: Path) -> tuple[dict, list[dict]]:
    resolved = read_json(study_dir / "protocol.resolved.json")
    jobs = []
    for p in sorted((study_dir / "jobs").glob("*/summary.json")):
        jobs.append(json.loads(p.read_text(encoding="utf-8")))
    return resolved, jobs


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_tables(study_dir: Path, resolved: dict, jobs: list[dict]) -> None:
    tables = study_dir / "tables"
    tables.mkdir(exist_ok=True)
    with open(tables / "jobs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "job_id", "stage", "model_id", "seed", "learning_rate", "status", "censored",
                "updates_completed", "updates_planned", "elapsed_s", "trainable_parameters",
                "final_cell_macro_query_accuracy", "final_query_accuracy",
                "final_sequence_accuracy", "test_cell_macro_query_accuracy", "protocol_sha256",
            ]
        )
        for s in jobs:
            fv = s.get("final_validation") or {}
            test = s.get("test") or {}
            w.writerow(
                [
                    s["job_id"], s["stage"], s["model_id"], s["seed"], s["learning_rate"], s["status"],
                    s["censored"], s["updates_completed"], s["updates_planned"], f"{s['elapsed_s']:.3f}",
                    s["size"]["params"], _fmt(fv.get("cell_macro_query_accuracy")),
                    _fmt(fv.get("query_accuracy")), _fmt(fv.get("cell_macro_sequence_accuracy")),
                    _fmt(test.get("cell_macro_query_accuracy")), s["protocol_sha256"],
                ]
            )

    with open(tables / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["job_id", "model_id", "seed", "split", "cell", "query_accuracy", "sequence_accuracy",
             "cross_entropy_nats", "examples", "failed_examples", "coverage"]
        )
        for s in jobs:
            for split_key, split_name in (("final_validation", "validation"), ("test", "test")):
                block = s.get(split_key) or {}
                for cell, row in (block.get("cells") or {}).items():
                    w.writerow(
                        [s["job_id"], s["model_id"], s["seed"], split_name, cell,
                         _fmt(row["query_accuracy"]), _fmt(row["sequence_accuracy"]),
                         _fmt(row["cross_entropy_nats"]), row["examples"], row["failed_examples"],
                         _fmt(row["coverage"])]
                    )

    (study_dir / "tables" / "summary.json").write_text(
        json.dumps({"header": {k: resolved.get(k) for k in HEADER_LABELS}, "jobs": jobs}, indent=2) + "\n",
        encoding="utf-8",
    )


def _scatter_svg(points: list[tuple[float, float, str]], *, xlabel: str, ylabel: str, title: str) -> str:
    W, H, pad = 720, 420, 56
    if not points:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120"><text x="20" y="40">{title}: no data</text></svg>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmax = max(xs) * 1.1 or 1.0
    ymax = max(ys) * 1.1 or 1.0
    def sx(v): return pad + (W - 2 * pad) * (v / xmax)
    def sy(v): return H - pad - (H - 2 * pad) * (v / ymax)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
        f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#333"/>',
        f'<text x="{W/2}" y="{H-14}" text-anchor="middle">{xlabel}</text>',
        f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle">{ylabel}</text>',
    ]
    colors = {"B0": "#1f77b4", "B1": "#ff7f0e", "B2": "#2ca02c", "B3": "#d62728", "T0": "#7f7f7f"}
    for x, y, label in points:
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{colors.get(label, "#555")}"/>')
        parts.append(f'<text x="{sx(x)+8:.1f}" y="{sy(y)-6:.1f}">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_charts(study_dir: Path, jobs: list[dict]) -> None:
    charts = study_dir / "charts"
    charts.mkdir(exist_ok=True)
    qc = []
    for s in jobs:
        fv = s.get("final_validation") or {}
        if fv.get("cell_macro_query_accuracy") is not None:
            qc.append((s["elapsed_s"] / 60.0, fv["cell_macro_query_accuracy"],
                       f"{s['model_id']}-s{s['seed']}"))
    (charts / "quality_vs_training_minutes.svg").write_text(
        _scatter_svg(qc, xlabel="training wall time (minutes)",
                     ylabel="validation cell-macro query accuracy",
                     title="Quality vs training cost (validation only, not test)"),
        encoding="utf-8",
    )
    # Validation curves: one polyline per job.
    W, H, pad = 720, 420, 56
    series = [(s["job_id"], s["validation_curve"]) for s in jobs if s.get("validation_curve")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14">Validation cell-macro query accuracy vs budget fraction</text>',
        f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#333"/>',
        f'<text x="{W/2}" y="{H-14}" text-anchor="middle">fraction of allocated training budget</text>',
        f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle">validation accuracy</text>',
    ]
    for i, (job_id, curve) in enumerate(series):
        pts = " ".join(
            f"{pad + (W - 2*pad) * c['fraction']:.1f},{H - pad - (H - 2*pad) * c['cell_macro_query_accuracy']:.1f}"
            for c in curve
        )
        color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f", "#9467bd", "#8c564b"][i % 7]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{W-pad+2}" y="{pad+14*i}" font-size="10" fill="{color}">{job_id}</text>')
    parts.append("</svg>")
    (charts / "validation_curves.svg").write_text("".join(parts), encoding="utf-8")


def write_report(study_dir: Path, resolved: dict, jobs: list[dict]) -> str:
    env = resolved.get("environment", {})
    lines = [
        "Loadstone study report",
        "======================",
        "",
        f"study            {study_dir.name}",
        f"suite            {resolved['suite']} {resolved.get('generator_version')}",
        f"comparison       {resolved['comparison_regime']}",
        f"protocol sha256  {resolved.get('protocol_sha256')}",
        f"dataset manifest {resolved.get('dataset_manifest_sha256')}",
        f"source tree      {env.get('source_tree')}",
        f"device           {env.get('device_name', env.get('device'))}  torch {env.get('torch')}  cuda {env.get('cuda')}",
        f"precision        {resolved.get('precision')}",
        f"data split       {resolved['dataset_manifest']} (validation / test are locked held-out sets)",
        f"context unit     1 example = 1 token sequence; evaluation state reset per example",
        f"metric           {resolved['primary_metric']} (chance 1/64 per query)",
        "",
        "Scope and limitations",
        "---------------------",
        "* Synthetic MQAR only. This is a controlled app variant, not a Zoology reproduction.",
        "* Results here are pilot/smoke scale unless a study reached its full confirmation budget.",
        "* Unsupported or failed cells are shown as such and never as a zero score.",
        "* Reported uncertainty is across independent training seeds; query positions inside one",
        "  trained model are not independent samples and are never counted as such.",
        "",
        "Jobs",
        "----",
        f"{'job_id':34s} {'model':6s} {'seed':>5s} {'status':16s} {'updates':>8s} {'min':>7s} {'val acc':>8s} {'test acc':>9s}",
    ]
    for s in jobs:
        fv = s.get("final_validation") or {}
        test = s.get("test") or {}
        lines.append(
            f"{s['job_id']:34s} {s['model_id']:6s} {s['seed']:>5d} {s['status']:16s} "
            f"{s['updates_completed']:>8d} {s['elapsed_s']/60:>7.2f} "
            f"{_fmt(fv.get('cell_macro_query_accuracy'), 4):>8s} {_fmt(test.get('cell_macro_query_accuracy'), 4):>9s}"
        )

    lines += ["", "Per-cell validation accuracy (in-distribution cells)", "---------------------------------------------------"]
    header = "job".ljust(34) + "".join(c.rjust(10) for c in resolved["evaluation_cells"])
    lines.append(header)
    for s in jobs:
        fv = s.get("final_validation") or {}
        cells = fv.get("cells") or {}
        row = s["job_id"].ljust(34) + "".join(
            _fmt((cells.get(c) or {}).get("query_accuracy"), 3).rjust(10) for c in resolved["evaluation_cells"]
        )
        lines.append(row)

    stages = {s.get("stage") for s in jobs}
    lines += ["", "Comparison notes", "----------------"]
    if "smoke" in stages:
        lines.append("* Every job in this study ran the smoke allocation (20 updates). These are")
        lines.append("  plumbing results: accuracies near 1/64 are expected and are NOT a ranking.")
    lines.append(f"* Regime: {resolved['comparison_regime']}. Sizes were chosen by a frozen")
    lines.append("  parameter-count search, not by accuracy; each job records its own width/depth.")
    b0 = next((s for s in jobs if s["model_id"] == "B0" and s.get("final_validation")), None)
    t0 = next((s for s in jobs if s["model_id"] == "T0" and s.get("final_validation")), None)
    if b0 and t0 and b0["seed"] == t0["seed"]:
        same = abs(
            b0["final_validation"]["cell_macro_query_accuracy"]
            - t0["final_validation"]["cell_macro_query_accuracy"]
        ) < 1e-12 and b0["size"]["params"] == t0["size"]["params"]
        if same:
            lines.append("* B0 and T0 built identically on this backbone (same parameter count and")
            lines.append("  identical losses). B0-T0 is a plumbing check, not a scientific contrast;")
            lines.append("  the relevant contrasts are B1/B2/B3 against B0.")
    learning = study_dir / "gates-learning.json"
    if learning.exists():
        try:
            doc = json.loads(learning.read_text(encoding="utf-8"))
            lines.append(
                f"* Learning-plumbing gate (same-16-example overfit, not capability): "
                f"{'PASS' if doc.get('all_passed') else 'FAIL'} "
                + ", ".join(
                    f"{r['model_id']}={r['best_accuracy_on_micro']:.2f}@lr{r['best_learning_rate']:g}"
                    for r in doc.get("rows", [])
                )
            )
        except Exception:
            pass

    lines += ["", "Coverage and failures", "---------------------"]
    for s in jobs:
        fv = s.get("final_validation") or {}
        failed = sum((c or {}).get("failed_examples", 0) for c in (fv.get("cells") or {}).values())
        lines.append(
            f"{s['job_id']:34s} coverage={_fmt(fv.get('coverage'), 4)} failed_examples={failed} "
            f"stop_reason={s.get('stop_reason')}"
        )

    lines += ["", "Artifacts", "---------",
              "jobs/<job_id>/events.jsonl, training-metrics.jsonl, evaluation-predictions.jsonl,",
              "resource-metrics.jsonl, model-effective-config.json, summary.json, checkpoints/",
              "tables/jobs.csv, tables/cells.csv, charts/*.svg", ""]
    text = "\n".join(lines)
    (study_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def build_report(study_dir: str | Path, formats: str = "txt,csv,json") -> str:
    study_dir = Path(study_dir)
    resolved, jobs = _study(study_dir)
    wanted = {f.strip() for f in formats.split(",")}
    if "csv" in wanted:
        write_tables(study_dir, resolved, jobs)
    write_charts(study_dir, jobs)
    if "json" in wanted:
        (study_dir / "report.json").write_text(
            json.dumps(
                {
                    "header": {k: resolved.get(k) for k in HEADER_LABELS}
                    | {"device": resolved.get("environment", {}).get("device_name")},
                    "jobs": jobs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return write_report(study_dir, resolved, jobs)
