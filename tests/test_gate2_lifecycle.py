"""Gate 2 lifecycle: cancel, pause and resume with checkpoint fidelity.

These exercise the runner end to end on a tiny matrix. They are slow-ish (a few
seconds each) and are kept out of the Gate 0 / Gate 1 suites on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loadstone.protocol import approval_record, expand_jobs, resolve
from loadstone.runner import Limits, StudyRunner
from loadstone.util import read_json, write_json


@pytest.fixture()
def study(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    protocol = json.loads((Path(__file__).resolve().parents[1] / "protocols" / "mqar-v1.json").read_text())
    protocol["stages"] = ["smoke"]
    protocol["model_ids"] = ["B0"]
    protocol["train_seeds"] = [17]
    protocol["total_accelerator_hours_limit"] = 1.0
    protocol["eval_per_cell"] = 4
    resolved = resolve(protocol, base_dir=Path.cwd(), device="cpu")
    resolved["study_id"] = "lifecycle"
    study_dir = tmp_path / "lifecycle"
    study_dir.mkdir()
    write_json(study_dir / "protocol.resolved.json", resolved)
    jobs = expand_jobs(resolved)
    jobs[0]["updates"] = 12
    jobs[0]["time_cap_s"] = 300
    approval = approval_record(resolved, plan_shown=True)
    limits = Limits(eval_per_cell=4, eval_batch_size=4)
    return study_dir, resolved, jobs[0], approval, limits


def test_pause_then_resume_continues_the_stream(study):
    study_dir, resolved, job, approval, limits = study
    runner = StudyRunner(study_dir, resolved, device="cpu", limits=limits)
    runner.prepare(approval)
    write_json(study_dir / "control.json", {"pause": True, "cancel": False})
    paused = runner.run_job(job)
    assert paused.status == "PAUSED"
    summary = read_json(study_dir / "jobs" / job["job_id"] / "summary.json")
    assert summary["updates_completed"] < 12
    ckpt = summary["resumable_checkpoint"]
    assert ckpt and Path(ckpt).exists()

    write_json(study_dir / "control.json", {"pause": False, "cancel": False})
    runner2 = StudyRunner(study_dir, resolved, device="cpu", limits=limits)
    resumed = runner2.run_job(job, resume_from=ckpt)
    assert resumed.status in ("COMPLETED", "CENSORED")
    final = read_json(study_dir / "jobs" / job["job_id"] / "summary.json")
    assert final["updates_completed"] == 12
    events = [json.loads(line) for line in (study_dir / "jobs" / job["job_id"] / "events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "resumed" for e in events)


def test_cancel_stops_without_claiming_completion(study):
    study_dir, resolved, job, approval, limits = study
    runner = StudyRunner(study_dir, resolved, device="cpu", limits=limits)
    runner.prepare(approval)
    write_json(study_dir / "control.json", {"cancel": True, "pause": False})
    res = runner.run_job(job)
    assert res.status == "CANCELLED"
    summary = read_json(study_dir / "jobs" / job["job_id"] / "summary.json")
    assert summary["updates_completed"] == 0
    assert summary["censored"] is False


def test_budget_ceiling_blocks_a_job_before_start(study):
    study_dir, resolved, job, approval, limits = study
    resolved = dict(resolved)
    resolved["total_accelerator_hours_limit"] = 0.0001
    runner = StudyRunner(study_dir, resolved, device="cpu", limits=limits)
    runner.prepare(approval)
    job = dict(job)
    job["time_cap_s"] = 3600
    res = runner.run_job(job)
    assert res.status == "BUDGET_EXHAUSTED"


def test_artifact_contract_is_written(study):
    study_dir, resolved, job, approval, limits = study
    runner = StudyRunner(study_dir, resolved, device="cpu", limits=limits)
    runner.prepare(approval)
    runner.run_job(job)
    job_dir = study_dir / "jobs" / job["job_id"]
    for name in (
        "events.jsonl",
        "training-metrics.jsonl",
        "evaluation-predictions.jsonl",
        "model-effective-config.json",
        "summary.json",
    ):
        assert (job_dir / name).exists(), name
    meta = read_json(job_dir / "model-effective-config.json")
    assert meta["trainable_parameters"] == meta["size"]["params"]
    assert meta["capabilities"]["representation"] == "discrete_symbol_ids"
    for path in (study_dir,):
        assert (path / "environment.json").exists()
        assert (path / "approval-and-limits.json").exists()
