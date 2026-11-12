"""Protocols: load, validate, resolve and freeze an immutable experiment plan.

A resolved protocol is the whole scientific claim in one hashed document: suite
version, seeds, adapters, size choices, cell list, job matrix, metric, stopping
rules, limits. Any change after freezing produces a new protocol ID.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .datasets import mqar
from .datasets.manifest import load_manifest, verify_manifest
from .models import CANDIDATE_WIDTH_GRID, T0_WIDTH_GRID, resolve_size, registered_adapters
from .util import environment_record, resolve_device, sha256_json, tree_hash, write_json

SCHEMA_VERSION = "0.1"

#: Fields the compiler accepts. Unknown fields fail validation (spec 17).
ALLOWED_FIELDS = {
    "schema_version",
    "status",
    "protocol_id",
    "study_id",
    "suite",
    "comparison_regime",
    "model_ids",
    "train_seeds",
    "validation_seed",
    "test_seed",
    "generator_version",
    "dataset_manifest",
    "dataset_manifest_sha256",
    "adapter_commits",
    "target_trainable_parameters",
    "parameter_tolerance_fraction",
    "learning_rates",
    "effective_batch_examples",
    "screen_updates",
    "confirmation_updates",
    "primary_metric",
    "evaluation_state_policy",
    "evaluation_cells",
    "eval_per_cell",
    "stages",
    "stage_limits",
    "total_accelerator_hours_limit",
    "total_currency_limit",
    "max_new_artifacts_bytes",
    "paid_execution_allowed",
    "precision",
    "notes",
}

#: fields produced by `resolve`; accepted when validating a resolved document
DERIVED_FIELDS = {
    "protocol_id",
    "protocol_sha256",
    "heldout_sha256",
    "capability_matrix",
    "job_matrix",
    "expanded_job_count",
    "worst_case_budget",
    "environment",
    "resolved_by",
}

PLACEHOLDERS = {"must-pin-before-freeze", "must-resolve-before-freeze", "must-resolve-per-model", ""}


class ProtocolError(ValueError):
    pass


def load_protocol(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"{path}: invalid JSON ({exc})") from exc


def validate_structure(protocol: dict, *, strict: bool = True) -> list[str]:
    problems: list[str] = []
    unknown = set(protocol) - ALLOWED_FIELDS - DERIVED_FIELDS
    if unknown:
        problems.append(f"unknown fields: {sorted(unknown)}")
    missing = ALLOWED_FIELDS - {"notes", "protocol_id", "study_id", "evaluation_cells"} - set(protocol)
    if missing:
        problems.append(f"missing required fields: {sorted(missing)}")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if protocol.get("suite") != mqar.SUITE:
        problems.append(f"only {mqar.SUITE!r} is implemented at this milestone")
    if protocol.get("comparison_regime") not in {"structural", "parameter_matched", "compute_matched"}:
        problems.append("comparison_regime must be structural, parameter_matched or compute_matched")
    if protocol.get("evaluation_state_policy") != "reset_per_example":
        problems.append("evaluation_state_policy must be 'reset_per_example' for MQAR")
    if protocol.get("primary_metric") != "id_cell_macro_query_accuracy":
        problems.append("primary_metric must be id_cell_macro_query_accuracy for MQAR")
    unknown_models = set(protocol.get("model_ids", [])) - {"B0", "B1", "B2", "B3", "T0"}
    if unknown_models:
        problems.append(f"unknown model ids: {sorted(unknown_models)}")
    for key in ("generator_version", "dataset_manifest_sha256", "adapter_commits"):
        value = protocol.get(key)
        if isinstance(value, str) and value in PLACEHOLDERS and strict:
            problems.append(f"{key} is an unresolved placeholder")
    return problems


def resolve(protocol: dict, *, base_dir: str | Path, device: str | None = None) -> dict:
    """Fill every derived field. Returns a new resolved protocol document."""
    problems = validate_structure(protocol, strict=False)
    if problems:
        raise ProtocolError("; ".join(problems))

    base = Path(base_dir)
    device = resolve_device(device)
    manifest_path = Path(protocol["dataset_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = base / manifest_path
    if not manifest_path.exists():
        raise ProtocolError(f"dataset manifest missing: {manifest_path}")
    ok, issues = verify_manifest(manifest_path)
    if not ok:
        raise ProtocolError("dataset manifest verification failed: " + "; ".join(issues))
    manifest = load_manifest(manifest_path)
    if manifest["suite"] != mqar.SUITE:
        raise ProtocolError("manifest suite mismatch")
    manifest_sha = sha256_json(manifest)
    declared = protocol.get("dataset_manifest_sha256")
    if declared not in (None, "must-resolve-before-freeze") and declared != manifest_sha:
        raise ProtocolError(
            f"dataset_manifest_sha256 mismatch: protocol says {declared}, manifest hashes to {manifest_sha}"
        )

    commit = tree_hash(Path(__file__).resolve().parents[2])
    adapters = registered_adapters(source_commit=commit, device=device)
    adapters = {k: adapters[k] for k in protocol["model_ids"]}

    cells = protocol.get("evaluation_cells") or [mqar.cell_id(*c) for c in mqar.ID_GRID]
    matrix: dict[str, dict] = {}
    for mid, ad in adapters.items():
        size = resolve_size(
            int(protocol["target_trainable_parameters"]),
            use_memory=ad.use_memory,
            use_reasoning=ad.use_reasoning,
            tolerance=float(protocol["parameter_tolerance_fraction"]),
            widths=T0_WIDTH_GRID if ad.kind == "t0" else CANDIDATE_WIDTH_GRID,
        )
        row = {"size": size.to_dict(), "capabilities": ad.capabilities().to_dict(), "cells": {}}
        for cid in cells:
            status, reason = ad.preflight(cid)
            row["cells"][cid] = {"status": status, "reason": reason}
        matrix[mid] = row

    jobs = expand_jobs(protocol)
    stage_limits = protocol["stage_limits"]
    worst_case_seconds = sum(stage_limits[j["stage"]]["time_cap_s"] for j in jobs)

    resolved = copy.deepcopy(protocol)
    resolved.update(
        {
            "status": "resolved",
            "protocol_id": f"{protocol['suite']}-{manifest_sha[:12]}",
            "generator_version": manifest["generator_version"],
            "dataset_manifest": str(manifest_path.relative_to(base)) if manifest_path.is_relative_to(base) else str(manifest_path),
            "dataset_manifest_sha256": manifest_sha,
            "heldout_sha256": manifest["heldout_sha256"],
            "adapter_commits": {mid: ad.source_commit for mid, ad in adapters.items()},
            "evaluation_cells": cells,
            "capability_matrix": matrix,
            "job_matrix": jobs,
            "expanded_job_count": len(jobs),
            "worst_case_budget": {
                "seconds": worst_case_seconds,
                "accelerator_hours": round(worst_case_seconds / 3600.0, 6),
                "basis": "sum of per-stage time caps for every expanded job",
            },
            "environment": environment_record(device),
            "resolved_by": "loadstone 0.1.0",
        }
    )
    resolved["protocol_sha256"] = sha256_json(
        {k: v for k, v in resolved.items() if k != "protocol_sha256"}
    )
    return resolved


def expand_jobs(protocol: dict) -> list[dict]:
    """Expand the full screening/confirmation matrix. One entry = one job."""
    stages = protocol.get("stages") or ["screen", "confirmation"]
    limits = protocol["stage_limits"]
    jobs: list[dict] = []
    if "screen" in stages:
        for mid in protocol["model_ids"]:
            for seed in protocol["train_seeds"]:
                for lr in protocol["learning_rates"]:
                    jobs.append(
                        {
                            "job_id": f"screen-{mid}-s{seed}-lr{lr:g}",
                            "stage": "screen",
                            "model_id": mid,
                            "seed": seed,
                            "learning_rate": lr,
                            "updates": protocol["screen_updates"],
                            "time_cap_s": limits["screen"]["time_cap_s"],
                        }
                    )
    if "confirmation" in stages:
        for mid in protocol["model_ids"]:
            for seed in protocol["train_seeds"]:
                jobs.append(
                    {
                        "job_id": f"confirm-{mid}-s{seed}",
                        "stage": "confirmation",
                        "model_id": mid,
                        "seed": seed,
                        "learning_rate": "selected_from_screen",
                        "updates": protocol["confirmation_updates"],
                        "time_cap_s": limits["confirmation"]["time_cap_s"],
                    }
                )
    if "smoke" in stages:
        for mid in protocol["model_ids"]:
            jobs.append(
                {
                    "job_id": f"smoke-{mid}-s{protocol['train_seeds'][0]}",
                    "stage": "smoke",
                    "model_id": mid,
                    "seed": protocol["train_seeds"][0],
                    "learning_rate": protocol["learning_rates"][1],
                    "updates": limits["smoke"]["updates"],
                    "time_cap_s": limits["smoke"]["time_cap_s"],
                }
            )
    return jobs


def freeze(resolved: dict, outdir: str | Path) -> Path:
    """Write the resolved protocol + its hash into the study directory."""
    outdir = Path(outdir)
    if resolved.get("total_accelerator_hours_limit") in (None, 0):
        raise ProtocolError(
            "total_accelerator_hours_limit is unset: the default cumulative budget blocks "
            "launch. Set an explicit maximum before freezing a launchable study."
        )
    resolved = copy.deepcopy(resolved)
    resolved["status"] = "frozen"
    resolved.pop("protocol_sha256", None)
    resolved["protocol_sha256"] = sha256_json(resolved)
    write_json(outdir / "protocol.resolved.json", resolved)
    (outdir / "protocol.sha256").write_text(resolved["protocol_sha256"] + "\n", encoding="utf-8")
    return outdir / "protocol.resolved.json"


def approval_record(resolved: dict, *, plan_shown: bool, limits: dict | None = None) -> dict:
    """App record of a real UI/CLI confirmation (never generated from a config)."""
    if not plan_shown:
        raise ProtocolError("approval requires that the expanded plan was shown to the user")
    return {
        "protocol_sha256": resolved["protocol_sha256"],
        "approved_by": "local-user",
        "approved_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "confirmed_job_count": resolved["expanded_job_count"],
        "confirmed_worst_case_accelerator_hours": resolved["worst_case_budget"]["accelerator_hours"],
        "limits": limits or {},
        "note": "Recorded from an explicit local confirmation after the job matrix was displayed.",
    }


def read_approval(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    for key in ("protocol_sha256", "approved_at", "confirmed_job_count"):
        if key not in rec:
            raise ProtocolError(f"approval record missing {key!r}")
    return rec


def summarize_plan(resolved: dict) -> str:
    lines = [
        f"protocol            {resolved['protocol_id']}  ({resolved['comparison_regime']})",
        f"suite               {resolved['suite']} {resolved['generator_version']}",
        f"manifest sha256     {resolved['dataset_manifest_sha256']}",
        f"models              {', '.join(resolved['model_ids'])}",
        f"train seeds         {resolved['train_seeds']}",
        f"learning rates      {resolved['learning_rates']}",
        f"cells               {len(resolved['evaluation_cells'])}",
        f"expanded jobs       {resolved['expanded_job_count']}",
        f"worst case          {resolved['worst_case_budget']['accelerator_hours']:.3f} accelerator-hours",
        f"approved ceiling    {resolved['total_accelerator_hours_limit']} accelerator-hours",
        "",
        "capability matrix:",
    ]
    for mid, row in resolved["capability_matrix"].items():
        size = row["size"]
        statuses = {c["status"] for c in row["cells"].values()}
        lines.append(
            f"  {mid}  w{size['width']} d{size['depth']} params={size['params']:,} "
            f"matched={size['matched']} cells={','.join(sorted(statuses))}"
        )
    return "\n".join(lines)
