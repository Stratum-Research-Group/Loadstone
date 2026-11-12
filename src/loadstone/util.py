"""Small shared utilities: deterministic hashing, seeding, atomic artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------- hashing


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def tree_hash(root: str | Path) -> str:
    """Hash of tracked source files (git-aware), used as the source commit stand-in."""
    root = Path(root)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        head = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"git:{head}{'-dirty' if dirty else ''}"
    except Exception:
        h = hashlib.sha256()
        for p in sorted(list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))):
            h.update(p.as_posix().encode())
            h.update(p.read_bytes())
        return f"tree:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------- seeding


def derive_seed(*parts: object) -> int:
    """Stable 64-bit seed from arbitrary parts (independent random streams)."""
    h = hashlib.sha256("|".join(repr(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def rng_for(*parts: object) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(derive_seed(*parts)))


# ---------------------------------------------------------------- artifacts


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_jsonable) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, rows: Iterable[dict], fsync: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=_jsonable) + "\n")
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- environment


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def environment_record(device: str) -> dict:
    rec: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "source_tree": tree_hash(Path(__file__).resolve().parents[2]),
        "device": device,
    }
    try:
        import torch

        rec["torch"] = torch.__version__
        rec["cuda"] = torch.version.cuda
        rec["cudnn"] = torch.backends.cudnn.version()
        rec["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        rec["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        if device == "cuda":
            rec["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            rec["device_total_mem_bytes"] = props.total_memory
            rec["device_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # pragma: no cover - torch always present in this env
        rec["torch_error"] = repr(exc)
    rec["numpy"] = np.__version__
    return rec


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
