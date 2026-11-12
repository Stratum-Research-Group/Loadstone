"""Model adapters: the integration boundary between the app and a network.

The app owns tasks, splits, solvers, scorers, budgets and reports. An adapter
owns the network, its state and its compute telemetry, and declares exactly what
it supports. Unsupported cells are reported as unsupported, never scored zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..datasets import mqar
from .blocks import SequenceModel, ffn_hidden, n_heads_for

VOCAB = mqar.ALPHABET_SIZE
N_CLASSES = mqar.N_VALUES


@dataclass(frozen=True)
class Capabilities:
    symbolic_classification: bool = True
    causal_byte_likelihood: bool = False
    generated_answers: bool = False
    streaming_continuation: bool = False
    explicit_reset: bool = True
    fast_memory_updates: bool = False
    fixed_loop_execution: bool = False
    adaptive_halting: bool = False
    per_item_compute: bool = False
    checkpoint_resume: bool = True
    vision: bool = False
    representation: str = "discrete_symbol_ids"
    max_input_length: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ size search

DEPTH_GRID = (1, 2, 3, 4, 6, 8)
#: the baseline grid is fixed by the specification (section 5)
T0_WIDTH_GRID = (32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512)
#: the candidate backbone exports its own finite, frozen grid. It adds
#: intermediate widths because the optional memory module costs 4w^2 per block,
#: which makes the 1M parameter band unreachable on the baseline grid alone.
CANDIDATE_WIDTH_GRID = T0_WIDTH_GRID + (112, 144, 224, 288)


def analytic_param_count(
    width: int, depth: int, *, use_memory: bool = False, use_reasoning: bool = False
) -> int:
    """Exact parameter count of `SequenceModel` for a configuration."""
    w = width
    f = ffn_hidden(w)
    total = VOCAB * w  # token embedding
    block = 4 * w * w + 3 * w * f + 2 * w  # qkv + out, swiglu, two norms
    if use_memory:
        block += 4 * w * w + w + 1  # q/k/v/out projections + norm + residual scale
    total += depth * block
    total += w  # final norm
    total += w * N_CLASSES  # task readout head (untied)
    if use_reasoning:
        total += 4 * w * w + 3 * w * f + 3 * w + w * w  # block + norms + gate
    return total


@dataclass
class SizeChoice:
    width: int
    depth: int
    params: int
    heads: int
    target: int
    tolerance: float
    matched: bool
    deviation: float

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_size(
    target: int,
    *,
    use_memory: bool = False,
    use_reasoning: bool = False,
    tolerance: float = 0.05,
    widths: tuple[int, ...] = T0_WIDTH_GRID,
) -> SizeChoice:
    """Deterministic parameter-count search over the frozen integer grid.

    Choose the valid configuration closest to the target; tie-break by smaller
    width then smaller depth (spec section 5). This is a parameter-count search,
    not an accuracy search, and it is frozen before training.
    """
    best: tuple[float, int, int, int] | None = None
    for depth in DEPTH_GRID:
        for width in widths:
            try:
                heads = n_heads_for(width)
            except ValueError:
                continue
            params = analytic_param_count(
                width, depth, use_memory=use_memory, use_reasoning=use_reasoning
            )
            dev = abs(params - target) / target
            key = (round(dev, 12), width, depth, heads)
            if best is None or key < (best[0], best[1], best[2], best[3]):
                best = (dev, width, depth, heads)
    if best is None:  # pragma: no cover
        raise RuntimeError("empty size-search space")
    dev, width, depth, heads = best
    params = analytic_param_count(
        width, depth, use_memory=use_memory, use_reasoning=use_reasoning
    )
    return SizeChoice(
        width=width,
        depth=depth,
        params=params,
        heads=heads,
        target=target,
        tolerance=tolerance,
        matched=abs(params - target) / target <= tolerance,
        deviation=abs(params - target) / target,
    )


# ------------------------------------------------------------------ batching


@dataclass
class Batch:
    tokens: torch.Tensor  # [B, L] long
    pad_mask: torch.Tensor  # [B, L] bool
    query_positions: torch.Tensor  # [B, Kmax] long, -1 padded
    targets: torch.Tensor  # [B, Kmax] long, -1 padded
    examples: list = field(default_factory=list)

    @property
    def n_examples(self) -> int:
        return int(self.tokens.shape[0])


def encode_batch(examples: list[mqar.Example], device: str) -> Batch:
    if not examples:
        raise ValueError("empty batch")
    kmax = max(e.k for e in examples)
    lmax = max(e.length for e in examples)
    tokens = torch.zeros(len(examples), lmax, dtype=torch.long)
    pad_mask = torch.zeros(len(examples), lmax, dtype=torch.bool)
    qpos = torch.full((len(examples), kmax), -1, dtype=torch.long)
    tgt = torch.full((len(examples), kmax), -1, dtype=torch.long)
    for i, ex in enumerate(examples):
        n = ex.length
        tokens[i, :n] = torch.tensor(ex.tokens, dtype=torch.long)
        pad_mask[i, :n] = True
        qpos[i, : ex.k] = torch.tensor(ex.query_positions, dtype=torch.long)
        tgt[i, : ex.k] = torch.tensor(ex.targets, dtype=torch.long)
    return Batch(
        tokens=tokens.to(device),
        pad_mask=pad_mask.to(device),
        query_positions=qpos.to(device),
        targets=tgt.to(device),
        examples=examples,
    )


def gather_query_logits(logits: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    """Pick logits at query positions: [B, Kmax, C] with zeros for padded slots."""
    B, _, C = logits.shape
    kmax = query_positions.shape[1]
    clamped = query_positions.clamp_min(0)
    idx = clamped.unsqueeze(-1).expand(B, kmax, C)
    out = torch.gather(logits, 1, idx)
    valid = (query_positions >= 0).unsqueeze(-1)
    return out.masked_fill(~valid, 0.0)


def masked_query_loss(query_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean CE per query, then mean per example (spec 10 'Batch and loss units')."""
    B, K, C = query_logits.shape
    valid = targets >= 0
    flat_logits = query_logits.reshape(B * K, C)
    flat_targets = targets.reshape(B * K).clamp_min(0)
    per_query = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction="none")
    per_query = per_query.view(B, K).masked_fill(~valid, 0.0)
    counts = valid.sum(dim=1).clamp_min(1)
    return (per_query.sum(dim=1) / counts).mean()


# ------------------------------------------------------------------ adapter


class TorchMQARAdapter:
    """Torch adapter for the MQAR symbolic suite.

    `kind` selects the identity:
      * ``t0``        causal Transformer baseline
      * ``candidate`` backbone with optional memory / latent reasoning
    """

    def __init__(
        self,
        *,
        adapter_id: str,
        kind: str,
        use_memory: bool = False,
        use_reasoning: bool = False,
        reasoning_loops: int = 2,
        device: str = "cpu",
        source_commit: str = "unknown",
    ) -> None:
        self.adapter_id = adapter_id
        self.adapter_version = "0.1.0"
        self.kind = kind
        self.use_memory = use_memory
        self.use_reasoning = use_reasoning
        self.reasoning_loops = reasoning_loops
        self.device = device
        self.source_commit = source_commit
        self.model: SequenceModel | None = None
        self.size: SizeChoice | None = None
        self.config: dict = {}
        self.counters = self._fresh_counters()

    # -- declaration ---------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            fast_memory_updates=self.use_memory,
            fixed_loop_execution=self.use_reasoning,
            per_item_compute=True,
            max_input_length=4096,
            representation="discrete_symbol_ids",
            notes=(
                "Full self-attention, stateless per example (state is empty; reset is trivial). "
                f"memory={'on' if self.use_memory else 'off'} reasoning={'on' if self.use_reasoning else 'off'} "
                f"loops={self.reasoning_loops if self.use_reasoning else 0}"
            ),
        )

    def metadata(self) -> dict:
        caps = self.capabilities()
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "source_commit": self.source_commit,
            "kind": self.kind,
            "config": dict(self.config),
            "size": self.size.to_dict() if self.size else None,
            "trainable_parameters": self.model.trainable_parameters() if self.model else None,
            "total_parameters": self.model.total_parameters() if self.model else None,
            "capabilities": caps.to_dict(),
            "state_schema": {"kind": "stateless", "fields": []},
        }

    def preflight(self, cell: str) -> tuple[str, str]:
        L, K = mqar.parse_cell(cell)
        caps = self.capabilities()
        if L > caps.max_input_length:
            return "UNSUPPORTED", f"length {L} exceeds declared max_input_length {caps.max_input_length}"
        if 2 * K > L // 2:
            return "GATE_FAILED", f"cell {cell} is impossible (requires 2K <= L/2)"
        return "SUPPORTED", ""

    # -- lifecycle -----------------------------------------------------

    def build(self, config: dict, initialization_seed: int) -> "TorchMQARAdapter":
        """Build the network. `config` carries the frozen width/depth choice."""
        self.config = dict(config)
        torch.manual_seed(initialization_seed)
        np.random.seed(initialization_seed % (2**32))
        width = int(config["width"])
        depth = int(config["depth"])
        self.model = SequenceModel(
            vocab=VOCAB,
            n_classes=N_CLASSES,
            width=width,
            depth=depth,
            n_heads=int(config.get("n_heads", n_heads_for(width))),
            use_memory=self.use_memory,
            use_reasoning=self.use_reasoning,
            reasoning_loops=self.reasoning_loops,
        ).to(self.device)
        self.size = SizeChoice(
            width=width,
            depth=depth,
            params=self.model.trainable_parameters(),
            heads=self.model.n_heads,
            target=int(config.get("target_trainable_parameters", width)),
            tolerance=float(config.get("parameter_tolerance_fraction", 0.05)),
            matched=bool(config.get("parameter_matched_claimed", False)),
            deviation=float(config.get("parameter_deviation", 0.0)),
        )
        assert self.size.params == analytic_param_count(
            width, depth, use_memory=self.use_memory, use_reasoning=self.use_reasoning
        ), "analytic parameter count disagrees with the built tensors"
        return self

    def _fresh_counters(self) -> dict:
        return {
            "updates": 0,
            "examples": 0,
            "tokens": 0,
            "supervised_targets": 0,
            "forward_passes": 0,
            "backward_passes": 0,
            "reasoning_loop_iterations": 0,
            "memory_chunks": 0,
        }

    def reset_counters(self) -> None:
        self.counters = self._fresh_counters()

    def reset_state(self, scope: str = "all") -> dict:
        """State is empty by design (no cache carried across examples)."""
        return {}

    # -- training ------------------------------------------------------

    def train_step(self, batch: Batch, optimizer: torch.optim.Optimizer) -> dict:
        assert self.model is not None
        self.model.train()
        logits = self.model(batch.tokens, batch.pad_mask)
        qlogits = gather_query_logits(logits, batch.query_positions)
        loss = masked_query_loss(qlogits, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.counters["forward_passes"] += 1
        self.counters["backward_passes"] += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        self.counters["updates"] += 1
        self.counters["examples"] += batch.n_examples
        self.counters["tokens"] += int(batch.pad_mask.sum())
        self.counters["supervised_targets"] += int((batch.targets >= 0).sum())
        if self.use_reasoning:
            self.counters["reasoning_loop_iterations"] += (
                self.reasoning_loops * batch.n_examples
            )
        if self.use_memory:
            n_chunks = math.ceil(int(batch.tokens.shape[1]) / 16)
            self.counters["memory_chunks"] += n_chunks * self.model.depth * batch.n_examples
        return {
            "loss": float(loss.detach()),
            "grad_norm_preclip": float(grad_norm),
            "counters": dict(self.counters),
        }

    # -- evaluation ----------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: Batch) -> list[torch.Tensor]:
        """Per-example query logits [K, 64]; state is fresh for every call."""
        assert self.model is not None
        self.model.eval()
        self.reset_state("evaluation")
        logits = self.model(batch.tokens, batch.pad_mask)
        self.counters["forward_passes"] += 1
        self.counters["examples"] += batch.n_examples
        qlogits = gather_query_logits(logits, batch.query_positions)
        out = []
        for i, ex in enumerate(batch.examples):
            out.append(qlogits[i, : ex.k].float().cpu())
        return out

    # -- checkpoints ---------------------------------------------------

    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> dict:
        assert self.model is not None
        payload = {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "kind": self.kind,
            "use_memory": self.use_memory,
            "use_reasoning": self.use_reasoning,
            "reasoning_loops": self.reasoning_loops,
            "config": dict(self.config),
            "model": self.model.state_dict(),
            "counters": dict(self.counters),
            "extra": extra or {},
        }
        tmp = Path(str(path) + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return payload

    def load_checkpoint(self, path: str | Path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload["adapter_id"] != self.adapter_id:
            raise ValueError(f"checkpoint is for {payload['adapter_id']}, not {self.adapter_id}")
        if self.model is None:
            self.build(payload["config"], initialization_seed=0)
        self.model.load_state_dict(payload["model"])
        self.counters = payload.get("counters", self._fresh_counters())
        return payload


# ------------------------------------------------------------------ registry


def registered_adapters(source_commit: str = "unknown", device: str = "cpu") -> dict[str, TorchMQARAdapter]:
    """The five starting configurations from spec section 5.

    B0/B1/B2/B3 share one backbone; only the optional components differ, and a
    bypassed component is never constructed (no parameters, no executed work).
    """
    from functools import partial

    defs = {
        "B0": dict(kind="candidate", use_memory=False, use_reasoning=False),
        "B1": dict(kind="candidate", use_memory=True, use_reasoning=False),
        "B2": dict(kind="candidate", use_memory=False, use_reasoning=True),
        "B3": dict(kind="candidate", use_memory=True, use_reasoning=True),
        "T0": dict(kind="t0", use_memory=False, use_reasoning=False),
    }
    return {
        mid: TorchMQARAdapter(
            adapter_id=mid, source_commit=source_commit, device=device, **kw
        )
        for mid, kw in defs.items()
    }
