# Review of `loadstone-architecture-benchmark-spec-v0.1`

Date of review: 12 November 2026. Scope: the specification as written, and the
first implementation built from it (Milestone 1, MQAR).

## Verdict

The specification is implementable as written for the MQAR target, and its
discipline is the strongest part: gates before claims, censored-not-converged,
capability matrix with `UNSUPPORTED` cells, immutable hashed protocols, and an
artifact contract that makes every chart traceable to a row.

Three structural gaps showed up while implementing it, all of them in the
comparison design rather than the task design:

1. **B0 and T0 are the same network** when the candidate's only optional
   components are memory and reasoning. Their parameter counts match to the
   parameter (981,600 here), and with the same seed they produce the same loss
   curve. `B0 − T0` therefore measures plumbing, not architecture.
2. **The parameter-matched band is unreachable on the baseline grid** for
   memory-enabled cells. The memory module costs `4w² + w + 1` per block, so the
   closest baseline-grid configuration for B1 was 917,280 parameters, 8.3 %
   below the 1 M target against a ±5 % band. Left as-is, the
   `parameter_matched` regime would have silently become a mismatch study.
3. **Learning rate interacts strongly with branch structure.** In the 500-update
   overfit gate every model reaches 100 % on the fixed micro-dataset at lr 1e-4
   and 3e-4, but at 1e-3 B1 stalls at 89 % and B3 at 98 % while B0/B2/T0 stay at
   100 %. The failure is an optimisation artefact of one rate, and any protocol
   that screened a single shared rate would have reported it as an architectural
   difference.

## Where the spec is precise enough to implement without guessing

* The MQAR alphabet, symbol ranges, pair placement rule, query placement and the
  scored position are all unambiguous.
* The enwik8 split offsets, the bits-per-byte formula and the "byte NLL only"
  rule.
* The artifact directory contract, run states, and the requirement that
  unsupported cells are never scored zero.
* The statistics rules: report every seed and cell, do not treat query positions
  as independent training runs.

## Decisions taken where the text left a choice

| # | Spec text | Ambiguity | Decision | Where |
|---|---|---|---|---|
| 1 | "Fill other positions with independently sampled distractors" | which positions, and whether distractors can sit in the query block | every first-half position not occupied by a key/value pair is a distractor; the last `2K` positions are exclusively the query block | `datasets/mqar.py` |
| 2 | "untied input/output embeddings" plus "a task head over the 64 possible values" | is the output embedding a 256-way LM head or the 64-way task head | input embedding `256×w`, readout `w×64`; both counted. A 256-way head would let the vocabulary dominate the 1 M target | `models/__init__.py::analytic_param_count` |
| 3 | "B0: candidate backbone with optional memory and latent reasoning bypassed" | what B0 is when those are the only optional parts | B0 is structurally identical to T0 here, and the report says so; the science lives in B1/B2/B3 vs B0 | `models/__init__.py::registered_adapters`, `report.py` |
| 4 | "A candidate adapter must export an equally finite valid size-search space" | whether the candidate may use a finer grid than T0 | T0 keeps the spec grid exactly; the candidate adds widths 112/144/224/288 so the ±5 % band is reachable for every ablation | `models/__init__.py::CANDIDATE_WIDTH_GRID` |
| 5 | fixed vs adaptive reasoning | which loop budget the MVP uses | fixed `loops=2` for every reasoning-enabled cell; the fixed/adaptive sweep is Milestone 3, as the spec scopes it to checkpoints whose enforcement is proven | `models/blocks.py::SequenceModel` |
| 6 | memory mechanism | none specified | chunkwise linear attention (fast-weight state) with a learned residual scale initialised at 0.1; the scale was added after a measured learning-plumbing failure and is recorded in the effective config | `models/blocks.py::LinearMemory` |
| 7 | "The default cumulative study budget is unset, which blocks launch" | how a user sets it | `freeze` refuses with an unset limit; `smoke`/`run` require an explicit confirmation and an approval record whose protocol hash must match | `protocol.py::freeze`, `cli.py` |
| 8 | test evaluation after selection | when test is scored | only in the `confirmation` stage, after validation-based checkpoint selection; smoke never touches test and reports `n/a` | `runner.py::run_job` |
| 9 | Gate 2 "at least 95 % primary accuracy" | at which learning rate | the gate sweeps all three screening rates and passes if any reaches 95 %, recording which. All five models pass (100 % at 1e-4 and 3e-4; B1 89 % and B3 98 % at 1e-3) | `gates.py::overfit_gate` |
| 10 | validation/test set sizes | how much is scored per cell | 512 per cell for confirmation, and the actual number is written into the resolved limits; the smoke study used 64 and says so | `protocols/local-limits.json`, `runner.py::Limits` |

## Findings and risks, in order of how much they could mislead

1. **Learning-rate × branch interaction** (measured, above: B1 89 % and B3 98 %
   at 1e-3 versus 100 % at 1e-4/3e-4). Per-model screening is already required by
   the spec; the extra consequence is that ablation claims must be made at each
   model's selected rate, and the spread across rates must be reported rather
   than averaged away.
2. **Three seeds are a pilot, not a claim.** Evaluation noise per cell is small
   (512 examples × K queries), but training-seed spread dominates. A 95 %
   t-interval on three seeds is very wide. Recommendation: five seeds for
   confirmation before any architecture claim.
3. **Wall-clock "compute matched" is not reproducible on a shared desktop.**
   The first smoke attempt here was refused by resource preflight because another
   project on the same machine held 13.9 GiB of a 16 GiB card. That is the
   correct behaviour, but it shows the fragility: matching accelerator time
   across architectures needs fixed clocks/power limits and a recorded occupancy
   policy, or a FLOP-ceiling comparison instead.
4. **Padding tax in throughput comparisons.** Effective batch 32 with mixed cell
   lengths means each batch pads to its longest example, so measured time
   depends on the draw order. Fine for quality, misleading for
   compute-to-quality. Recommendation: stratified per-cell draws if compute
   matching matters, recorded as a protocol change.
5. **`worst_case_budget` is a ceiling, not an estimate.** It sums per-stage time
   caps, so an approved study usually consumes far less. Reports must not present
   it as expected cost.
6. **OOD lengths conflate two things.** RoPE extrapolation beyond trained lengths
   is not a neutral probe of memory; the OOD table needs a position-scheme
   control before it can carry an interpretation.
7. **Distractor repeats are common** (125 filler symbols over up to 512
   positions). This is fine for the intended induction-style probe, since keys
   are disjoint symbols, but it means "recall across distractions" is not the
   same claim as "recall across many distinct associations".
8. **The 10 GiB artifact cap is enforced after the fact.** Real jobs here produce
   tens of megabytes; a hard guarantee needs a preallocated reservation, which
   the spec asks for and this implementation does not yet do.
9. **Unspecified architecture constants** (attention window, RoPE base, dropout,
   QK norm) are chosen once, recorded in `model-effective-config.json`, and never
   varied per model. A future candidate that changes them will show up as a
   protocol difference rather than a silent advantage.
10. **Gate 2's "95 % on 16 examples" is weak evidence of capability** and was
    treated accordingly: it caught a real optimisation pathology here, but it is
    never reported as a score.

## Protocol changes recommended before the first controlled study (Milestone 2)

* Five training seeds for confirmation (three for screening).
* Either implement compute-matched properly (fixed clock policy, recorded
  occupancy, FLOP ceiling with ≤5 % overshoot) or drop it from the MVP claim.
* Report per-model selected learning rates and the across-rate spread side by
  side with the ablation table.
* Add a position-scheme control baseline before reading any OOD cell.
* Keep the per-cell draw stratification change as an explicit protocol revision
  if compute-to-quality curves are going to be published.

## What this tree covers

Milestone 1 only: MQAR generator/oracle/scorer with Gate 0 evidence, T0 plus the
B0–B3 adapters with Gate 1 mechanics evidence, a bounded smoke run plus the
learning-plumbing gate, cancellation/pause/resume, budget accounting, the
artifact contract and the report/table/chart export.

Not implemented, and registered as such so nothing can substitute for them: S5,
Rule110, decimal addition, enwik8 bytes, the adaptive-halting sweep, and the
desktop UI. Every one of those raises `NotImplementedError` from the registry
rather than planning a different task.
