#!/usr/bin/env python3
"""Generate crisp, high-resolution architecture and pipeline diagrams for Loadstone."""

from __future__ import annotations

import pathlib
import subprocess

DOCS_DIR = pathlib.Path(__file__).resolve().parent

INK = "#15161A"       # Graphite / dark text
DARK_BG = "#1E2025"   # Card background dark
MID = "#3B3E46"       # Slate / border
MUTED = "#6B7280"     # Secondary labels
BORDER = "#D1CBC1"    # Neutral light borders
PANEL_BG = "#FDFCF9"  # Panel background
PAPER = "#F4F1EA"     # Canvas background
ACCENT = "#C8452B"    # Iron-oxide red / magnetite (brand accent)
ACCENT_MUTED = "#E86A50"
BLUE = "#2B6CB0"      # Protocol / Gate blue
GREEN = "#2F855A"     # Validated / Approved green
PURPLE = "#6B46C1"    # Model / Runner purple
AMBER = "#D69E2E"     # Telemetry / Checkpoints amber

# 1. System Architecture Diagram SVG
ARCH_SVG = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1100" width="1600" height="1100" style="background:{PAPER}; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.07" />
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="{INK}" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="{ACCENT}" />
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="{BLUE}" />
    </marker>
  </defs>

  <!-- Header Banner -->
  <rect x="0" y="0" width="1600" height="90" fill="{INK}" />
  <!-- Brand Lodestone Mark Accent -->
  <polygon points="50,22 71,33.5 50,45 29,33.5" fill="{ACCENT}"/>
  <polygon points="29,33.5 50,45 50,68 29,56.5" fill="#FFFFFF" fill-opacity="0.9"/>
  <polygon points="71,33.5 50,45 50,68 71,56.5" fill="#888888"/>
  <path d="M24.5 37 A30 30 0 0 1 40.5 19.8" fill="none" stroke="{ACCENT}" stroke-width="3.2"/>
  <path d="M75.5 53 A30 30 0 0 1 59.5 70.2" fill="none" stroke="{ACCENT}" stroke-width="3.2"/>

  <text x="95" y="47" fill="#FFFFFF" font-size="26" font-weight="bold" letter-spacing="0.5">LOADSTONE</text>
  <text x="260" y="47" fill="#A0AEC0" font-size="20" font-weight="normal">| System Architecture Diagram (spec-v0.1)</text>
  <text x="95" y="70" fill="#E2E8F0" font-size="13">Local-First Sequence-Model Benchmark Harness &amp; Verifiable Gate Engine</text>
  <rect x="1410" y="30" width="130" height="30" rx="6" fill="{ACCENT}" />
  <text x="1475" y="50" fill="#FFFFFF" font-size="13" font-weight="bold" text-anchor="middle">MILESTONE 1</text>

  <!-- Section 1: User & Control Plane (Top Left) -->
  <g transform="translate(60, 130)">
    <rect width="440" height="230" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="440" height="42" rx="10" fill="{INK}" />
    <rect y="32" width="440" height="10" fill="{INK}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">1. Control &amp; Interface Surface</text>
    <text x="420" y="27" fill="#CBD5E0" font-size="12" text-anchor="end">CLI &amp; UI Layer</text>

    <!-- Sub-blocks -->
    <rect x="20" y="60" width="190" height="70" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="35" y="85" fill="{INK}" font-size="14" font-weight="bold">Unified CLI</text>
    <text x="35" y="105" fill="{MUTED}" font-size="12">src/loadstone/cli.py</text>
    <text x="35" y="120" fill="{BLUE}" font-size="11">plan, freeze, approve, run</text>

    <rect x="230" y="60" width="190" height="70" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="245" y="85" fill="{INK}" font-size="14" font-weight="bold">Desktop UI / API</text>
    <text x="245" y="105" fill="{MUTED}" font-size="12">Localhost binding</text>
    <text x="245" y="120" fill="{MUTED}" font-size="11">Non-cloud standalone</text>

    <rect x="20" y="145" width="400" height="65" rx="6" fill="#EDF2F7" stroke="{BORDER}" />
    <text x="35" y="170" fill="{INK}" font-size="13" font-weight="bold">Explicit Gate Commands</text>
    <text x="35" y="192" fill="{MID}" font-size="12">loadstone validate (Gates 0, 1) &amp; loadstone overfit (Gate 2)</text>
  </g>

  <!-- Section 2: Protocol & Governance Engine (Top Center) -->
  <g transform="translate(560, 130)">
    <rect width="480" height="230" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="480" height="42" rx="10" fill="{BLUE}" />
    <rect y="32" width="480" height="10" fill="{BLUE}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">2. Protocol Governance &amp; Expansion</text>
    <text x="460" y="27" fill="#EBF8FF" font-size="12" text-anchor="end">Immutable Contracts</text>

    <rect x="20" y="60" width="205" height="70" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="32" y="85" fill="{INK}" font-size="13" font-weight="bold">Protocol Resolver</text>
    <text x="32" y="103" fill="{MUTED}" font-size="11">Validate schemas</text>
    <text x="32" y="118" fill="{BLUE}" font-size="11">protocols/mqar-v1.json</text>

    <rect x="245" y="60" width="215" height="70" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="257" y="85" fill="{INK}" font-size="13" font-weight="bold">Matrix Expander</text>
    <text x="257" y="103" fill="{MUTED}" font-size="11">5 models × 3 seeds × 3 lrs</text>
    <text x="257" y="118" fill="{ACCENT}" font-size="11">Full 60-job grid view</text>

    <rect x="20" y="145" width="440" height="65" rx="6" fill="#EBF8FF" stroke="{BLUE}" stroke-width="1.2" />
    <text x="35" y="170" fill="{BLUE}" font-size="13" font-weight="bold">Cryptographic Freeze &amp; Spend Approval</text>
    <text x="35" y="192" fill="{INK}" font-size="12">SHA-256 protocol hash · Strict hour/FLOP cost ceiling budget</text>
  </g>

  <!-- Section 3: Verification & Preflight Gates (Top Right) -->
  <g transform="translate(1100, 130)">
    <rect width="440" height="230" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="440" height="42" rx="10" fill="{GREEN}" />
    <rect y="32" width="440" height="10" fill="{GREEN}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">3. Verification &amp; Gate Engine</text>
    <text x="420" y="27" fill="#F0FFF4" font-size="12" text-anchor="end">Preflight Enforcers</text>

    <rect x="20" y="58" width="120" height="75" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="30" y="80" fill="{GREEN}" font-size="12" font-weight="bold">Gate 0: Data</text>
    <text x="30" y="98" fill="{MUTED}" font-size="11">Oracle solver</text>
    <text x="30" y="113" fill="{MUTED}" font-size="11">Pure scorer</text>
    <text x="30" y="126" fill="{INK}" font-size="10">Exact fixtures</text>

    <rect x="155" y="58" width="130" height="75" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="165" y="80" fill="{GREEN}" font-size="12" font-weight="bold">Gate 1: Adapter</text>
    <text x="165" y="98" fill="{MUTED}" font-size="11">Shape contract</text>
    <text x="165" y="113" fill="{MUTED}" font-size="11">State ownership</text>
    <text x="165" y="126" fill="{INK}" font-size="10">Size search</text>

    <rect x="300" y="58" width="120" height="75" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="310" y="80" fill="{GREEN}" font-size="12" font-weight="bold">Gate 2: Learn</text>
    <text x="310" y="98" fill="{MUTED}" font-size="11">Overfit sweep</text>
    <text x="310" y="113" fill="{MUTED}" font-size="11">≥95% accuracy</text>
    <text x="310" y="126" fill="{INK}" font-size="10">Plumbing check</text>

    <rect x="20" y="145" width="400" height="65" rx="6" fill="#F0FFF4" stroke="{GREEN}" />
    <text x="35" y="170" fill="{GREEN}" font-size="13" font-weight="bold">Preflight Hardware Enforcer</text>
    <text x="35" y="192" fill="{INK}" font-size="12">Checks VRAM, CUDA device availability, disk space, limits</text>
  </g>

  <!-- Connectors between Top Row -->
  <line x1="500" y1="245" x2="560" y2="245" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />
  <line x1="1040" y1="245" x2="1100" y2="245" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Section 4: Data Engine (Middle Left) -->
  <g transform="translate(60, 420)">
    <rect width="440" height="300" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="440" height="42" rx="10" fill="{MID}" />
    <rect y="32" width="440" height="10" fill="{MID}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">4. Dataset &amp; Split Subsystem</text>
    <text x="420" y="27" fill="#CBD5E0" font-size="12" text-anchor="end">datasets/mqar.py</text>

    <rect x="20" y="60" width="400" height="70" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="35" y="85" fill="{INK}" font-size="14" font-weight="bold">MQAR Generator &amp; Oracle</text>
    <text x="35" y="105" fill="{MUTED}" font-size="12">Alphabet: PAD=0, QUERY=1, END=2, Keys 3–66, Values 67–130</text>
    <text x="35" y="120" fill="{INK}" font-size="12">Gold values never appear in input stream; chance = 1/64</text>

    <rect x="20" y="145" width="400" height="70" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="35" y="170" fill="{INK}" font-size="14" font-weight="bold">Split Materialisation &amp; Locking</text>
    <text x="35" y="190" fill="{MUTED}" font-size="12">9 In-distribution cells (L ∈ {128,256,512} × K ∈ {4,8,16})</text>
    <text x="35" y="205" fill="{ACCENT}" font-size="12">SHA-256 dataset-manifest.json locks held-out splits</text>

    <rect x="20" y="230" width="400" height="50" rx="6" fill="#F7FAFC" stroke="{BORDER}" />
    <text x="35" y="260" fill="{INK}" font-size="12" font-weight="bold">Validation (512/cell) &amp; Confirmation Test Splits</text>
  </g>

  <!-- Section 5: Architecture Adapters & Model Subsystem (Middle Center) -->
  <g transform="translate(560, 420)">
    <rect width="480" height="300" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="480" height="42" rx="10" fill="{PURPLE}" />
    <rect y="32" width="480" height="10" fill="{PURPLE}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">5. Model Adapter Boundary &amp; Ablations</text>
    <text x="460" y="27" fill="#E9D8FD" font-size="12" text-anchor="end">models/ &amp; blocks.py</text>

    <rect x="20" y="60" width="440" height="75" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="35" y="82" fill="{INK}" font-size="13" font-weight="bold">Adapter Contract &amp; Capability Declarations</text>
    <text x="35" y="100" fill="{MUTED}" font-size="11">metadata(), build(), encode(), train_step(), predict(), reset_state()</text>
    <text x="35" y="118" fill="{PURPLE}" font-size="11">Unsupported cells marked explicitly (NEVER recorded as zero)</text>

    <!-- Model Grid -->
    <g transform="translate(20, 150)">
      <rect x="0" y="0" width="80" height="65" rx="5" fill="#FAF5FF" stroke="{PURPLE}" />
      <text x="40" y="24" fill="{PURPLE}" font-size="14" font-weight="bold" text-anchor="middle">T0</text>
      <text x="40" y="42" fill="{INK}" font-size="10" text-anchor="middle">Transformer</text>
      <text x="40" y="55" fill="{MUTED}" font-size="9" text-anchor="middle">Baseline</text>

      <rect x="90" y="0" width="80" height="65" rx="5" fill="#FAF5FF" stroke="{PURPLE}" />
      <text x="130" y="24" fill="{PURPLE}" font-size="14" font-weight="bold" text-anchor="middle">B0</text>
      <text x="130" y="42" fill="{INK}" font-size="10" text-anchor="middle">Candidate</text>
      <text x="130" y="55" fill="{MUTED}" font-size="9" text-anchor="middle">Core Backb.</text>

      <rect x="180" y="0" width="80" height="65" rx="5" fill="#FAF5FF" stroke="{PURPLE}" />
      <text x="220" y="24" fill="{PURPLE}" font-size="14" font-weight="bold" text-anchor="middle">B1</text>
      <text x="220" y="42" fill="{INK}" font-size="10" text-anchor="middle">+ Memory</text>
      <text x="220" y="55" fill="{MUTED}" font-size="9" text-anchor="middle">Chunk LinAttn</text>

      <rect x="270" y="0" width="80" height="65" rx="5" fill="#FAF5FF" stroke="{PURPLE}" />
      <text x="310" y="24" fill="{PURPLE}" font-size="14" font-weight="bold" text-anchor="middle">B2</text>
      <text x="310" y="42" fill="{INK}" font-size="10" text-anchor="middle">+ Reason</text>
      <text x="310" y="55" fill="{MUTED}" font-size="9" text-anchor="middle">Loop Block</text>

      <rect x="360" y="0" width="80" height="65" rx="5" fill="#FAF5FF" stroke="{ACCENT}" stroke-width="1.5" />
      <text x="400" y="24" fill="{ACCENT}" font-size="14" font-weight="bold" text-anchor="middle">B3</text>
      <text x="400" y="42" fill="{INK}" font-size="10" text-anchor="middle">Full Model</text>
      <text x="400" y="55" fill="{MUTED}" font-size="9" text-anchor="middle">Mem + Reason</text>
    </g>

    <rect x="20" y="230" width="440" height="50" rx="6" fill="#F7FAFC" stroke="{BORDER}" />
    <text x="35" y="250" fill="{INK}" font-size="12" font-weight="bold">Parameter Matched Grid (1M ± 5%)</text>
    <text x="35" y="268" fill="{MUTED}" font-size="11">Expanded width search {112, 144, 224, 288} maintains parity</text>
  </g>

  <!-- Section 6: Execution Engine & Telemetry (Middle Right) -->
  <g transform="translate(1100, 420)">
    <rect width="440" height="300" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="440" height="42" rx="10" fill="{ACCENT}" />
    <rect y="32" width="440" height="10" fill="{ACCENT}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">6. Runner, Budget &amp; Lifecycle</text>
    <text x="420" y="27" fill="#FED7D7" font-size="12" text-anchor="end">runner.py</text>

    <rect x="20" y="60" width="400" height="70" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="35" y="85" fill="{INK}" font-size="13" font-weight="bold">Controlled Training Loop</text>
    <text x="35" y="103" fill="{MUTED}" font-size="11">Effective batch 32 · AdamW · Cosine lr schedule</text>
    <text x="35" y="120" fill="{INK}" font-size="11">Evaluated at step intervals against validation splits</text>

    <rect x="20" y="145" width="190" height="65" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="30" y="170" fill="{ACCENT}" font-size="12" font-weight="bold">Lifecycle Controls</text>
    <text x="30" y="190" fill="{INK}" font-size="11">Pause / Cancel / Resume</text>
    <text x="30" y="202" fill="{MUTED}" font-size="10">control.json signals</text>

    <rect x="230" y="145" width="190" height="65" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="240" y="170" fill="{ACCENT}" font-size="12" font-weight="bold">Checkpoints</text>
    <text x="240" y="190" fill="{INK}" font-size="11">Verified atomic saves</text>
    <text x="240" y="202" fill="{MUTED}" font-size="10">Resumes without leaks</text>

    <rect x="20" y="225" width="400" height="55" rx="6" fill="#FFF5F5" stroke="{ACCENT}" />
    <text x="35" y="247" fill="{ACCENT}" font-size="12" font-weight="bold">Hard Budget Enforcer</text>
    <text x="35" y="265" fill="{INK}" font-size="11">Enforces hour/FLOP limits; aborts if limit reached</text>
  </g>

  <!-- Connectors Middle Row -->
  <line x1="500" y1="570" x2="560" y2="570" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />
  <line x1="1040" y1="570" x2="1100" y2="570" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />
  <line x1="780" y1="360" x2="780" y2="420" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />
  <line x1="1320" y1="360" x2="1320" y2="420" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Section 7: Storage & Append-Only Artifacts (Bottom Left/Center) -->
  <g transform="translate(60, 770)">
    <rect width="700" height="260" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="700" height="42" rx="10" fill="{INK}" />
    <rect y="32" width="700" height="10" fill="{INK}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">7. Self-Describing Artifact Contract &amp; SQLite Store</text>
    <text x="680" y="27" fill="#CBD5E0" font-size="12" text-anchor="end">store.py · studies/&lt;study_id&gt;/</text>

    <!-- Sub-boxes -->
    <rect x="20" y="60" width="320" height="80" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="35" y="85" fill="{INK}" font-size="13" font-weight="bold">Self-Describing Directory Tree</text>
    <text x="35" y="105" fill="{MUTED}" font-size="11">protocol.resolved.json · protocol.sha256</text>
    <text x="35" y="120" fill="{MUTED}" font-size="11">approval.json · environment.json · gates.json</text>
    <text x="35" y="133" fill="{MUTED}" font-size="11">jobs/&lt;job_id&gt;/summary.json, step_metrics.csv</text>

    <rect x="360" y="60" width="320" height="80" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="375" y="85" fill="{INK}" font-size="13" font-weight="bold">SQLite Index (studies/loadstone.db)</text>
    <text x="375" y="105" fill="{MUTED}" font-size="11">Pure query accelerator for large suites</text>
    <text x="375" y="120" fill="{MUTED}" font-size="11">Files stay primary source of truth</text>
    <text x="375" y="133" fill="{BLUE}" font-size="11">Exportable and reproducible without DB</text>

    <rect x="20" y="155" width="660" height="85" rx="6" fill="#EDF2F7" stroke="{BORDER}" />
    <text x="35" y="180" fill="{INK}" font-size="13" font-weight="bold">Verification Invariants</text>
    <text x="35" y="200" fill="{MID}" font-size="12">1. Append-only artifacts ensure zero in-place mutation of past runs</text>
    <text x="35" y="220" fill="{MID}" font-size="12">2. Checkpoint selection strictly based on validation splits; confirmation runs test split once</text>
  </g>

  <!-- Section 8: Analysis & Deterministic Reporting (Bottom Right) -->
  <g transform="translate(800, 770)">
    <rect width="740" height="260" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow)" />
    <rect width="740" height="42" rx="10" fill="{DARK_BG}" />
    <rect y="32" width="740" height="10" fill="{DARK_BG}" />
    <text x="20" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">8. Evaluation, Analysis &amp; Report Exporter</text>
    <text x="720" y="27" fill="#A0AEC0" font-size="12" text-anchor="end">report.py · scoring.py</text>

    <rect x="20" y="60" width="220" height="90" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="35" y="85" fill="{INK}" font-size="13" font-weight="bold">Pure Scorers</text>
    <text x="35" y="105" fill="{MUTED}" font-size="11">Cross-entropy value loss</text>
    <text x="35" y="120" fill="{MUTED}" font-size="11">Exact top-1 accuracy</text>
    <text x="35" y="135" fill="{MUTED}" font-size="11">bits-per-byte (enwik8)</text>

    <rect x="260" y="60" width="220" height="90" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="275" y="85" fill="{INK}" font-size="13" font-weight="bold">Report Exporters</text>
    <text x="275" y="105" fill="{MUTED}" font-size="11">Plain text (report.txt)</text>
    <text x="275" y="120" fill="{MUTED}" font-size="11">JSON summary (report.json)</text>
    <text x="275" y="135" fill="{MUTED}" font-size="11">Tabular CSV (tables/*.csv)</text>

    <rect x="500" y="60" width="220" height="90" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
    <text x="515" y="85" fill="{INK}" font-size="13" font-weight="bold">Standalone Charts</text>
    <text x="515" y="105" fill="{MUTED}" font-size="11">Learning curves (.svg)</text>
    <text x="515" y="120" fill="{MUTED}" font-size="11">Accuracy by length × K</text>
    <text x="515" y="135" fill="{MUTED}" font-size="11">Pareto compute curves</text>

    <rect x="20" y="165" width="700" height="75" rx="6" fill="#FFF5F5" stroke="{ACCENT}" stroke-width="1.2" />
    <text x="35" y="190" fill="{ACCENT}" font-size="13" font-weight="bold">Strict Non-Misleading Aggregation</text>
    <text x="35" y="210" fill="{INK}" font-size="12">Every chart traceable to raw measurements · No collapsed single scores · Separate seed spreads</text>
    <text x="35" y="228" fill="{MUTED}" font-size="11">Reports state explicitly: "censored-not-converged" if job halted by budget caps</text>
  </g>

  <!-- Connectors Bottom Row -->
  <line x1="1260" y1="720" x2="1260" y2="770" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />
  <line x1="800" y1="650" x2="450" y2="770" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" stroke-dasharray="4" />
  <line x1="760" y1="900" x2="800" y2="900" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)" />

</svg>'''

# 2. Benchmark Pipeline Diagram SVG
PIPELINE_SVG = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1250" width="1600" height="1250" style="background:{PAPER}; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="shadow2" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.07" />
    </filter>
    <marker id="pipe-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="{INK}" />
    </marker>
    <marker id="pipe-arrow-accent" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="{ACCENT}" />
    </marker>
  </defs>

  <!-- Header Banner -->
  <rect x="0" y="0" width="1600" height="90" fill="{INK}" />
  <polygon points="50,22 71,33.5 50,45 29,33.5" fill="{ACCENT}"/>
  <polygon points="29,33.5 50,45 50,68 29,56.5" fill="#FFFFFF" fill-opacity="0.9"/>
  <polygon points="71,33.5 50,45 50,68 71,56.5" fill="#888888"/>
  <path d="M24.5 37 A30 30 0 0 1 40.5 19.8" fill="none" stroke="{ACCENT}" stroke-width="3.2"/>
  <path d="M75.5 53 A30 30 0 0 1 59.5 70.2" fill="none" stroke="{ACCENT}" stroke-width="3.2"/>

  <text x="95" y="47" fill="#FFFFFF" font-size="26" font-weight="bold" letter-spacing="0.5">LOADSTONE</text>
  <text x="260" y="47" fill="#A0AEC0" font-size="20" font-weight="normal">| Experiment Execution Pipeline &amp; Lifecycle</text>
  <text x="95" y="70" fill="#E2E8F0" font-size="13">End-to-End Workflow: Specification &#x2192; Verification Gates &#x2192; Run &#x2192; Report</text>
  <rect x="1390" y="30" width="150" height="30" rx="6" fill="{ACCENT}" />
  <text x="1465" y="50" fill="#FFFFFF" font-size="12" font-weight="bold" text-anchor="middle">CONTROL PIPELINE</text>

  <!-- Step 1: Definition & Data Prep -->
  <g transform="translate(60, 120)">
    <rect width="320" height="240" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow2)" />
    <rect width="320" height="38" rx="10" fill="{INK}" />
    <rect y="28" width="320" height="10" fill="{INK}" />
    <circle cx="28" cy="19" r="11" fill="{ACCENT}" />
    <text x="28" y="24" fill="#FFFFFF" font-size="12" font-weight="bold" text-anchor="middle">1</text>
    <text x="48" y="24" fill="#FFFFFF" font-size="14" font-weight="bold">Data &amp; Protocol Definition</text>

    <text x="20" y="65" fill="{MUTED}" font-size="11" font-weight="bold">COMMANDS &amp; TASKS</text>
    <rect x="20" y="75" width="280" height="42" rx="5" fill="#EDF2F7" />
    <text x="30" y="93" fill="{INK}" font-size="11" font-family="monospace">loadstone data build</text>
    <text x="30" y="107" fill="{MUTED}" font-size="10">--out data/mqar-v1 --per-cell 512</text>

    <text x="20" y="135" fill="{MUTED}" font-size="11" font-weight="bold">ACTIVITIES</text>
    <text x="20" y="153" fill="{INK}" font-size="12">&#x2022; Sample synthetic MQAR (256 sym)</text>
    <text x="20" y="171" fill="{INK}" font-size="12">&#x2022; Partition 9 ID cells + OOD cells</text>
    <text x="20" y="189" fill="{INK}" font-size="12">&#x2022; Compute SHA-256 for split files</text>
    <text x="20" y="207" fill="{INK}" font-size="12">&#x2022; Emit dataset-manifest.json</text>

    <rect x="20" y="218" width="280" height="1" fill="{BORDER}" />
  </g>

  <!-- Step 2: Verification Gates -->
  <g transform="translate(440, 120)">
    <rect width="320" height="240" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow2)" />
    <rect width="320" height="38" rx="10" fill="{GREEN}" />
    <rect y="28" width="320" height="10" fill="{GREEN}" />
    <circle cx="28" cy="19" r="11" fill="#FFFFFF" />
    <text x="28" y="24" fill="{GREEN}" font-size="12" font-weight="bold" text-anchor="middle">2</text>
    <text x="48" y="24" fill="#FFFFFF" font-size="14" font-weight="bold">Pre-Execution Gates</text>

    <text x="20" y="65" fill="{MUTED}" font-size="11" font-weight="bold">VALIDATION COMMANDS</text>
    <rect x="20" y="75" width="280" height="42" rx="5" fill="#F0FFF4" stroke="{GREEN}" stroke-width="0.8" />
    <text x="30" y="93" fill="{INK}" font-size="11" font-family="monospace">loadstone validate --stage unit</text>
    <text x="30" y="107" fill="{INK}" font-size="11" font-family="monospace">loadstone overfit --device cuda</text>

    <text x="20" y="135" fill="{MUTED}" font-size="11" font-weight="bold">ENFORCEMENT CHECKS</text>
    <text x="20" y="153" fill="{INK}" font-size="12">&#x2022; Gate 0: Oracle &amp; Scorer fixtures</text>
    <text x="20" y="171" fill="{INK}" font-size="12">&#x2022; Gate 1: Adapter contract &amp; state</text>
    <text x="20" y="189" fill="{INK}" font-size="12">&#x2022; Gate 2: 500-step overfit &#x2265; 95% acc</text>
    <text x="20" y="207" fill="{GREEN}" font-size="12" font-weight="bold">&#x2022; Blocks launch if any gate fails</text>
  </g>

  <!-- Step 3: Protocol Freeze -->
  <g transform="translate(820, 120)">
    <rect width="320" height="240" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow2)" />
    <rect width="320" height="38" rx="10" fill="{BLUE}" />
    <rect y="28" width="320" height="10" fill="{BLUE}" />
    <circle cx="28" cy="19" r="11" fill="#FFFFFF" />
    <text x="28" y="24" fill="{BLUE}" font-size="12" font-weight="bold" text-anchor="middle">3</text>
    <text x="48" y="24" fill="#FFFFFF" font-size="14" font-weight="bold">Protocol Freeze &amp; Lock</text>

    <text x="20" y="65" fill="{MUTED}" font-size="11" font-weight="bold">FREEZE COMMAND</text>
    <rect x="20" y="75" width="280" height="42" rx="5" fill="#EBF8FF" stroke="{BLUE}" stroke-width="0.8" />
    <text x="30" y="93" fill="{INK}" font-size="11" font-family="monospace">loadstone freeze </text>
    <text x="30" y="107" fill="{MUTED}" font-size="10">  --protocol mqar.json --hours 30</text>

    <text x="20" y="135" fill="{MUTED}" font-size="11" font-weight="bold">LOCKED ARTIFACTS</text>
    <text x="20" y="153" fill="{INK}" font-size="12">&#x2022; Resolve seeds, ablations, grids</text>
    <text x="20" y="171" fill="{INK}" font-size="12">&#x2022; Write protocol.resolved.json</text>
    <text x="20" y="189" fill="{INK}" font-size="12">&#x2022; Write SHA-256 protocol.sha256</text>
    <text x="20" y="207" fill="{BLUE}" font-size="12">&#x2022; Unset budget blocks launch</text>
  </g>

  <!-- Step 4: Spend & Plan Approval -->
  <g transform="translate(1200, 120)">
    <rect width="340" height="240" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow2)" />
    <rect width="340" height="38" rx="10" fill="{DARK_BG}" />
    <rect y="28" width="340" height="10" fill="{DARK_BG}" />
    <circle cx="28" cy="19" r="11" fill="{ACCENT}" />
    <text x="28" y="24" fill="#FFFFFF" font-size="12" font-weight="bold" text-anchor="middle">4</text>
    <text x="48" y="24" fill="#FFFFFF" font-size="14" font-weight="bold">Spend &amp; Matrix Approval</text>

    <text x="20" y="65" fill="{MUTED}" font-size="11" font-weight="bold">APPROVAL COMMAND</text>
    <rect x="20" y="75" width="300" height="42" rx="5" fill="#EDF2F7" />
    <text x="30" y="93" fill="{INK}" font-size="11" font-family="monospace">loadstone approve </text>
    <text x="30" y="107" fill="{MUTED}" font-size="10">  --study mqar-study-1 --yes</text>

    <text x="20" y="135" fill="{MUTED}" font-size="11" font-weight="bold">HUMAN-IN-THE-LOOP SAFETY</text>
    <text x="20" y="153" fill="{INK}" font-size="12">&#x2022; Print full expanded job matrix</text>
    <text x="20" y="171" fill="{INK}" font-size="12">&#x2022; Display maximum hours / spend cap</text>
    <text x="20" y="189" fill="{INK}" font-size="12">&#x2022; Emit signed approval.json</text>
    <text x="20" y="207" fill="{ACCENT}" font-size="12" font-weight="bold">&#x2022; Mandatory authorization step</text>
  </g>

  <!-- Connectors Step 1 -> 2 -> 3 -> 4 -->
  <line x1="380" y1="240" x2="440" y2="240" stroke="{INK}" stroke-width="2" marker-end="url(#pipe-arrow)" />
  <line x1="760" y1="240" x2="820" y2="240" stroke="{INK}" stroke-width="2" marker-end="url(#pipe-arrow)" />
  <line x1="1140" y1="240" x2="1200" y2="240" stroke="{INK}" stroke-width="2" marker-end="url(#pipe-arrow)" />

  <!-- Flow down to Execution Phase -->
  <path d="M 1370 360 L 1370 410 L 800 410 L 800 460" fill="none" stroke="{ACCENT}" stroke-width="2.5" marker-end="url(#pipe-arrow-accent)" />

  <!-- Central Box: Controlled Execution Engine -->
  <g transform="translate(60, 470)">
    <rect width="1480" height="390" rx="10" fill="{PANEL_BG}" stroke="{ACCENT}" stroke-width="2" filter="url(#shadow2)" />
    <rect width="1480" height="44" rx="10" fill="{ACCENT}" />
    <rect y="34" width="1480" height="10" fill="{ACCENT}" />
    <circle cx="30" cy="22" r="12" fill="#FFFFFF" />
    <text x="30" y="27" fill="{ACCENT}" font-size="13" font-weight="bold" text-anchor="middle">5</text>
    <text x="52" y="27" fill="#FFFFFF" font-size="16" font-weight="bold">Controlled Execution Engine &amp; Job Lifecycle Loop</text>
    <text x="1450" y="27" fill="#FED7D7" font-size="13" text-anchor="end">loadstone run --frozen-protocol &lt;path&gt; --approval &lt;path&gt;</text>

    <!-- Sub-stages inside Runner -->
    <!-- 5A: Preflight -->
    <g transform="translate(30, 65)">
      <rect width="260" height="300" rx="8" fill="#FFFFFF" stroke="{BORDER}" />
      <rect width="260" height="32" rx="8" fill="{INK}" />
      <rect y="24" width="260" height="8" fill="{INK}" />
      <text x="130" y="21" fill="#FFFFFF" font-size="13" font-weight="bold" text-anchor="middle">5.1 Preflight &amp; Lock</text>

      <text x="15" y="55" fill="{INK}" font-size="12" font-weight="bold">Hash Verification</text>
      <text x="15" y="72" fill="{MUTED}" font-size="11">Re-verify protocol SHA-256</text>
      <text x="15" y="87" fill="{MUTED}" font-size="11">Confirm approval match</text>

      <text x="15" y="115" fill="{INK}" font-size="12" font-weight="bold">Resource Check</text>
      <text x="15" y="132" fill="{MUTED}" font-size="11">Inspect CUDA memory (VRAM)</text>
      <text x="15" y="147" fill="{MUTED}" font-size="11">Check GPU contention</text>
      <text x="15" y="162" fill="{MUTED}" font-size="11">Refuses run if device occupied</text>

      <text x="15" y="190" fill="{INK}" font-size="12" font-weight="bold">Environment Snapshot</text>
      <text x="15" y="207" fill="{MUTED}" font-size="11">Record PyTorch / CUDA vers</text>
      <text x="15" y="222" fill="{MUTED}" font-size="11">Record git HEAD commit</text>
      <text x="15" y="237" fill="{MUTED}" font-size="11">Emit environment.json</text>

      <rect x="15" y="255" width="230" height="30" rx="4" fill="#F7FAFC" stroke="{BORDER}" />
      <text x="130" y="275" fill="{GREEN}" font-size="11" font-weight="bold" text-anchor="middle">&#x2714; Ready for Launch</text>
    </g>

    <!-- 5B: Job Dispatch & Scheduling -->
    <g transform="translate(320, 65)">
      <rect width="260" height="300" rx="8" fill="#FFFFFF" stroke="{BORDER}" />
      <rect width="260" height="32" rx="8" fill="{BLUE}" />
      <rect y="24" width="260" height="8" fill="{BLUE}" />
      <text x="130" y="21" fill="#FFFFFF" font-size="13" font-weight="bold" text-anchor="middle">5.2 Job Matrix Queue</text>

      <text x="15" y="55" fill="{INK}" font-size="12" font-weight="bold">Sequential Dispatcher</text>
      <text x="15" y="72" fill="{MUTED}" font-size="11">Ordered jobs j000..j059</text>
      <text x="15" y="87" fill="{MUTED}" font-size="11">5 models &#x00D7; 3 seeds &#x00D7; 3 lrs</text>

      <text x="15" y="115" fill="{INK}" font-size="12" font-weight="bold">Hard Budget Tracking</text>
      <text x="15" y="132" fill="{MUTED}" font-size="11">Track wall-clock &amp; FLOPs</text>
      <text x="15" y="147" fill="{ACCENT}" font-size="11">Abort if allocation exceeded</text>

      <text x="15" y="175" fill="{INK}" font-size="12" font-weight="bold">Live Control Channel</text>
      <text x="15" y="192" fill="{MUTED}" font-size="11">Listens for control.json</text>
      <text x="15" y="207" fill="{MUTED}" font-size="11">Safe Pause / Resume / Cancel</text>

      <rect x="15" y="245" width="230" height="40" rx="4" fill="#EBF8FF" stroke="{BLUE}" />
      <text x="130" y="263" fill="{BLUE}" font-size="11" font-weight="bold" text-anchor="middle">Zero Unchecked Runs</text>
      <text x="130" y="277" fill="{MUTED}" font-size="10" text-anchor="middle">No silent overruns</text>
    </g>

    <!-- 5C: Step Training Loop -->
    <g transform="translate(610, 65)">
      <rect width="270" height="300" rx="8" fill="#FFFFFF" stroke="{BORDER}" />
      <rect width="270" height="32" rx="8" fill="{PURPLE}" />
      <rect y="24" width="270" height="8" fill="{PURPLE}" />
      <text x="135" y="21" fill="#FFFFFF" font-size="13" font-weight="bold" text-anchor="middle">5.3 Step Optimization Loop</text>

      <text x="15" y="55" fill="{INK}" font-size="12" font-weight="bold">Dynamic Batching</text>
      <text x="15" y="72" fill="{MUTED}" font-size="11">Effective batch 32</text>
      <text x="15" y="87" fill="{MUTED}" font-size="11">Pad to longest sequence in batch</text>

      <text x="15" y="115" fill="{INK}" font-size="12" font-weight="bold">Loss &amp; Gradient Step</text>
      <text x="15" y="132" fill="{MUTED}" font-size="11">Model predict() &#x2192; cross-entropy</text>
      <text x="15" y="147" fill="{MUTED}" font-size="11">Gradient clipping + AdamW</text>
      <text x="15" y="162" fill="{MUTED}" font-size="11">Cosine schedule with warmup</text>

      <text x="15" y="190" fill="{INK}" font-size="12" font-weight="bold">Streamed Telemetry</text>
      <text x="15" y="207" fill="{MUTED}" font-size="11">Step loss, grad norm, FLOPs</text>
      <text x="15" y="222" fill="{MUTED}" font-size="11">Append to step_metrics.csv</text>

      <rect x="15" y="250" width="240" height="35" rx="4" fill="#FAF5FF" stroke="{PURPLE}" />
      <text x="135" y="272" fill="{PURPLE}" font-size="11" font-weight="bold" text-anchor="middle">Self-Describing Artifacts</text>
    </g>

    <!-- 5D: Evaluation & Checkpointing -->
    <g transform="translate(910, 65)">
      <rect width="260" height="300" rx="8" fill="#FFFFFF" stroke="{BORDER}" />
      <rect width="260" height="32" rx="8" fill="{DARK_BG}" />
      <rect y="24" width="260" height="8" fill="{DARK_BG}" />
      <text x="130" y="21" fill="#FFFFFF" font-size="13" font-weight="bold" text-anchor="middle">5.4 Eval &amp; Checkpoints</text>

      <text x="15" y="55" fill="{INK}" font-size="12" font-weight="bold">Periodic Validation</text>
      <text x="15" y="72" fill="{MUTED}" font-size="11">Evaluates 9 validation splits</text>
      <text x="15" y="87" fill="{MUTED}" font-size="11">Tracks best validation accuracy</text>

      <text x="15" y="115" fill="{INK}" font-size="12" font-weight="bold">Checkpoint Manager</text>
      <text x="15" y="132" fill="{MUTED}" font-size="11">Writes model state + optimizer</text>
      <text x="15" y="147" fill="{MUTED}" font-size="11">Atomic write to checkpoints/</text>

      <text x="15" y="175" fill="{INK}" font-size="12" font-weight="bold">Halting Logic</text>
      <text x="15" y="192" fill="{MUTED}" font-size="11">Max updates reached</text>
      <text x="15" y="207" fill="{MUTED}" font-size="11">OR spend cap reached</text>
      <text x="15" y="222" fill="{MUTED}" font-size="11">OR operator pause</text>

      <rect x="15" y="250" width="230" height="35" rx="4" fill="#EDF2F7" />
      <text x="130" y="272" fill="{INK}" font-size="11" font-weight="bold" text-anchor="middle">Selection on Validation</text>
    </g>

    <!-- 5E: Test Evaluation Phase -->
    <g transform="translate(1200, 65)">
      <rect width="250" height="300" rx="8" fill="#FFFFFF" stroke="{BORDER}" />
      <rect width="250" height="32" rx="8" fill="{ACCENT}" />
      <rect y="24" width="250" height="8" fill="{ACCENT}" />
      <text x="125" y="21" fill="#FFFFFF" font-size="13" font-weight="bold" text-anchor="middle">5.5 Confirmation Test</text>

      <text x="15" y="55" fill="{ACCENT}" font-size="12" font-weight="bold">Held-Out Test Scoring</text>
      <text x="15" y="72" fill="{MUTED}" font-size="11">Executed ONCE per model</text>
      <text x="15" y="87" fill="{MUTED}" font-size="11">Uses selected best checkpoint</text>

      <text x="15" y="115" fill="{INK}" font-size="12" font-weight="bold">Strict Rules</text>
      <text x="15" y="132" fill="{MUTED}" font-size="11">Smoke runs NEVER touch test</text>
      <text x="15" y="147" fill="{MUTED}" font-size="11">No training on test answers</text>
      <text x="15" y="162" fill="{MUTED}" font-size="11">512 examples per cell</text>

      <text x="15" y="190" fill="{INK}" font-size="12" font-weight="bold">Outputs</text>
      <text x="15" y="207" fill="{MUTED}" font-size="11">Final test accuracy by (L, K)</text>
      <text x="15" y="222" fill="{MUTED}" font-size="11">Length OOD performance</text>

      <rect x="15" y="250" width="220" height="35" rx="4" fill="#FFF5F5" stroke="{ACCENT}" />
      <text x="125" y="272" fill="{ACCENT}" font-size="11" font-weight="bold" text-anchor="middle">Unbiased Generalization</text>
    </g>

    <!-- Internal Connectors in Runner -->
    <line x1="290" y1="215" x2="320" y2="215" stroke="{INK}" stroke-width="1.5" marker-end="url(#pipe-arrow)" />
    <line x1="580" y1="215" x2="610" y2="215" stroke="{INK}" stroke-width="1.5" marker-end="url(#pipe-arrow)" />
    <line x1="880" y1="215" x2="910" y2="215" stroke="{INK}" stroke-width="1.5" marker-end="url(#pipe-arrow)" />
    <line x1="1170" y1="215" x2="1200" y2="215" stroke="{INK}" stroke-width="1.5" marker-end="url(#pipe-arrow)" />
  </g>

  <!-- Connect Down to Step 6 -->
  <line x1="800" y1="860" x2="800" y2="920" stroke="{INK}" stroke-width="2.5" marker-end="url(#pipe-arrow)" />

  <!-- Step 6: Indexing, Synthesis & Reports -->
  <g transform="translate(60, 920)">
    <rect width="1480" height="260" rx="10" fill="{PANEL_BG}" stroke="{BORDER}" stroke-width="1.5" filter="url(#shadow2)" />
    <rect width="1480" height="40" rx="10" fill="{INK}" />
    <rect y="30" width="1480" height="10" fill="{INK}" />
    <circle cx="30" cy="20" r="12" fill="{ACCENT}" />
    <text x="30" y="25" fill="#FFFFFF" font-size="13" font-weight="bold" text-anchor="middle">6</text>
    <text x="52" y="25" fill="#FFFFFF" font-size="16" font-weight="bold">Synthesis, Verification &amp; Deterministic Reporting</text>
    <text x="1450" y="25" fill="#A0AEC0" font-size="13" text-anchor="end">loadstone report --study &lt;study_dir&gt; --format txt,csv,json</text>

    <!-- 3 columns -->
    <g transform="translate(30, 60)">
      <rect width="440" height="175" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
      <text x="20" y="30" fill="{INK}" font-size="14" font-weight="bold">SQLite Accelerated Indexing</text>
      <text x="20" y="55" fill="{MUTED}" font-size="12">&#x2022; Ingests self-describing job summary.json files</text>
      <text x="20" y="75" fill="{MUTED}" font-size="12">&#x2022; Builds fast lookup index in loadstone.db</text>
      <text x="20" y="95" fill="{MUTED}" font-size="12">&#x2022; Enables SQL querying across runs and sweeps</text>
      <text x="20" y="115" fill="{BLUE}" font-size="12">&#x2022; Optional: Harness runs seamlessly without SQLite</text>
      <rect x="20" y="130" width="400" height="30" rx="4" fill="#EDF2F7" />
      <text x="220" y="150" fill="{INK}" font-size="11" text-anchor="middle">Append-Only Integrity Guarantee</text>
    </g>

    <g transform="translate(520, 60)">
      <rect width="440" height="175" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
      <text x="20" y="30" fill="{INK}" font-size="14" font-weight="bold">Statistical Aggregation</text>
      <text x="20" y="55" fill="{MUTED}" font-size="12">&#x2022; Reports every seed separately (no hidden variance)</text>
      <text x="20" y="75" fill="{MUTED}" font-size="12">&#x2022; Compares ablations at selected learning rate</text>
      <text x="20" y="95" fill="{MUTED}" font-size="12">&#x2022; Separates B0-T0 plumbing from architectural gains</text>
      <text x="20" y="115" fill="{ACCENT}" font-size="12">&#x2022; Flags "censored-not-converged" if budget stopped</text>
      <rect x="20" y="130" width="400" height="30" rx="4" fill="#FFF5F5" stroke="{ACCENT}" />
      <text x="220" y="150" fill="{ACCENT}" font-size="11" font-weight="bold" text-anchor="middle">Non-Misleading Science Discipline</text>
    </g>

    <g transform="translate(1010, 60)">
      <rect width="440" height="175" rx="6" fill="#FFFFFF" stroke="{BORDER}" />
      <text x="20" y="30" fill="{INK}" font-size="14" font-weight="bold">Deterministic Artifact Generation</text>
      <text x="20" y="55" fill="{INK}" font-size="12">&#x2022; <tspan font-family="monospace">report.txt</tspan>: Human-readable summary</text>
      <text x="20" y="75" fill="{INK}" font-size="12">&#x2022; <tspan font-family="monospace">report.json</tspan>: Machine-readable evaluation bundle</text>
      <text x="20" y="95" fill="{INK}" font-size="12">&#x2022; <tspan font-family="monospace">tables/*.csv</tspan>: Full matrices per model &amp; cell</text>
      <text x="20" y="115" fill="{INK}" font-size="12">&#x2022; <tspan font-family="monospace">charts/*.svg</tspan>: Reproducible loss &amp; Pareto curves</text>
      <rect x="20" y="130" width="400" height="30" rx="4" fill="#F0FFF4" stroke="{GREEN}" />
      <text x="220" y="150" fill="{GREEN}" font-size="11" font-weight="bold" text-anchor="middle">&#x2714; Complete Traceability to Raw Data</text>
    </g>
  </g>

</svg>'''


def main():
    arch_svg_path = DOCS_DIR / "architecture_diagram.svg"
    arch_png_path = DOCS_DIR / "architecture_diagram.png"
    pipe_svg_path = DOCS_DIR / "pipeline_diagram.svg"
    pipe_png_path = DOCS_DIR / "pipeline_diagram.png"

    print("Writing architecture SVG...")
    arch_svg_path.write_text(ARCH_SVG, encoding="utf-8")
    print(f"Generated {arch_svg_path}")

    print("Rendering architecture PNG...")
    subprocess.run(
        ["rsvg-convert", "-w", "2400", "-f", "png", "-o", str(arch_png_path), str(arch_svg_path)],
        check=True,
    )
    print(f"Generated {arch_png_path}")

    print("Writing pipeline SVG...")
    pipe_svg_path.write_text(PIPELINE_SVG, encoding="utf-8")
    print(f"Generated {pipe_svg_path}")

    print("Rendering pipeline PNG...")
    subprocess.run(
        ["rsvg-convert", "-w", "2400", "-f", "png", "-o", str(pipe_png_path), str(pipe_svg_path)],
        check=True,
    )
    print(f"Generated {pipe_png_path}")


if __name__ == "__main__":
    main()
