#!/usr/bin/env python3
"""Build the logo specimen sheet (preview.html) with every asset inlined.

Inlined rather than linked so the sheet renders correctly wherever it is opened,
including inside the chat preview frame.
"""

from __future__ import annotations

import pathlib

OUT = pathlib.Path(__file__).resolve().parent
INK = "#15161A"
PAPER = "#F4F1EA"
ACCENT = "#C8452B"
MID = "#3B3E46"


def svg(name: str, *, height: str | None = None, width: str | None = None) -> str:
    text = (OUT / name).read_text(encoding="utf-8").strip()
    style = ""
    if height:
        style += f"height:{height};"
    if width:
        style += f"width:{width};"
    if style:
        text = text.replace("<svg ", f'<svg style="{style}" ', 1)
    return text


specimens = [
    ("Primary mark", "mark.svg", 148, PAPER, "Full mark: stone plus induced field. Use at 64 px and above."),
    ("Reverse", "mark-reverse.svg", 148, INK, "On ink or photography. Never place the full-colour mark on mid-grey."),
    ("One colour", "mark-mono.svg", 148, PAPER, "Print, stamps, embroidery, any single-ink process."),
    ("Compact", "mark-compact.svg", 148, PAPER, "Field removed. Use at 64 px and below, where arcs would blur."),
]

sizes = "".join(
    f'<div class="size"><div class="s-ico">{svg("mark-compact.svg", height=f"{s}px")}</div>'
    f'<span>{s} px</span></div>'
    for s in (64, 32, 24, 16)
)

swatches = "".join(
    f'<div class="sw"><i style="background:{hexv}"></i><b>{name}</b><code>{hexv}</code></div>'
    for name, hexv in (("Ink", INK), ("Stone", MID), ("Iron oxide", ACCENT), ("Paper", PAPER))
)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Loadstone — logo specimens</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: var(--app-font, ui-sans-serif, system-ui, sans-serif); color: var(--foreground); }}
  .wrap {{ padding: 2px 0 6px; }}
  h1 {{ font-size: 13px; letter-spacing: .16em; text-transform: uppercase; font-weight: 600;
        color: var(--muted-foreground); margin: 0 0 14px; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
  .tile {{ border: 1px solid var(--border); display: flex; flex-direction: column; }}
  .tile .art {{ height: 168px; display: grid; place-items: center; padding: 12px; }}
  .tile .cap {{ padding: 8px 10px 10px; font-size: 11.5px; line-height: 1.45; color: var(--muted-foreground);
                border-top: 1px solid var(--border); }}
  .tile .cap b {{ display: block; color: var(--foreground); font-size: 12px; margin-bottom: 2px; }}
  .row {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 10px; margin-top: 10px; align-items: start; }}
  .lockup {{ border: 1px solid var(--border); padding: 22px; background: {PAPER}; }}
  .lockup svg {{ width: 100%; height: auto; display: block; }}
  .side h2 {{ font-size: 11.5px; letter-spacing: .14em; text-transform: uppercase; margin: 0 0 8px;
              color: var(--muted-foreground); font-weight: 600; }}
  .side p {{ margin: 0 0 14px; font-size: 12px; line-height: 1.55; color: var(--muted-foreground); }}
  .side code {{ font-family: ui-monospace, monospace; font-size: 11px; color: var(--foreground); }}
  .sw {{ display: flex; align-items: center; gap: 8px; font-size: 11.5px; margin-bottom: 6px; }}
  .sw i {{ width: 14px; height: 14px; border: 1px solid var(--border); display: inline-block; }}
  .sw b {{ font-weight: 500; width: 74px; }}
  .sw code {{ color: var(--muted-foreground); }}
  .sizes {{ display: flex; gap: 18px; align-items: flex-end; margin: 0; padding: 14px 22px 0; }}
  .size {{ display: flex; flex-direction: column; align-items: center; gap: 7px; font-size: 10.5px;
           color: var(--muted-foreground); }}
  .s-ico {{ display: flex; align-items: flex-end; }}
  .stack {{ margin-top: 10px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
  .stack img, .stack svg {{ width: 100%; height: auto; }}
  .paper {{ background: {PAPER}; border: 1px solid var(--border); padding: 18px; }}
  .dark {{ background: {INK}; border: 1px solid var(--border); padding: 18px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Loadstone — identity specimens</h1>
  <div class="grid">
    {''.join(
      f'<div class="tile"><div class="art" style="background:{bg}">{svg(src, height="118px")}</div>'
      f'<div class="cap"><b>{title}</b>{note}</div></div>'
      for title, src, bg, note in [(t, s, b, n) for t, s, _h, b, n in specimens]
    )}
  </div>

  <div class="row">
    <div>
      <div class="lockup">{svg("lockup-horizontal.svg", width="100%")}</div>
      <div class="sizes">{sizes}</div>
    </div>
    <div class="side">
      <h2>Palette</h2>
      {swatches}
      <h2 style="margin-top:16px">Rules</h2>
      <p>Clear space on all sides = <code>0.25 ×</code> mark height. Minimum mark height
      <code>16 px</code> (compact) / <code>64 px</code> (full). Minimum lockup width
      <code>120 px</code>. Never restretch, recolour the stone facets, or place the
      wordmark in a different face.</p>
      <p>Wordmark set in Nimbus Sans Bold, outlined, optical tracking <code>0.19 em</code>
      between ink edges.</p>
    </div>
  </div>

  <div class="stack">
    <div class="paper">{svg("lockup-stacked.svg", width="100%")}</div>
    <div class="dark">{svg("mark-reverse.svg", height="180px")}</div>
  </div>
</div>
</body>
</html>
"""

(OUT / "preview.html").write_text(html, encoding="utf-8")
print("wrote", OUT / "preview.html")
