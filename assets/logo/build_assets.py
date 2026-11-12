#!/usr/bin/env python3
"""Build the Loadstone logo assets from one source of truth.

The mark is a lodestone: an isometric stone block lit from above (accent top
facet) with two segments of an induced field around it — the same 180°-rotational
pairing the benchmark itself uses (B vs T). The wordmark is drawn as outlines, so
no font is required to open the SVGs.

Run:  .venv/bin/python assets/logo/build_assets.py
Needs fonttools (asset-time only, not a runtime dependency).
"""

from __future__ import annotations

import pathlib
import subprocess

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

OUT = pathlib.Path(__file__).resolve().parent
INK = "#15161A"       # graphite
MID = "#3B3E46"       # second stone facet
PAPER = "#F4F1EA"     # lab paper
ACCENT = "#C8452B"    # iron-oxide red (magnetite / rust)

FONT_NAME = "/usr/share/fonts/gsfonts/NimbusSans-Bold.otf"
FONT_TECH = "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf"

# ---------------------------------------------------------------- mark geometry
# 64 x 64 grid. Cube spans x 11..53, y 9..55; arcs are opposite segments of one
# circle of radius 30 centred on the cube's projected centre (32, 32).
TOP = "32,9 53,20.5 32,32 11,20.5"
LEFT = "11,20.5 32,32 32,55 11,43.5"
RIGHT = "53,20.5 32,32 32,55 53,43.5"
ARC_A = "M6.5 24 A30 30 0 0 1 22.5 6.8"
ARC_B = "M57.5 40 A30 30 0 0 1 41.5 57.2"
ARC_W = 3.4


def cube(top: str, left: str, right: str) -> str:
    return (
        f'<polygon points="{TOP}" fill="{top}"/>'
        f'<polygon points="{LEFT}" fill="{left}"/>'
        f'<polygon points="{RIGHT}" fill="{right}"/>'
    )


def arcs(color: str, width: float = ARC_W) -> str:
    return (
        f'<path d="{ARC_A}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="butt"/>'
        f'<path d="{ARC_B}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="butt"/>'
    )


def mark_svg(*, top=ACCENT, left=INK, right=MID, arc=ACCENT, compact: bool = False) -> str:
    body = cube(top, left, right) if compact else arcs(arc) + cube(top, left, right)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" '
        'aria-label="Loadstone">'
        f"{body}</svg>"
    )


# ---------------------------------------------------------------- wordmark paths
def text_paths(font_path: str, text: str, *, size: float, tracking_em: float) -> tuple[str, float, float]:
    """Outline `text` with optical spacing; returns (paths, width, cap height).

    Letters are placed by ink bounds rather than advance widths, so the visible
    gap is constant (O/A and S/T no longer read looser than L/O), and the letter
    pair spacing stays proportional at every size.
    """
    font = TTFont(font_path)
    upm = int(font["head"].unitsPerEm)
    cmap = font.getBestCmap() or {}
    glyphs = font.getGlyphSet()
    scale = size / upm
    gap = tracking_em * upm  # in font units, applied between ink edges
    x = 0.0
    parts = []
    prev = ""
    last_gap = gap
    # a few optical corrections that uniform ink spacing still gets wrong
    kern = {"ST": -0.07, "NE": 0.015, "DS": -0.02}
    for ch in text:
        name = cmap.get(ord(ch))
        if name is None:
            raise KeyError(f"{font_path} has no glyph for {ch!r}")
        pair_gap = gap + kern.get(prev + ch, 0.0) * upm
        prev = ch
        bounds = BoundsPen(glyphs)
        glyphs[name].draw(bounds)
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(pen)
        d = pen.getCommands()
        if bounds.bounds is None:  # e.g. a space
            x += gap
            continue
        xmin, _ymin, xmax, _ymax = bounds.bounds
        if d:
            parts.append(
                f'<path transform="translate({(x - xmin) * scale:.3f} 0) '
                f'scale({scale:.6f} {-scale:.6f})" d="{d}"/>'
            )
        x += (xmax - xmin) + pair_gap
        last_gap = pair_gap
    width = (x - last_gap) * scale
    cap = getattr(font["OS/2"], "sCapHeight", 0) or int(0.7 * upm)
    return "".join(parts), width, cap * scale


def lockup(horizontal: bool = True) -> str:
    """Mark plus outlined wordmark. Geometry is proportional, so it scales cleanly."""
    mark_h = 54.0
    name_size = 32.0          # font size units of the wordmark
    tracking = 0.19           # em, applied between ink edges
    paths, text_w, cap = text_paths(FONT_NAME, "LOADSTONE", size=name_size, tracking_em=tracking)
    gap = 23.0
    pad = 3.0
    if horizontal:
        total_w = pad + mark_h + gap + text_w + pad
        total_h = mark_h + 2 * pad
        baseline = total_h / 2 + cap / 2
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
            'role="img" aria-label="Loadstone">'
            f'<g transform="translate({pad} {pad})">{arcs(ACCENT)}{cube(ACCENT, INK, MID)}</g>'
            f'<g transform="translate({pad + mark_h + gap} {baseline})" fill="{INK}">{paths}</g>'
            "</svg>"
        )
    total_w = max(mark_h, text_w) + 2 * pad
    total_h = pad + mark_h + 34 + cap + pad
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
        'role="img" aria-label="Loadstone">'
        f'<g transform="translate({(total_w - mark_h) / 2:.1f} {pad})">{arcs(ACCENT)}'
        f"{cube(ACCENT, INK, MID)}</g>"
        f'<g transform="translate({(total_w - text_w) / 2:.1f} {pad + mark_h + 34 + cap})" fill="{INK}">'
        f"{paths}</g></svg>"
    )


def social_card() -> str:
    """1200x630 summary card: mark, name, one line of what the project is."""
    W, H = 1200, 630
    paths, text_w, cap = text_paths(FONT_NAME, "LOADSTONE", size=64, tracking_em=0.19)
    tech, tech_w, _ = text_paths(FONT_TECH, "ARCHITECTURE BENCHMARK HARNESS", size=20, tracking_em=0.18)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Loadstone">'
        f'<rect width="{W}" height="{H}" fill="{PAPER}"/>'
        f'<g transform="translate(96 150) scale(3.1)">{arcs(ACCENT)}{cube(ACCENT, INK, MID)}</g>'
        f'<g transform="translate(96 430)" fill="{INK}">{paths}</g>'
        f'<g transform="translate(100 470)" fill="{ACCENT}">{tech}</g>'
        f'<rect x="96" y="108" width="{W - 192}" height="2" fill="{INK}" opacity="0.14"/>'
        "</svg>"
    )


def write(name: str, svg: str) -> pathlib.Path:
    p = OUT / name
    p.write_text(svg + "\n", encoding="utf-8")
    return p


def render(svg_path: pathlib.Path, png: pathlib.Path, width: int, height: int | None = None) -> None:
    cmd = ["rsvg-convert", "-w", str(width), str(svg_path), "-o", str(png)]
    if height is not None:
        cmd[1:1] = ["-h", str(height)]
    subprocess.run(cmd, check=True)


def main() -> None:
    assets = {
        "mark.svg": mark_svg(),
        "mark-compact.svg": mark_svg(compact=True),
        "mark-reverse.svg": mark_svg(top=ACCENT, left=PAPER, right="#8E8B84"),
        "mark-mono.svg": mark_svg(top=INK, left=INK, right=MID, arc=INK),
        "lockup-horizontal.svg": lockup(horizontal=True),
        "lockup-stacked.svg": lockup(horizontal=False),
        "social-card.svg": social_card(),
    }
    for name, svg in assets.items():
        write(name, svg)

    # Rasters. The compact mark is the small-size cut; the full mark is used above 64px.
    for size in (1024, 512, 256, 128):
        render(OUT / "mark.svg", OUT / f"mark-{size}.png", size)
    for size in (64, 32, 16):
        render(OUT / "mark-compact.svg", OUT / f"mark-{size}.png", size)
    render(OUT / "lockup-horizontal.svg", OUT / "lockup-horizontal.png", 1400)
    render(OUT / "social-card.svg", OUT / "social-card.png", 1200, 630)

    # multi-resolution favicon from the compact cut
    ico_parts = []
    for size in (16, 32, 48):
        part = OUT / f".favicon-{size}.png"
        render(OUT / "mark-compact.svg", part, size)
        ico_parts.append(part)
    subprocess.run(["magick", *map(str, ico_parts), str(OUT / "favicon.ico")], check=True)
    for part in ico_parts:
        part.unlink()
    print("wrote", len(assets), "svgs, rasters and favicon.ico to", OUT)


if __name__ == "__main__":
    main()
