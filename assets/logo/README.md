# Loadstone identity

The mark is a lodestone: an isometric stone lit from above, with two segments of
an induced field around it. The pair of arcs is the same 180°-rotational pairing
the benchmark uses for compared cells; the top facet is the only charged surface.

Everything here is generated from `build_assets.py` (one geometry source, outlined
wordmark, no font needed to open the SVGs). `build_preview.py` regenerates the
specimen sheet.

```sh
.venv/bin/python assets/logo/build_assets.py    # svgs, pngs, favicon.ico
.venv/bin/python assets/logo/build_preview.py   # preview.html specimen sheet
```

Asset generation needs `fonttools` and `rsvg-convert`; neither is a runtime
dependency of the app.

## Files

| File | Use |
|---|---|
| `mark.svg` | primary mark, full colour, ≥ 64 px |
| `mark-compact.svg` | cube only, no field — ≤ 64 px, favicons, tight UI |
| `mark-reverse.svg` | on ink or photography |
| `mark-mono.svg` | single-ink print, stamps, etching |
| `lockup-horizontal.svg` | default lockup for docs, README, slides |
| `lockup-stacked.svg` | square-ish placements, covers, merchandise |
| `social-card.svg` | 1200 × 630 link preview |
| `favicon.ico` | 16 / 32 / 48 px |
| `preview.html` | specimen sheet (all variants in one page) |
| `mark-*.png`, `lockup-horizontal.png`, `social-card.png` | rasters where SVG is not accepted |

## Palette

| Name | Value | Role |
|---|---|---|
| Ink | `#15161A` | stone (unlit face), wordmark, rules |
| Stone | `#3B3E46` | stone (lit face) |
| Iron oxide | `#C8452B` | charged facet, induced field, accents only |
| Paper | `#F4F1EA` | ground |

Iron oxide is the only accent. It marks the charged surface and the field —
used as a fill for large areas or as a highlight, never as body text colour on
paper without checking contrast.

## Rules

* Clear space on all four sides = `0.25 ×` mark height. Nothing enters it.
* Minimum mark height: `16 px` compact, `64 px` full (below that the arcs blur).
* Minimum lockup width: `120 px`. Below that, use the mark alone.
* Do not restretch, rotate, or recolour individual stone facets; do not swap the
  paired arc for a single arc (the pairing is the meaning).
* Do not place the full-colour mark on mid-grey — use the reverse or mono cut.
* The wordmark is outlined from Nimbus Sans Bold with `0.19 em` ink tracking and
  small optical corrections on `ST`, `NE`, `DS`. Do not re-set it in another face.

## Why these shapes

* The block is a measurement unit: solid, countable, the thing being weighed.
* The two arcs are opposite segments of one circle centred on the block — balance,
  and the paired comparison the app is built around.
* The top facet is the only lit surface: the charged face is where the load goes.
