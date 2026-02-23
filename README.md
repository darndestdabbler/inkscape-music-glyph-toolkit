# Music Glyph Toolkit for Inkscape

A professional Inkscape extension designed to bridge the gap between visual vector editing and code-based music typesetting (like LilyPond or custom web rendering apps). 

This toolkit standardizes musical assets by applying strict mathematical coordinate translations, proportional semitone-based scaling, absolute path conversion, and direct code generation.

## Features

* **Multi-Tabbed Interface:** Choose between dedicated `Notehead` processing or `General Glyph` (Clefs, Time Signatures) formatting, with always-visible export settings below.
* **1 Semitone = 100 Units:** A standardized internal coordinate grid. If a user sets a notehead height to 2 semitones, the geometric height of the path is strictly baked to 200 units.
* **Intelligent Stem Anchoring:** Automatically attaches stems (as `<rect>` elements) to the exact visual `minY at maxX` (Top-Right) and `maxY at minX` (Bottom-Left) of the notehead curve. Stems can be repositioned in Inkscape before export.
* **Standard Stem Height:** 350 units (3.5 semitones), matching standard engraving proportions.
* **Dynamic Stem Width:** Calculated based on standard typographic rules: `max(Height/9, 20 units)`. This ensures stems are never thinner than a standard staff line, even for tall cluster chords.
* **Strict MCLZ Path Conversion:** Automatically normalizes messy SVG outputs (translating `S`, `Q`, `T`, `A`, `H`, `V` commands) into strictly absolute `M` (Move), `L` (Line), `C` (Cubic Bezier), and `Z` (ClosePath) coordinates.
* **LilyPond Export:** Generates a `.ly` Scheme file with `\version` header, path coordinates scaled to staff-space units (/100), Y-axis inverted, and stem attachment point defines.
* **Export Validation:** The LilyPond exporter validates that all paths use strict M/C/L/Z commands and that noteheads are centered at the origin before exporting.

## Installation

1. Download both `music_glyph_toolkit.inx` and `music_glyph_toolkit.py`.
2. Move both files into your Inkscape User Extensions folder:
   * **Windows:** `%APPDATA%\inkscape\extensions\`
   * **Mac/Linux:** `~/.config/inkscape/extensions/`
3. Restart Inkscape. 
4. The tool is available under **Extensions > Music Typography > Music Glyph Toolkit...**

## Pre-Requisites & Safety 

**Text and shapes must be converted to paths!**
Inkscape Python extensions do not have access to your system's typography engine. If you type the number "3" and try to run the tool, it cannot calculate the mathematical curves.
* Always select your objects and press `Ctrl+Shift+C` (Path > Object to Path) before running the toolkit.
* If you have overlapping shapes, press `Ctrl++` (Path > Union) to fuse them into a single path.
*(Note: The extension includes safety checks and will warn you via an alert box if it finds unconverted text or geometry.)*

## Usage Guide

### Mode 1: Notehead Asset Creation
1. Draw your notehead and optional hollow cutout.
2. Convert them to paths (`Ctrl+Shift+C`).
3. Open the Object Properties (`Ctrl+Shift+O`) and set the ID of the main shape to `notehead`. If you have a cutout, set its ID to `hollow`.
4. Run the extension, select the **Notehead Formatter** tab, ensure the Action is set to **Update SVG Canvas**, input the desired height in semitones, and click Apply.
5. The tool generates stem rectangles (`stem-up` and `stem-down`) automatically. You can reposition these in Inkscape before exporting.

### Mode 2: General Glyph (Clefs, Rests, Accidentals)
1. Draw your glyph and convert to a path. Union multiple paths if needed.
2. Run the extension, select the **General Glyph Formatter** tab, input the desired height in semitones, and click Apply.
3. The tool calculates the collective bounding box, centers everything at 0,0, and proportionally scales the drawing to the designated semitone height. 

### Mode 3: LilyPond Code Export
1. Prepare your assets on the canvas using Mode 1 or Mode 2.
2. Optionally reposition stems in Inkscape to fine-tune attachment points.
3. Re-open the extension and switch the **Action** dropdown to **Export to LilyPond (.ly)**.
4. Click Apply.
5. The extension validates the canvas (checking for forbidden path commands and proper centering), then exports. If validation fails, you'll see a descriptive error message telling you what to fix.

**Using the exported `.ly` file:** The export produces a compilable LilyPond file with Scheme path lists and stem attachment points. To use the paths in LilyPond, wrap each in `make-path-stencil`:

```lilypond
#(define my-notehead-stencil
   (make-path-stencil
     my-notehead-path  ; the exported path list
     0.1               ; line thickness
     1                 ; X scale
     1))               ; Y scale
```

Stem attachment points are exported as `(x . y)` pairs in staff-space coordinates:

```lilypond
%% Use the stem attachment points for custom stem positioning:
%% notehead-basic-notehead-stem-up   → '(x . y) for stem-up attachment
%% notehead-basic-notehead-stem-down → '(x . y) for stem-down attachment
```

Stem rectangles (`stem-up`, `stem-down`) are automatically excluded from the path export since LilyPond handles stem rendering natively.

## Guidelines

Both formatters generate horizontal semitone reference lines as SVG `<line>` elements in a dedicated Inkscape layer called **Semitone Guidelines**. The number of guidelines is computed dynamically to cover the full vertical extent of all visible elements (including stems), so large glyphs like clefs get as many lines as they need.

Each semitone is marked by a pair of abutting dashed lines — the meeting point of the two colors is the exact semitone position. The two line colors are chosen dynamically to contrast against the notehead, hollow, and stem fill/stroke colors.

Lines are labeled using the convention `semitone i`, `semitone i + 1`, `semitone i - 1`, etc., where `i` is the glyph center. The center line is at SVG y=0, with each semitone spaced at 100 SVG units.

| Example lines | SVG Y | ID pattern |
|---|---|---|
| semitone i + N | −N×100 | `guideline-semi-iplusN` |
| semitone i | 0 | `guideline-semi-i` |
| semitone i − N | +N×100 | `guideline-semi-iminusN` |

The guidelines layer can be toggled on/off via the **Show semitone guidelines** checkbox in the extension UI, or by toggling layer visibility in Inkscape's Layers panel.

## Viewport

The SVG root element's `width`, `height`, and `viewBox` are set to exactly match the target notehead/glyph dimensions with no margin. The viewBox is centered at the origin:

```
width  = tnh × (notehead_width / notehead_height)
height = tnh
viewBox = "-width/2 -height/2 width height"
```

This tight fit simplifies scaling and positioning in consuming applications (e.g., JavaScript renderers).

### Full View

A `<view>` element with `id="full-view"` is also added to the SVG. Its `viewBox` encompasses all objects including stems, so a consuming program can optionally display the complete glyph with stems by referencing the SVG with `#full-view`:

```html
<img src="notehead.svg#full-view" />
```

In glyph mode (no stems), the full view matches the main viewBox.

## Architecture

The path normalization pipeline has three stages:

```
parse_and_normalize(d_str)  →  List[Tuple[str, List[float]]]
                                       ↓
                          ┌────────────┴────────────┐
                          ↓                         ↓
             format_svg_d(commands)       format_scheme(commands)
               → SVG d attribute            → LilyPond Scheme
```

`parse_and_normalize` is the core engine. It tokenizes the SVG `d` string, converts all commands to absolute coordinates, and reduces everything to M/C/L/Z. The two formatters are trivial serializers that consume the same structured output — `format_svg_d` for writing back to SVG, and `format_scheme` for LilyPond export (with /100 scaling and Y-axis inversion).

The extension has two operating modes controlled by the always-visible **Action** dropdown:

* **Update SVG Canvas** — Runs the selected formatter tab (Notehead or Glyph), then applies MCLZ normalization.
* **Export to LilyPond** — Skips all processing, validates the canvas state, and exports to a `.ly` file. This decoupling allows you to manually adjust stems and other elements between processing and export.

## Testing

The test suite has three tiers:

**Tier 1 — Unit tests** (`test_music_glyph_toolkit.py`): 61 pure pytest tests covering every SVG command type, edge cases, and all three pipeline functions. No Inkscape required.

```bash
python -m pytest test_music_glyph_toolkit.py -v
```

**Tier 2 — Integration tests** (`tests/test_integration.py`): Uses `inkex.tester` to run the full extension on fixture SVGs and compare against frozen reference outputs. Requires Inkscape.

**Tier 3 — LilyPond export verification** (same file): String-level tests validating Scheme syntax, coordinate scaling, Y-inversion, stem exclusion, and `\version` header. No Inkscape required.

```bash
python -m pytest tests/test_integration.py -v -k "LilyPond"
```

See `tests/README.md` for full setup instructions.
