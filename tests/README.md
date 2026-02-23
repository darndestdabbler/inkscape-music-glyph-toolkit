# Music Glyph Toolkit — Test Suite

## Structure

```
tests/
├── README.md                    ← You are here
├── test_integration.py          ← Tier 2 (inkex.tester) + Tier 3 (LilyPond export)
├── fixtures/
│   ├── input/                   ← Test SVG files
│   │   ├── notehead_basic.svg       Simple elliptical notehead (MCLZ-clean)
│   │   ├── notehead_hollow.svg      Notehead + hollow cutout
│   │   ├── notehead_arcs.svg        Notehead using arc commands (A→C)
│   │   ├── notehead_for_export.svg  Pre-normalized, with stems (LilyPond export)
│   │   ├── glyph_mixed_commands.svg Treble-clef-like shape: S, Q, T, H, V, A, rel
│   │   └── glyph_multipath.svg      Multi-path glyph with chained S (Bug 1)
│   └── expected/                ← Frozen reference outputs (you create these)
│       └── (empty until you freeze)

../test_music_glyph_toolkit.py   ← Tier 1 unit tests (no Inkscape needed)
../music_glyph_toolkit.py        ← The extension under test
```

## Running the Tests

### Tier 1: Unit Tests (no Inkscape)

```bash
# From the project root
python -m pytest test_music_glyph_toolkit.py -v
```

Runs 61 pure Python tests covering `parse_and_normalize`, `format_svg_d`,
`format_scheme`, `arc_to_curves`, and `quadratic_to_cubic`. Mocks `inkex`
automatically — works anywhere with Python 3.8+ and pytest.

### Tier 3: LilyPond Export (no Inkscape)

```bash
python -m pytest tests/test_integration.py -v -k "LilyPond"
```

Runs string-level tests on Scheme output: balanced parens, valid keywords,
÷100 scaling, Y-inversion, stem exclusion, header format. Parses fixture
SVGs directly — no Inkscape needed.

### Tier 2: Integration Tests (requires Inkscape)

```bash
# Inkscape's Python must be on your path, or run with Inkscape's Python
python -m pytest tests/test_integration.py -v -k "not LilyPond"
```

These use `inkex.tester.ComparisonMixin` to run the full extension on input
fixtures and compare against frozen reference SVGs.

**First-time setup — freezing references:**

1. Open each fixture SVG in `tests/fixtures/input/` in Inkscape
2. Run the extension with the parameters shown in `test_integration.py`
3. Visually verify the output is correct
4. Save the result to `tests/fixtures/expected/` with a matching filename
5. Re-run the Tier 2 tests — they should now pass

If your `inkex` version supports `--create-references`, you can auto-generate
the expected files, then inspect them before committing.

## Fixture Design

| Fixture | Commands Exercised | Extension Mode |
|---|---|---|
| `notehead_basic.svg` | M, C, Z (already clean) | Notehead formatter |
| `notehead_hollow.svg` | M, C, Z (two paths) | Notehead + hollow |
| `notehead_arcs.svg` | A (two arcs forming ellipse) | Notehead + A→C |
| `notehead_for_export.svg` | M, C, Z + stems | LilyPond export |
| `glyph_mixed_commands.svg` | S, Q, T, H, V, h, v, a, l | Glyph formatter |
| `glyph_multipath.svg` | L, S chained implicit (Bug 1) | Multi-path glyph |
