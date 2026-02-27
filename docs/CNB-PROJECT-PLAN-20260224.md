# ChromaticNotationBase — Project Plan

**Version:** 5.0 (Cumulative Rules & Palette System)
**Date:** February 24, 2026
**Status:** Phases 0–2 complete. Phase 3 spec complete. Sessions 1–5 complete. Architecture revised for cumulative notehead rules, palette-based color system, inline staff/ledger styling, and accessible color switching.

---

## 1. VISION AND GOALS

ChromaticNotationBase (CNB) is a configurable engine for chromatic music notation in LilyPond. It is not a single notation system — it is infrastructure that can produce multiple chromatic notation systems (Kite, Clairnote SN, Chromatic Lyre, and others) from declarative configuration.

**Target audience:** LilyPond programmers who want to explore alternative chromatic notations.

**The workflow CNB enables:**

```
Inkscape + Music Glyph Toolkit    (Shape authoring & normalization)
    ↓ produces clean SVG with data- attributes
Chromatic Configurator            (Notation design & rule assignment)
    ↓ reads SVG metadata, user configures rules
.ly config file                   (Generated Scheme code)
    ↓
LilyPond + CNB engine             (Renders music)
    ↓
PDF / SVG / PNG output
```

**What was hard before CNB:** Converting SVG paths to Scheme format by hand. Calculating stem attachment points manually. Writing modulus-cycling logic in Scheme. Debugging visual output without structured IDs. All of this required deep Scheme knowledge and tedious trial-and-error.

**What CNB makes easy:** The Inkscape extension normalizes paths, computes stem anchors, and pre-bakes LilyPond code. The Configurator assigns shapes to pitches, defines staff geometry, and generates the config file. The engine renders music. The notation designer focuses on creative decisions.

**Release criteria:** CNB will not be released until the Configurator can generate a working `.ly` config file that produces correct output with the CNB engine. The existing Kite 4-369 notation and Joy to the World test score serve as the validation baseline.

---

## 2. CURRENT STATE (February 2026)

### 2.1 What Works

**CNB Engine (LilyPond/Scheme)** — 17 modules in `cnb/`, all functional:
- Chromatic staff positioning (configurable semitones-per-staff-space)
- Custom notehead stencils with shape/color/hollow cycling
- Dozenal (base-12) clef symbols
- Notehead-based key signatures (seventh chord stacks)
- Half notes = double noteheads, whole notes = quadruple noteheads
- Proximity-based ledger line visibility
- Beam thickness/spacing compensation, articulation Y-offset correction, augmentation dot avoidance
- Off-key note indicators (6 modes)
- **Procedure-oriented configuration model** with named definition tables, hash-table-based registration, resolver, and config cache
- **Named color references** for score-level accessibility overrides
- Per-line thickness and per-line color support
- Unit tests passing (108+) in `test-unit.ly`

**Inkscape Extension (Music Glyph Toolkit)** — Fully functional:
- Notehead and general glyph formatting modes
- 1 semitone = 100 SVG units coordinate grid
- Strict MCLZ path normalization (all SVG commands → absolute M/C/L/Z)
- Intelligent stem anchor generation as `<rect>` elements
- LilyPond `.ly` export with /100 scaling and Y-axis inversion
- `data-` attributes on SVG elements (`data-semitone-height`, `data-stem-up-x/y`, `data-stem-down-x/y`)
- Semitone guideline layer for visual reference
- ViewBox = notehead bounding box, origin-centered
- `<view id="full-view">` for complete glyph including stems
- 61 unit tests + integration tests + LilyPond export verification tests

**Testing Pipeline (Python)** — `test-ids.ly`, `harvest_svg.py`, `run_tests.py`:
- SVG id injection, SVG harvesting to structured JSON
- Integration tests passing (C major scale, 72/72 assertions)

**Chromatic Configurator** — ~60% complete (v3.0, Session 5):
- 6-tab UI: Meta, Noteheads, Colors, Staff, Shapes, Styles
- Card-based shape management with mini-previews (single, half-up, half-down)
- Named color system with preset palettes (Rich Tricolor, Colorblind-Safe, Krzywinski 12-Semitone, Monochrome)
- Per-shape overlap % for double-notehead minim rendering
- Auto-derived staff geometry (semitones-per-staff-space, staff-line-positions)
- Full `.ly` export (Sections A/B/C) — procedure model only
- JSON save/load with version migration
- Export validation
- SVG import from Music Glyph Toolkit files (file picker + drag-and-drop, data-attribute reading, type system)
- 193/193 unit tests passing (Node + browser)
- **Planned:** Cumulative notehead rules (shape/hollow/color as separate rule types)
- **Planned:** Palette-based color system with colorblind-safe alternatives
- **Planned:** Inline staff/ledger line styling (replaces style library)
- **Planned:** Accessible color switch in .ly export

**Color Palette System** — Complete prototype:
- `music-color-palettes.js`: self-contained library with CIEDE2000 color difference, nearestColorName, matchToCBSafe, curated standard + CB-safe palettes
- `palette-config.html`: configuration UI prototype with compact card + modal editor
- Standard/Colorblind-Safe mode toggle, preset palettes, custom palette editor
- Auto-generated color names (e.g., `Shamrock-#1b9e77`)
- Auto-generated CB-safe alternative mapping

### 2.2 Architecture Changes in v5.0

**Cumulative notehead rules replace first-match-wins.** The previous model had each rule carry a complete config (shape + color + hollow). The new model has three rule types (`shape`, `hollow`, `color`) that layer on top of each other. Last-match-wins per property. Defaults: first shape in defs or standard-oval, solid, black.

**Palette-based color system replaces ad-hoc color definitions.** Colors are defined as palettes (standard or colorblind-safe) and rules reference palette colors. Auto-generated names serve as Scheme symbols. Two palette sectors: notehead colors and line colors.

**Accessible color switching.** The .ly export emits a conditional block in Section A that registers either primary or CB-safe colors depending on a flag (`cnb:use-accessible-colors`). The LilyPond programmer controls this flag in their music file.

**Inline staff/ledger styling replaces style library.** Each staff/ledger rule carries its own color, thickness (as % of semitone height), and dash pattern. No indirection through named styles.

**Tab simplification.** Colors and Styles tabs eliminated. Meta tab gains two palette cards (notehead + line). 4 tabs total: Meta, Noteheads, Staff, Shapes.

### 2.3 Known Issues

1. **`test-ids.ly` and `off-key.ly` callback collision** — both set `after-line-breaking` on NoteHead grobs. Need callback chaining.
2. **Legacy anchor Y-convention change.** The Y-inversion in `buildAttachmentPoints` means legacy shapes' .ly anchor values now have inverted Y compared to previous exports. This is MORE correct but could affect existing .ly files.
3. **No validation filtering by shape type yet** — notehead rule dropdowns still show all shapes.
4. **Engine needs `cnb:use-accessible-colors` support** — new boolean flag + conditional color registration.

---

## 3. ARCHITECTURE

### 3.1 Three-Tool Pipeline

```
┌─────────────────────────────────────────────────────┐
│  INKSCAPE + MUSIC GLYPH TOOLKIT                     │
│                                                     │
│  Knows: Geometry, paths, semitone scaling, stems    │
│  Produces: Clean SVG with data-* attributes         │
│  Doesn't know: Pitch classes, colors, staff layout  │
└─────────────────┬───────────────────────────────────┘
                  │ .svg files (one per shape variant)
                  ▼
┌─────────────────────────────────────────────────────┐
│  CHROMATIC CONFIGURATOR                              │
│                                                     │
│  Knows: Pitch assignment, colors, staff geometry,   │
│         duration rules, LilyPond code generation    │
│  Consumes: SVG data-* attributes (trusts geometry)  │
│  Produces: .ly config file + .json save file        │
│  Doesn't know: How to normalize SVGs                │
└─────────────────┬───────────────────────────────────┘
                  │ .ly config file
                  ▼
┌─────────────────────────────────────────────────────┐
│  CNB ENGINE (LilyPond/Scheme)                       │
│                                                     │
│  Knows: Rendering, positioning, engraving           │
│  Consumes: Procedure-model config                   │
│  Produces: PDF / SVG / PNG                          │
└─────────────────────────────────────────────────────┘
```

### 3.2 SVG Contract (Inkscape Extension → Configurator)

The Configurator expects SVGs produced by the Music Glyph Toolkit with these conventions:

**Elements by ID:**
- `#notehead` — main notehead path (required)
- `#hollow` — hollow cutout path (optional)
- `#stem-up` — stem-up `<rect>` (optional; auto-generated by extension)
- `#stem-down` — stem-down `<rect>` (optional; auto-generated by extension)

**Data attributes on `#notehead`:**
- `data-semitone-height` — height in semitones (e.g., `"2.0"`)
- `data-stem-up-x`, `data-stem-up-y` — stem-up attachment point (centerline of rect, SVG coords)
- `data-stem-down-x`, `data-stem-down-y` — stem-down attachment point
- `data-lilypond-notehead` — (planned) pre-baked LilyPond path in `'((moveto ...) ...)` format

**Data attributes on `#hollow`:**
- `data-lilypond-hollow` — (planned) pre-baked LilyPond hollow path

**Coordinate system:**
- 1 semitone = 100 SVG units
- Origin (0,0) = geometric center of notehead
- ViewBox tightly wraps notehead bounding box
- Paths use only absolute M, C, L, Z commands
- Positive Y is down (SVG convention); LilyPond export inverts Y

### 3.3 Procedure Model (Only Model)

The engine uses a single Scheme procedure `(semitone, duration, key) → config alist`. The input space is bounded: 12 semitones × 4 duration categories = 48 entries per key. The config cache is rebuilt on key changes.

Config alist fields:
- `shape` — symbol referencing `cnb:path-defs`
- `color` — symbol (→ `cnb:color-defs`), hex string, or integer (legacy)
- `hollow` — `#f` or symbol referencing `cnb:hollow-defs`
- `attachment` — symbol referencing `cnb:attachment-defs`
- `off-key-mark` — `#f` or symbol/built-in

### 3.4 Color Strategy (v5.0)

**Palette-based color definitions.** Colors are selected from curated palettes (standard or colorblind-safe). Each color in the palette gets an auto-generated name via `nearestColorName()` from the `music-color-palettes.js` library (e.g., `Shamrock-#1b9e77` → Scheme symbol `shamrock-1b9e77`).

**Two palette sectors:**
- **Notehead palette** — colors available for notehead color rules. Defaults to Dark2 · 7.
- **Line palette** — colors available for staff/ledger line rules. Defaults to Monochrome (black only).

**Accessible color switching.** The .ly export emits a conditional block:

```scheme
#(if (and (defined? 'cnb:use-accessible-colors) cnb:use-accessible-colors)
  (begin
    (cnb:register-color! 'shamrock-1b9e77 "#009f81")  ;; CB-safe: Teal
    ...)
  (begin
    (cnb:register-color! 'shamrock-1b9e77 "#1b9e77")  ;; Primary
    ...))
```

Symbol names are identical in both branches. Section C references the symbol regardless of which palette is active. The LilyPond programmer sets `cnb:use-accessible-colors` before `\include`-ing the config. For monochrome palettes, both branches are identical (black→black). For CB-safe-as-primary, both branches are also identical.

**No more ad-hoc color definitions.** The Colors tab and `App.colors` dictionary are replaced by `App.noteheadPalette` and `App.linePalette`.

### 3.5 Cumulative Notehead Rules (v5.0)

**Three rule types:**
- `shape` — assigns a shape to a set of pitch classes
- `hollow` — assigns hollowness (true/false) to a set of pitch classes
- `color` — assigns a palette color to a set of pitch classes

**Resolution semantics:**
- Rules are evaluated top-to-bottom
- Last-match-wins per property (shape, hollow, color independently)
- Defaults before any rules: shape = first shape in defs or standard-oval, hollow = false, color = black

**Scheme generation strategy:** Resolve all 12 pitch classes by walking rules, then group pitch classes with identical final configs into `cond` clauses. This produces clean, minimal Scheme code.

**Each rule also carries:**
- `pitchClasses` — 'all' or array of pitch class indices
- `durations` — 'all' or array (disabled for now, defaults to 'all')
- `keys` — 'all' (disabled for now)

### 3.6 Staff/Ledger Line Rules (v5.0)

Each staff or ledger line rule carries its own styling directly (no style library indirection):
- `type` — 'staff' or 'ledger'
- `color` — hex color (from line palette or free picker)
- `thickness` — percentage of semitone height (staff default: 20%, ledger default: 40%)
- `dash` — dash pattern or null
- `pitchClasses` — which positions get this line
- `override` — (ledger only) which notes trigger visibility
- `keys` — 'all' (disabled for now)

### 3.7 Minim (Half Note) Strategy

**Default behavior:** When the same shape is assigned to both crotchet (quarter) and minim (half) duration categories, the engine automatically renders double noteheads for minims. The Configurator owns the minim offset logic:

- Crotchet attachment points come directly from the SVG's `data-stem-*` attributes
- Minim attachment points are derived by `deriveMinimAnchors()` using per-shape overlap %
- Per-shape overlap % defaults to 25 (configurable per shape in the Shapes tab)
- The 4-point attachment model (crotchet-up/down, minim-up/down) is emitted in the `.ly` export

---

## 4. INTERFACE CONTRACT: CNB Engine ↔ Configurator

See **CNB-LY-EXPORT-SPEC-v1_2.md** for the complete specification. Summary:

**Registration functions:**
- `cnb:register-path!`, `cnb:register-hollow!`, `cnb:register-attachment!`, `cnb:register-color!`, `cnb:register-off-key-mark!`

**Core engine settings:**
- `cnb:semitones-per-staff-space`, `cnb:staff-line-positions`, `cnb:false-hollow`, `cnb:multinote-padding`, `cnb:default-magnify`, `cnb:file-suffix`, `cnb:notation-label`, `cnb:notation-footnote`

**New in v5.0:**
- `cnb:use-accessible-colors` — boolean flag, set by the LilyPond programmer in their music file before `\include`-ing the config. When `#t`, color registrations use CB-safe alternatives.

**The procedure:**
- `cnb:get-notehead-config` — `(semitone, duration, key) → alist`

---

## 5. FILE STRUCTURE

### 5.1 Configurator Project

```
cnb-configurator/
├── chromatic-configurator.html       Main app (UI layer)
├── configurator-core.js             Testable logic (no DOM)
├── music-color-palettes.js          Color palette library (no dependencies)
├── configurator-tests.html          Browser test runner (193+ tests)
├── run-tests.js                     Node test runner
│
├── reference/                        Read-only CNB reference files
│   ├── Kite-4-369-Notation-Procedure.ly
│   ├── CNB-LY-EXPORT-SPEC-v1_2.md
│   └── colors-reference.md
│
└── test-configs/                     Test JSON files
    └── kite-4-369.json
```

### 5.2 CNB Engine (LilyPond project)

```
cnb-project/
├── ChromaticNotationBase.ly          Main engine entry point
├── Kite-4-369-Notation-Procedure.ly  Reference config
├── test-unit.ly                      Unit test suite (108+)
├── cnb/                              Engine modules (17 files)
└── scores/                           Example music files
```

---

## 6. WORK PHASES

### ~~Phase 0: Preserve Baseline~~ ✓ COMPLETE
### ~~Phase 1: Procedure Infrastructure~~ ✓ COMPLETE (Session 7)
### ~~Phase 2: Engraver Rewired~~ ✓ COMPLETE (Session 7)
### ~~Phase 3: Export Spec~~ ✓ COMPLETE (CNB-LY-EXPORT-SPEC-v1_0.md)
### ~~Configurator Sessions 1–5~~ ✓ COMPLETE
- Architecture + build, unit tests (75→193), UI fixes, per-shape overlap, stem thickness, Krzywinski palette, SVG import rewrite, type system, coordinate scaling fix

### Phase 4: Inkscape Extension Update (1 session)

**Goal:** Add `data-lilypond-*` attributes to SVG output.

- [ ] Add `data-lilypond-notehead` attribute to `#notehead` element
- [ ] Add `data-lilypond-hollow` attribute to `#hollow` element
- [ ] Add checkbox "Include LilyPond path data" in extension UI
- [ ] Tests and README update

### Phase 5: Engine Accessible Color Support (1 session)

**Goal:** Add `cnb:use-accessible-colors` flag to the engine.

- [ ] Define `cnb:use-accessible-colors` variable (default `#f`)
- [ ] Verify conditional color registration pattern works (the Configurator generates the `if` block — engine just needs the variable defined)
- [ ] Add unit tests
- [ ] Document in engine README

See **BATON-ENGINE-ACCESSIBLE-COLORS.md** for details.

### Phase 6: Configurator Session 6 — Palette System & Cumulative Rules (1 session)

**Goal:** Integrate the palette system, implement cumulative notehead rules, update Section C generation.

- [ ] Integrate `music-color-palettes.js` into project
- [ ] Add two palette cards to Meta tab (notehead + line)
- [ ] Rewrite notehead rules to cumulative model (shape/hollow/color rule types)
- [ ] New popup forms for each rule type
- [ ] Update resolution logic and Section C generation
- [ ] Update Section A to emit conditional color registration
- [ ] Remove Colors tab
- [ ] Update tests

See **BATON-SESSION-6-20260224.md** for details.

### Phase 7: Configurator Session 7 — Staff/Ledger Inline Styling & Display (1 session)

**Goal:** Staff/ledger rules with inline styling, display improvements, cleanup.

- [ ] Rewrite staff/ledger rules with inline color/thickness/dash
- [ ] Staff line thickness as % of semitone height (default 20%)
- [ ] Ledger line thickness as % of semitone height (default 40%)
- [ ] Display stems (~25% semitone height thick, length = max(7 semitones, 2.5× notehead height))
- [ ] Remove Styles tab
- [ ] Separate add buttons for staff vs. ledger rules
- [ ] Update tests

See **BATON-SESSION-7-20260224.md** for details.

### Phase 8: Configurator End-to-End Verification (1 session)

- [ ] Design Kite 4-369 notation using full pipeline
- [ ] Export `.ly` from Configurator
- [ ] Render Joy to the World with exported config
- [ ] Compare against golden baseline
- [ ] Fix any discrepancies
- [ ] Document the full workflow in configurator-README.md

### Future Phases (deferred)

- **Duration-aware rules:** Enable duration field in cumulative rules (currently disabled, defaults to 'all')
- **Key-aware rules:** Enable keys field in rules
- **Off-key mark assignment:** Add off-key mode selector per rule
- **Staff geometry procedure:** `cnb:get-staff-config` for key-dependent staff layouts
- **Engine vector model removal:** Strip dual-model branches from engine modules
- **Clef/glyph support in Configurator:** The extension supports General Glyph mode; Configurator could manage non-notehead glyphs

---

## 7. GLOSSARY

| Term | Definition |
|------|-----------|
| **Accessible colors** | Colorblind-safe alternative palette, auto-generated by `matchToCBSafe()` |
| **Baker** | Legacy SVG normalization subsystem — replaced by Inkscape extension |
| **CB-safe** | Colorblind-safe — colors tested for distinguishability across color vision deficiencies |
| **Config cache** | Hash table mapping `(pitch-class, duration-category) → resolved config`. Rebuilt on key change. |
| **Configurator** | Web-based tool for designing notation systems and exporting `.ly` configs |
| **Crotchet** | Quarter note (filled notehead) |
| **Cumulative rules** | Notehead rules that layer properties independently; last-match-wins per property |
| **Duration category** | 0=breve+, 1=whole, 2=half, 3=quarter+ |
| **False-hollow** | White overlay technique for angular shapes (vs. true path-winding hollow) |
| **MCLZ** | The four absolute SVG path commands after normalization: Move, Cubic, Line, closePath |
| **Minim** | Half note (double notehead by default) |
| **Music Glyph Toolkit** | Inkscape extension for normalizing musical glyph SVGs |
| **Named reference** | A symbol (like `kite-up`) resolved to data via definition table lookup |
| **Palette sector** | A color palette assigned to a purpose (notehead or line) |
| **Pitch class** | Semitone mod 12 (C=0, C#=1, ..., B=11) |
| **Procedure model** | The configuration architecture: `(semitone, duration, key) → config alist` |

---

## CHANGE LOG

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial plan |
| 2.0 | Feb 19, 2026 | Procedure-oriented architecture design |
| 3.0 | Feb 21, 2026 | Post-Session 7. Phases 0–2 complete. Interface contract added. |
| 3.1 | Feb 21, 2026 | Interface contract gaps filled. Color palette updates. |
| 4.0 | Feb 23, 2026 | Inkscape extension integration. Baker obsoleted. Vector model sunsetted. |
| **5.0** | **Feb 24, 2026** | **Cumulative notehead rules. Palette-based color system with `music-color-palettes.js`. Accessible color switching (`cnb:use-accessible-colors`). Inline staff/ledger styling. Colors + Styles tabs eliminated. Tab count 6→4.** |
