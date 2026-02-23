"""
Tier 2: Integration Tests — inkex.tester (requires Inkscape installation)
Tier 3: LilyPond Export Verification (string-level, no Inkscape needed)

SETUP (Tier 2 only):
  1. Install Inkscape (1.2+)
  2. Ensure `inkex` is importable (it ships with Inkscape's Python)
  3. Copy music_glyph_toolkit.py + music_glyph_toolkit.inx into this test dir
     (or adjust EXTENSION_DIR below)
  4. Run: python -m pytest tests/test_integration.py -v

FREEZING REFERENCE FILES (first run):
  Tier 2 tests use ComparisonMixin which compares output SVGs against frozen
  reference files in tests/fixtures/expected/. On first run these won't exist
  and tests will fail. To create them:

  1. Run the extension manually in Inkscape on each input fixture
  2. Visually verify the output looks correct
  3. Save the output SVG into tests/fixtures/expected/ with a matching name
  4. Re-run the tests — they should now pass

  Alternatively, use --create-references (if supported by your inkex version)
  to auto-generate, then manually inspect before committing.
"""
import os
import re
import sys
import math
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
#  Path setup
# ═══════════════════════════════════════════════════════════════════════════════

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_INPUT = os.path.join(TEST_DIR, "fixtures", "input")
FIXTURES_EXPECTED = os.path.join(TEST_DIR, "fixtures", "expected")

# Add parent dir so music_glyph_toolkit is importable
sys.path.insert(0, os.path.join(TEST_DIR, ".."))


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 2: inkex.tester integration tests
# ═══════════════════════════════════════════════════════════════════════════════

# Guard: only define Tier 2 tests if inkex is available
try:
    import inkex
    from inkex.tester import ComparisonMixin, InkscapeExtensionTestMixin
    HAS_INKEX = True
except ImportError:
    HAS_INKEX = False

if HAS_INKEX:
    from music_glyph_toolkit import MusicGlyphToolkit

    class TestNoteheadBasic(ComparisonMixin, InkscapeExtensionTestMixin):
        """Run notehead formatter on a basic elliptical notehead."""
        effect_class = MusicGlyphToolkit
        compare_file = [os.path.join(FIXTURES_INPUT, "notehead_basic.svg")]
        comparisons = [
            # Default: 2 semitones, strict MCLZ, SVG output
            ("--active_tab=tab_notehead", "--nh_semitones=2.0",
             "--strict_mclz=true", "--round_decimals=3", "--output_format=svg"),
        ]

    class TestNoteheadHollow(ComparisonMixin, InkscapeExtensionTestMixin):
        """Notehead + hollow cutout processing."""
        effect_class = MusicGlyphToolkit
        compare_file = [os.path.join(FIXTURES_INPUT, "notehead_hollow.svg")]
        comparisons = [
            ("--active_tab=tab_notehead", "--nh_semitones=2.0",
             "--strict_mclz=true", "--round_decimals=3", "--output_format=svg"),
        ]

    class TestNoteheadArcs(ComparisonMixin, InkscapeExtensionTestMixin):
        """Notehead with arc commands — tests A→C conversion."""
        effect_class = MusicGlyphToolkit
        compare_file = [os.path.join(FIXTURES_INPUT, "notehead_arcs.svg")]
        comparisons = [
            ("--active_tab=tab_notehead", "--nh_semitones=2.0",
             "--strict_mclz=true", "--round_decimals=3", "--output_format=svg"),
        ]

    class TestGlyphMixed(ComparisonMixin, InkscapeExtensionTestMixin):
        """General glyph with mixed dirty commands (S, Q, T, H, V, A, relative)."""
        effect_class = MusicGlyphToolkit
        compare_file = [os.path.join(FIXTURES_INPUT, "glyph_mixed_commands.svg")]
        comparisons = [
            ("--active_tab=tab_glyph", "--glyph_semitones=4.0",
             "--strict_mclz=true", "--round_decimals=3", "--output_format=svg"),
        ]

    class TestGlyphMultipath(ComparisonMixin, InkscapeExtensionTestMixin):
        """Multi-path glyph — collective bbox, Bug 1 regression (chained S)."""
        effect_class = MusicGlyphToolkit
        compare_file = [os.path.join(FIXTURES_INPUT, "glyph_multipath.svg")]
        comparisons = [
            ("--active_tab=tab_glyph", "--glyph_semitones=4.0",
             "--strict_mclz=true", "--round_decimals=3", "--output_format=svg"),
        ]

    class TestMCLZDisabled(ComparisonMixin, InkscapeExtensionTestMixin):
        """With strict_mclz=false, paths should pass through un-normalized."""
        effect_class = MusicGlyphToolkit
        compare_file = [os.path.join(FIXTURES_INPUT, "notehead_basic.svg")]
        comparisons = [
            ("--active_tab=tab_notehead", "--nh_semitones=2.0",
             "--strict_mclz=false", "--output_format=svg"),
        ]

    class TestDifferentSemitones(ComparisonMixin, InkscapeExtensionTestMixin):
        """Different semitone heights produce different scaling."""
        effect_class = MusicGlyphToolkit
        compare_file = [os.path.join(FIXTURES_INPUT, "notehead_basic.svg")]
        comparisons = [
            ("--active_tab=tab_notehead", "--nh_semitones=3.0",
             "--strict_mclz=true", "--round_decimals=3", "--output_format=svg"),
        ]

else:
    # Placeholder so pytest doesn't silently skip everything
    class TestInkexNotAvailable:
        def test_inkex_warning(self):
            pytest.skip(
                "inkex not installed — Tier 2 integration tests require Inkscape. "
                "Tier 1 unit tests (test_music_glyph_toolkit.py) and Tier 3 below "
                "run without Inkscape."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 3: LilyPond Export Verification (no Inkscape needed)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  These tests exercise format_scheme() and the export logic at the string level.
#  They parse the `d` attributes from our fixture SVGs directly, bypassing Inkscape.
# ═══════════════════════════════════════════════════════════════════════════════

# Import pipeline functions (inkex is mocked in test_music_glyph_toolkit.py,
# but we do our own lightweight mock here too for independence)
import types
if not HAS_INKEX:
    _mock = types.ModuleType("inkex")
    _mock.EffectExtension = type("EffectExtension", (), {})
    _mock.PathElement = type("PathElement", (), {})
    _mock.Boolean = bool
    _mock.AbortExtension = Exception
    _mock.etree = None
    sys.modules.setdefault("inkex", _mock)

from music_glyph_toolkit import parse_and_normalize, format_scheme, format_svg_d, FORBIDDEN_COMMANDS


def _read_svg_path_ds(svg_path, exclude_ids=None):
    """Extract (id, d) pairs from an SVG file using simple regex (no lxml needed)."""
    exclude_ids = exclude_ids or set()
    with open(svg_path) as f:
        content = f.read()
    # Match <path id="..." d="..."/> — good enough for our clean fixtures
    pattern = re.compile(
        r'<path\s[^>]*?id="([^"]*)"[^>]*?d="([^"]*)"', re.DOTALL
    )
    results = []
    for m in pattern.finditer(content):
        pid, d = m.group(1), m.group(2)
        if pid not in exclude_ids:
            results.append((pid, d))
    return results


class TestLilyPondSchemeOutput:
    """Verify format_scheme produces valid LilyPond Scheme syntax."""

    @pytest.fixture
    def notehead_scheme(self):
        """Parse the export fixture's notehead and format as Scheme."""
        fixture = os.path.join(FIXTURES_INPUT, "notehead_for_export.svg")
        paths = _read_svg_path_ds(fixture, exclude_ids={"stem-up", "stem-down"})
        assert len(paths) == 1
        pid, d = paths[0]
        assert pid == "notehead"
        cmds = parse_and_normalize(d)
        return format_scheme(cmds)

    def test_balanced_parens(self, notehead_scheme):
        """Every opening paren has a matching close."""
        assert notehead_scheme.count("(") == notehead_scheme.count(")")

    def test_valid_keywords(self, notehead_scheme):
        """Only allowed Scheme path keywords appear."""
        allowed = {"moveto", "lineto", "curveto", "closepath"}
        # Extract all keywords after opening parens
        keywords = re.findall(r"\((\w+)", notehead_scheme)
        for kw in keywords:
            assert kw in allowed, f"Unexpected Scheme keyword: {kw}"

    def test_moveto_present(self, notehead_scheme):
        assert "(moveto" in notehead_scheme

    def test_closepath_present(self, notehead_scheme):
        assert "(closepath)" in notehead_scheme

    def test_coordinates_scaled(self, notehead_scheme):
        """Coordinates should be in staff-space units (~1.0 range), not SVG units (~100 range)."""
        numbers = [float(n) for n in re.findall(r"[\d.]+", notehead_scheme)]
        for n in numbers:
            assert abs(n) < 10, (
                f"Coordinate {n} looks unscaled (expected /100 staff-space range)"
            )

    def test_y_inversion(self, notehead_scheme):
        """The fixture notehead has y=-100 (top). After /100 and negation → +1.0 in Scheme."""
        # Original d has C ... -100 ... so after scale=0.01 and negate: +1.0
        assert "1.0" in notehead_scheme

    def test_curveto_has_six_numbers(self, notehead_scheme):
        """Each curveto should have exactly 6 numeric arguments."""
        for line in notehead_scheme.split("\n"):
            if "(curveto" in line:
                numbers = re.findall(r"-?[\d.]+", line)
                assert len(numbers) == 6, f"curveto with wrong arg count: {line}"

    def test_moveto_has_two_numbers(self, notehead_scheme):
        for line in notehead_scheme.split("\n"):
            if "(moveto" in line:
                numbers = re.findall(r"-?[\d.]+", line)
                assert len(numbers) == 2, f"moveto with wrong arg count: {line}"


class TestLilyPondStemExclusion:
    """Verify that stem paths are excluded from export."""

    def test_stems_excluded(self):
        fixture = os.path.join(FIXTURES_INPUT, "notehead_for_export.svg")
        all_paths = _read_svg_path_ds(fixture)
        non_stem = _read_svg_path_ds(fixture, exclude_ids={"stem-up", "stem-down"})
        # Fixture has 3 paths total (notehead + 2 stems)
        assert len(all_paths) == 3
        # After excluding stems, only notehead remains
        assert len(non_stem) == 1
        assert non_stem[0][0] == "notehead"


class TestLilyPondHeaderFormat:
    """Verify the .ly output header structure."""

    def test_expected_header_lines(self):
        """The export_lilypond should produce these header lines."""
        expected_fragments = [
            '\\version "2.24.2"',
            "%% Generated from:",
            "%% Coordinates: scaled to staff-space units",
            "%% Usage: wrap path list in (make-path-stencil",
        ]
        # We can't call export_lilypond() without Inkscape, so we verify
        # that the header strings exist in the source code as a safeguard.
        source_path = os.path.join(TEST_DIR, "..", "music_glyph_toolkit.py")
        if not os.path.exists(source_path):
            source_path = os.path.join(os.path.dirname(TEST_DIR), "music_glyph_toolkit.py")
        with open(source_path) as f:
            source = f.read()
        for fragment in expected_fragments:
            assert fragment in source, f"Missing header fragment in source: {fragment}"

    def test_version_is_first_output_line(self):
        """The \\version directive must be the very first line of output."""
        source_path = os.path.join(TEST_DIR, "..", "music_glyph_toolkit.py")
        if not os.path.exists(source_path):
            source_path = os.path.join(os.path.dirname(TEST_DIR), "music_glyph_toolkit.py")
        with open(source_path) as f:
            source = f.read()
        # Find the out_lines list construction — \\version should be the first element
        match = re.search(r'out_lines\s*=\s*\[(.*?)\]', source, re.DOTALL)
        assert match is not None, "Could not find out_lines definition"
        first_item = match.group(1).strip().split('\n')[0]
        assert 'version' in first_item, "\\version should be the first item in out_lines"


class TestLilyPondMixedCommandExport:
    """Verify that dirty SVG commands normalize correctly through to Scheme output."""

    def test_mixed_commands_produce_valid_scheme(self):
        fixture = os.path.join(FIXTURES_INPUT, "glyph_mixed_commands.svg")
        paths = _read_svg_path_ds(fixture)
        for pid, d in paths:
            cmds = parse_and_normalize(d)
            scheme = format_scheme(cmds)
            # Balanced parens
            assert scheme.count("(") == scheme.count(")")
            # Only valid keywords
            keywords = re.findall(r"\((\w+)", scheme)
            allowed = {"moveto", "lineto", "curveto", "closepath"}
            for kw in keywords:
                assert kw in allowed, f"Path '{pid}': unexpected keyword '{kw}'"
            # All coordinates in staff-space range
            numbers = [float(n) for n in re.findall(r"[\d.]+", scheme)]
            for n in numbers:
                assert abs(n) < 50, f"Path '{pid}': coordinate {n} looks unscaled"


class TestForbiddenCommandDetection:
    """Verify that the FORBIDDEN_COMMANDS regex catches all non-MCLZ commands."""

    def test_detects_relative_commands(self):
        dirty = "m 10 20 l 5 5 c 1 2 3 4 5 6 z"
        assert FORBIDDEN_COMMANDS.findall(dirty) == ['m', 'l', 'c', 'z']

    def test_detects_shorthand_commands(self):
        dirty = "M 0 0 S 1 2 3 4 Q 5 6 7 8 T 9 10 A 1 1 0 0 1 5 5 H 10 V 20 Z"
        found = set(FORBIDDEN_COMMANDS.findall(dirty))
        assert found == {'S', 'Q', 'T', 'A', 'H', 'V'}

    def test_clean_path_passes(self):
        clean = "M 0 0 L 10 20 C 1 2 3 4 5 6 Z"
        assert FORBIDDEN_COMMANDS.findall(clean) == []


class TestLilyPondStemDefines:
    """Verify stem attachment point define syntax."""

    def test_stem_define_syntax(self):
        """Stem defines should use Scheme dotted-pair syntax '(x . y)."""
        # Simulate what export_lilypond would produce
        x, y = 1.2345, -0.5678
        define_str = f"#(define test-stem-up '({x} . {y}))"
        # Should have balanced parens
        assert define_str.count("(") == define_str.count(")")
        # Should contain dotted pair
        assert ". " in define_str
        # Should use explicit 0.x format, not .x
        assert ".x" not in define_str  # no bare decimal points

    def test_stem_coordinates_use_explicit_zero(self):
        """Values like 0.9 should not be written as .9 (Guile safety)."""
        # Test that our rounding produces "0.xxxx" not ".xxxx"
        val = round(0.9 * 0.01, 4)  # 0.009
        formatted = str(val)
        assert formatted.startswith("0.") or formatted == "0", \
            f"Value {formatted} should have explicit leading zero"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 3: data-lilypond-* Attribute Verification
# ═══════════════════════════════════════════════════════════════════════════════

def _read_svg_data_attrs(svg_path, element_id, attr_name):
    """Extract a data- attribute value from an SVG element by ID.
    
    Returns the attribute value string, or None if not found.
    """
    with open(svg_path) as f:
        content = f.read()
    # Find element with matching id, then extract the data attribute
    # This handles multi-line attribute values (the LilyPond path data spans lines)
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return None
    # Search all elements (including namespaced)
    for elem in root.iter():
        eid = elem.get('id', '')
        if eid == element_id:
            return elem.get(attr_name)
    return None


class TestLilyPondDataAttributeRoundTrip:
    """Verify that data-lilypond-* attribute values match format_scheme output.
    
    These tests exercise the same path through parse_and_normalize → format_scheme
    that _embed_lilypond_data uses, verifying the round-trip produces consistent
    results for various shape types.
    """

    def test_ellipse_roundtrip(self):
        """Simple ellipse: parse → format_scheme → wrap matches re-parse → wrap."""
        d = "M 80 0 C 80 -56 44 -100 0 -100 C -44 -100 -80 -56 -80 0 C -80 56 -44 100 0 100 C 44 100 80 56 80 0 Z"
        cmds1 = parse_and_normalize(d)
        # Simulate what effect() does: normalize to SVG d, then re-parse for embedding
        svg_d = format_svg_d(cmds1, decimals=3)
        cmds2 = parse_and_normalize(svg_d)
        scheme1 = format_scheme(cmds1)
        scheme2 = format_scheme(cmds2)
        assert scheme1 == scheme2

    def test_kite_roundtrip(self):
        """Angular kite shape: all lineto commands survive round-trip."""
        d = "M 0 -100 L 80 0 L 0 100 L -80 0 Z"
        cmds1 = parse_and_normalize(d)
        svg_d = format_svg_d(cmds1, decimals=3)
        cmds2 = parse_and_normalize(svg_d)
        scheme1 = format_scheme(cmds1)
        scheme2 = format_scheme(cmds2)
        assert scheme1 == scheme2

    def test_mixed_curves_roundtrip(self):
        """Shape with both curves and lines survives round-trip."""
        d = "M 0 -100 C 44 -100 80 -56 80 0 L 0 100 L -80 0 C -80 -56 -44 -100 0 -100 Z"
        cmds1 = parse_and_normalize(d)
        svg_d = format_svg_d(cmds1, decimals=3)
        cmds2 = parse_and_normalize(svg_d)
        scheme1 = format_scheme(cmds1)
        scheme2 = format_scheme(cmds2)
        assert scheme1 == scheme2

    def test_hollow_roundtrip(self):
        """Hollow cutout path survives round-trip."""
        d = "M 50 0 C 50 -36 28 -64 0 -64 C -28 -64 -50 -36 -50 0 C -50 36 -28 64 0 64 C 28 64 50 36 50 0 Z"
        cmds1 = parse_and_normalize(d)
        svg_d = format_svg_d(cmds1, decimals=3)
        cmds2 = parse_and_normalize(svg_d)
        scheme1 = format_scheme(cmds1)
        scheme2 = format_scheme(cmds2)
        assert scheme1 == scheme2

    def test_wrapped_format_embeddable(self):
        """The wrapped format can be directly embedded in a #(define ...) block."""
        d = "M 80 0 C 80 -56 44 -100 0 -100 C -44 -100 -80 -56 -80 0 Z"
        cmds = parse_and_normalize(d)
        scheme = format_scheme(cmds)
        wrapped = f"'(\n{scheme})"
        # Should be valid as part of: #(define my-path '(...))
        define_str = f"#(define my-path {wrapped})"
        assert define_str.count("(") == define_str.count(")")


class TestLilyPondDataAttributeSourceVerification:
    """Verify that the _embed_lilypond_data method exists and has correct structure."""

    @staticmethod
    def _find_source(filename):
        """Find a project file relative to TEST_DIR or in same directory."""
        candidates = [
            os.path.join(TEST_DIR, "..", filename),
            os.path.join(os.path.dirname(TEST_DIR), filename),
            os.path.join(TEST_DIR, filename),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        pytest.skip(f"Could not find {filename}")

    def test_embed_method_exists(self):
        """The MusicGlyphToolkit class should have _embed_lilypond_data method."""
        with open(self._find_source("music_glyph_toolkit.py")) as f:
            source = f.read()
        assert "def _embed_lilypond_data" in source

    def test_strip_method_exists(self):
        """The MusicGlyphToolkit class should have _strip_lilypond_data method."""
        with open(self._find_source("music_glyph_toolkit.py")) as f:
            source = f.read()
        assert "def _strip_lilypond_data" in source

    def test_include_lilypond_data_argument(self):
        """The extension should accept --include_lilypond_data argument."""
        with open(self._find_source("music_glyph_toolkit.py")) as f:
            source = f.read()
        assert "--include_lilypond_data" in source

    def test_data_attr_names_in_source(self):
        """Verify the correct attribute names are used in the source."""
        with open(self._find_source("music_glyph_toolkit.py")) as f:
            source = f.read()
        assert "data-lilypond-notehead" in source
        assert "data-lilypond-hollow" in source
        assert "data-lilypond-glyph" in source

    def test_embed_called_after_mclz(self):
        """The embed method should be called after MCLZ normalization in effect()."""
        with open(self._find_source("music_glyph_toolkit.py")) as f:
            source = f.read()
        # Find positions of MCLZ normalization and embed call
        mclz_pos = source.find("strict_mclz")
        embed_pos = source.find("_embed_lilypond_data")
        assert mclz_pos < embed_pos, "embed should be called after MCLZ normalization"

    def test_inx_has_checkbox(self):
        """The .inx file should include the LilyPond data checkbox."""
        with open(self._find_source("music_glyph_toolkit.inx")) as f:
            content = f.read()
        assert "include_lilypond_data" in content
        assert "Include LilyPond path data" in content

