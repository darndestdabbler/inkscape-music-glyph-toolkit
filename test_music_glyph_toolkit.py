"""
Tier 1 Unit Tests — Music Glyph Toolkit
Pure pytest tests with no Inkscape dependency.
Tests: parse_and_normalize, format_svg_d, format_scheme, arc_to_curves, quadratic_to_cubic
"""
import sys
import types
import math
import pytest

# --- Mock inkex so the module can be imported without Inkscape ---
_inkex_mock = types.ModuleType("inkex")
_inkex_mock.EffectExtension = type("EffectExtension", (), {})
_inkex_mock.PathElement = type("PathElement", (), {})
_inkex_mock.Boolean = bool
_inkex_mock.AbortExtension = Exception
_inkex_mock.Guide = type("Guide", (), {})
sys.modules["inkex"] = _inkex_mock
sys.modules["inkex.PathElement"] = _inkex_mock.PathElement
sys.modules["inkex.Guide"] = _inkex_mock.Guide

from music_glyph_toolkit import (
    parse_and_normalize,
    format_svg_d,
    format_scheme,
    arc_to_curves,
    quadratic_to_cubic,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def approx_cmd(cmd_tuple, expected_tuple, rel=1e-6):
    """Assert a single (cmd, args) tuple matches within tolerance."""
    cmd, args = cmd_tuple
    exp_cmd, exp_args = expected_tuple
    assert cmd == exp_cmd, f"Command mismatch: {cmd} != {exp_cmd}"
    assert len(args) == len(exp_args), f"Arg count mismatch for {cmd}: {len(args)} != {len(exp_args)}"
    for i, (a, e) in enumerate(zip(args, exp_args)):
        assert a == pytest.approx(e, rel=rel), (
            f"{cmd} arg[{i}]: {a} != {e} (within rel={rel})"
        )


def approx_cmds(result, expected, rel=1e-6):
    """Assert full command lists match within tolerance."""
    assert len(result) == len(expected), (
        f"Command count mismatch: {len(result)} != {len(expected)}\n"
        f"  Got:      {result}\n  Expected: {expected}"
    )
    for i, (r, e) in enumerate(zip(result, expected)):
        approx_cmd(r, e, rel=rel)


# ═══════════════════════════════════════════════════════════════════════════════
#  parse_and_normalize
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseBasicPassthrough:
    """Commands already in MCLZ absolute form pass through unchanged."""

    def test_m_l_z(self):
        result = parse_and_normalize("M 10 20 L 30 40 Z")
        approx_cmds(result, [
            ("M", [10, 20]),
            ("L", [30, 40]),
            ("Z", []),
        ])

    def test_cubic(self):
        result = parse_and_normalize("M 0 0 C 1 2 3 4 5 6 Z")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("C", [1, 2, 3, 4, 5, 6]),
            ("Z", []),
        ])

    def test_multiple_cubics(self):
        result = parse_and_normalize("M 0 0 C 1 2 3 4 5 6 C 7 8 9 10 11 12")
        assert len(result) == 3
        approx_cmd(result[2], ("C", [7, 8, 9, 10, 11, 12]))


class TestRelativeToAbsolute:
    """Relative commands (lowercase) convert to absolute."""

    def test_m_l(self):
        result = parse_and_normalize("m 10 20 l 5 5")
        approx_cmds(result, [
            ("M", [10, 20]),
            ("L", [15, 25]),
        ])

    def test_m_relative_then_absolute(self):
        result = parse_and_normalize("m 10 20 L 100 200")
        approx_cmds(result, [
            ("M", [10, 20]),
            ("L", [100, 200]),
        ])

    def test_relative_cubic(self):
        result = parse_and_normalize("M 10 10 c 1 2 3 4 5 6")
        approx_cmds(result, [
            ("M", [10, 10]),
            ("C", [11, 12, 13, 14, 15, 16]),
        ])

    def test_chained_relative_l(self):
        result = parse_and_normalize("M 0 0 l 10 0 10 0 10 0")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [10, 0]),
            ("L", [20, 0]),
            ("L", [30, 0]),
        ])

    def test_relative_arc(self):
        result = parse_and_normalize("M 100 100 a 25 25 0 0 1 50 0")
        # Should produce curves ending at absolute (150, 100)
        last_cmd, last_args = result[-1]
        assert last_cmd == "C"
        assert last_args[-2] == pytest.approx(150, rel=1e-4)
        assert last_args[-1] == pytest.approx(100, rel=1e-4)


class TestHVExpansion:
    """H and V commands expand to L."""

    def test_absolute_h_v(self):
        result = parse_and_normalize("M 0 0 H 100 V 50")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [100, 0]),
            ("L", [100, 50]),
        ])

    def test_relative_h_v(self):
        result = parse_and_normalize("M 10 20 h 5 v 10")
        approx_cmds(result, [
            ("M", [10, 20]),
            ("L", [15, 20]),
            ("L", [15, 30]),
        ])

    def test_chained_h(self):
        result = parse_and_normalize("M 0 0 H 10 20 30")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [10, 0]),
            ("L", [20, 0]),
            ("L", [30, 0]),
        ])

    def test_chained_v(self):
        result = parse_and_normalize("M 0 0 V 10 20 30")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [0, 10]),
            ("L", [0, 20]),
            ("L", [0, 30]),
        ])


class TestImplicitLineto:
    """Extra coordinate pairs after M/m become implicit L commands."""

    def test_implicit_l_after_M(self):
        result = parse_and_normalize("M 0 0 10 20 30 40")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [10, 20]),
            ("L", [30, 40]),
        ])

    def test_implicit_l_after_m(self):
        result = parse_and_normalize("m 0 0 10 20 30 40")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [10, 20]),
            ("L", [40, 60]),  # relative: (10+30, 20+40)
        ])

    def test_implicit_repeat_after_bare_numbers(self):
        """Numbers without a preceding command use last_command (implicit L after M)."""
        result = parse_and_normalize("M 0 0 L 10 10 20 20")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [10, 10]),
            ("L", [20, 20]),
        ])


class TestZResetsCurrentPoint:
    """Z closes the subpath and resets current point to the subpath start."""

    def test_z_resets_for_relative(self):
        result = parse_and_normalize("M 10 10 L 20 20 Z l 5 0")
        approx_cmds(result, [
            ("M", [10, 10]),
            ("L", [20, 20]),
            ("Z", []),
            ("L", [15, 10]),  # relative to (10,10), not (20,20)
        ])

    def test_z_resets_for_next_m_relative(self):
        result = parse_and_normalize("M 10 10 L 50 50 Z m 5 5")
        approx_cmds(result, [
            ("M", [10, 10]),
            ("L", [50, 50]),
            ("Z", []),
            ("M", [15, 15]),  # relative to (10,10)
        ])


class TestSmoothCubic:
    """S/s smooth cubic — reflects CP2 of previous C/S over current point."""

    def test_s_after_c(self):
        result = parse_and_normalize("M 0 0 C 10 20 30 40 50 50 S 70 60 90 50")
        # Reflected CP1 = 2*(50,50) - (30,40) = (70, 60)
        approx_cmds(result, [
            ("M", [0, 0]),
            ("C", [10, 20, 30, 40, 50, 50]),
            ("C", [70, 60, 70, 60, 90, 50]),
        ])

    def test_s_after_non_cubic(self):
        """S after L: no reflection, CP1 = current point."""
        result = parse_and_normalize("M 0 0 L 50 50 S 70 60 90 50")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [50, 50]),
            ("C", [50, 50, 70, 60, 90, 50]),
        ])

    def test_s_chained_implicit_bug1(self):
        """Bug 1 regression test: chained implicit S after L.
        
        First S after L: no reflection (correct).
        Second implicit S: MUST reflect because last_command is now S.
        """
        result = parse_and_normalize("M 0 0 L 10 10 S 30 40 50 50 70 60 90 50")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("L", [10, 10]),
            ("C", [10, 10, 30, 40, 50, 50]),       # no reflection (after L)
            ("C", [70, 60, 70, 60, 90, 50]),        # reflected: 2*(50,50)-(30,40) = (70,60)
        ])

    def test_s_chained_implicit_after_c(self):
        """Chained implicit S after C — both segments should reflect."""
        result = parse_and_normalize("M 0 0 C 0 0 10 20 30 40 S 50 60 70 80 90 100 110 120")
        # First S: reflect (10,20) over (30,40) -> (50, 60)
        approx_cmd(result[2], ("C", [50, 60, 50, 60, 70, 80]))
        # Second S: reflect (50,60) over (70,80) -> (90, 100)
        approx_cmd(result[3], ("C", [90, 100, 90, 100, 110, 120]))

    def test_relative_s(self):
        result = parse_and_normalize("M 0 0 C 0 0 10 20 30 40 s 20 20 40 10")
        # Reflected CP1 = 2*(30,40) - (10,20) = (50, 60) [absolute]
        # CP2 = (30+20, 40+20) = (50, 60), end = (30+40, 40+10) = (70, 50)
        approx_cmd(result[2], ("C", [50, 60, 50, 60, 70, 50]))


class TestSmoothQuadratic:
    """T/t smooth quadratic — reflects control point of previous Q/T."""

    def test_t_after_q(self):
        result = parse_and_normalize("M 0 0 Q 50 100 100 0 T 200 0")
        # After Q: last_control = (50, 100), current = (100, 0)
        # Reflected control = 2*(100,0) - (50,100) = (150, -100)
        # Then quadratic_to_cubic with (100,0) (150,-100) (200,0)
        cmd, args = result[2]
        assert cmd == "C"
        # Verify endpoint
        assert args[4] == pytest.approx(200)
        assert args[5] == pytest.approx(0)

    def test_t_after_non_quadratic(self):
        """T after L: no reflection, control point = current point (degenerate)."""
        result = parse_and_normalize("M 0 0 L 100 0 T 200 0")
        cmd, args = result[2]
        assert cmd == "C"
        assert args[4] == pytest.approx(200)
        assert args[5] == pytest.approx(0)

    def test_t_chained_implicit_bug1(self):
        """Bug 1 regression for T: chained implicit T after L.
        
        First T after L: no reflection.
        Second implicit T: MUST reflect because last_command is now T.
        """
        result = parse_and_normalize("M 0 0 L 50 0 T 100 0 150 0")
        # First T after L: control = (50,0) [current point, no reflection]
        # -> cubic from (50,0) through ctrl(50,0) to (100,0) = straight line
        # Second T: last_control was (50,0), current is (100,0)
        # reflected = 2*(100,0) - (50,0) = (150, 0)
        # -> cubic from (100,0) through ctrl(150,0) to (150,0)
        assert len(result) == 4  # M, L, C, C
        assert result[2][0] == "C"
        assert result[3][0] == "C"
        # Second T endpoint
        assert result[3][1][4] == pytest.approx(150)
        assert result[3][1][5] == pytest.approx(0)

    def test_relative_t(self):
        result = parse_and_normalize("M 0 0 Q 50 100 100 0 t 100 0")
        # t 100 0 -> absolute endpoint (200, 0)
        cmd, args = result[2]
        assert cmd == "C"
        assert args[4] == pytest.approx(200)
        assert args[5] == pytest.approx(0)


class TestQuadraticToCubic:
    """Q/q quadratic Bézier elevated to cubic via 2/3 rule."""

    def test_q_to_c_known_values(self):
        result = parse_and_normalize("M 0 0 Q 50 100 100 0")
        approx_cmds(result, [
            ("M", [0, 0]),
            ("C", [
                100 / 3, 200 / 3,     # CP1: 0 + 2/3*(50-0), 0 + 2/3*(100-0)
                200 / 3, 200 / 3,     # CP2: 100 + 2/3*(50-100), 0 + 2/3*(100-0)
                100, 0                 # endpoint
            ]),
        ])

    def test_relative_q(self):
        result = parse_and_normalize("M 10 10 q 40 90 90 -10")
        # Absolute: control=(50,100), end=(100,0)
        approx_cmds(result, [
            ("M", [10, 10]),
            ("C", [
                10 + 2 / 3 * 40, 10 + 2 / 3 * 90,
                100 + 2 / 3 * (50 - 100), 0 + 2 / 3 * (100 - 0),
                100, 0,
            ]),
        ])

    def test_chained_q(self):
        """Multiple implicit Q segments."""
        result = parse_and_normalize("M 0 0 Q 25 50 50 0 75 50 100 0")
        assert len(result) == 3  # M, C, C
        assert result[1][0] == "C"
        assert result[2][0] == "C"
        assert result[2][1][4] == pytest.approx(100)
        assert result[2][1][5] == pytest.approx(0)


class TestArcConversion:
    """A/a arc commands convert to cubic Bézier approximations."""

    def test_semicircle(self):
        result = parse_and_normalize("M 0 0 A 25 25 0 0 1 50 0")
        # Semicircle r=25 from (0,0) to (50,0) — should produce 2 cubic segments
        # (pi radians, split at pi/2 each)
        assert result[0] == ("M", [0, 0])
        cubics = [r for r in result[1:] if r[0] == "C"]
        assert len(cubics) == 2
        # Final endpoint should be (50, 0)
        assert cubics[-1][1][-2] == pytest.approx(50, rel=1e-4)
        assert cubics[-1][1][-1] == pytest.approx(0, abs=1e-4)

    def test_quarter_circle(self):
        result = parse_and_normalize("M 25 0 A 25 25 0 0 1 0 25")
        cubics = [r for r in result[1:] if r[0] == "C"]
        assert len(cubics) == 1  # quarter arc = 1 segment
        assert cubics[0][1][-2] == pytest.approx(0, abs=1e-4)
        assert cubics[0][1][-1] == pytest.approx(25, rel=1e-4)

    def test_full_circle_two_arcs(self):
        """Full circle needs two arc commands (single arc can't close on itself)."""
        result = parse_and_normalize(
            "M 0 0 A 25 25 0 0 1 50 0 A 25 25 0 0 1 0 0"
        )
        cubics = [r for r in result if r[0] == "C"]
        assert len(cubics) == 4  # 2 semicircles × 2 segments each
        # Endpoint should return to origin
        assert cubics[-1][1][-2] == pytest.approx(0, abs=1e-3)
        assert cubics[-1][1][-1] == pytest.approx(0, abs=1e-3)

    def test_elliptical_arc(self):
        result = parse_and_normalize("M 0 0 A 50 25 0 0 1 100 0")
        cubics = [r for r in result if r[0] == "C"]
        assert len(cubics) >= 1
        assert cubics[-1][1][-2] == pytest.approx(100, rel=1e-4)
        assert cubics[-1][1][-1] == pytest.approx(0, abs=1e-4)

    def test_elliptical_arc_with_rotation(self):
        result = parse_and_normalize("M 0 0 A 50 25 45 0 1 100 0")
        cubics = [r for r in result if r[0] == "C"]
        assert len(cubics) >= 1
        assert cubics[-1][1][-2] == pytest.approx(100, rel=1e-4)
        assert cubics[-1][1][-1] == pytest.approx(0, abs=1e-3)

    def test_large_arc_flag(self):
        """large_arc=1 takes the long way around."""
        short = parse_and_normalize("M 0 0 A 25 25 0 0 1 50 0")
        long_ = parse_and_normalize("M 0 0 A 25 25 0 1 1 50 0")
        short_cubics = [r for r in short if r[0] == "C"]
        long_cubics = [r for r in long_ if r[0] == "C"]
        # Long arc should produce more segments (or different path)
        assert len(long_cubics) >= len(short_cubics)


class TestArcEdgeCases:
    """Edge cases for arc_to_curves directly."""

    def test_zero_radius(self):
        result = arc_to_curves(0, 0, 0, 25, 0, False, True, 50, 0)
        assert result == [("L", [50, 0])]

    def test_zero_rx(self):
        result = arc_to_curves(0, 0, 0, 25, 0, False, True, 50, 0)
        assert result == [("L", [50, 0])]

    def test_zero_ry(self):
        result = arc_to_curves(0, 0, 25, 0, 0, False, True, 50, 0)
        assert result == [("L", [50, 0])]

    def test_coincident_endpoints(self):
        result = arc_to_curves(10, 20, 25, 25, 0, False, True, 10, 20)
        assert result == []

    def test_negative_radii_treated_as_positive(self):
        result = arc_to_curves(0, 0, -25, -25, 0, False, True, 50, 0)
        assert len(result) > 0
        assert all(r[0] == "C" for r in result)
        assert result[-1][1][-2] == pytest.approx(50, rel=1e-4)
        assert result[-1][1][-1] == pytest.approx(0, abs=1e-4)


class TestQuadraticToCubicDirect:
    """Direct tests of the quadratic_to_cubic helper."""

    def test_known_values(self):
        # Q from (0,0) ctrl (50,100) end (100,0)
        cx1, cy1, cx2, cy2, ex, ey = quadratic_to_cubic(0, 0, 50, 100, 100, 0)
        assert cx1 == pytest.approx(100 / 3)
        assert cy1 == pytest.approx(200 / 3)
        assert cx2 == pytest.approx(200 / 3)
        assert cy2 == pytest.approx(200 / 3)
        assert ex == pytest.approx(100)
        assert ey == pytest.approx(0)

    def test_straight_line(self):
        """Q with collinear control point produces a cubic straight line."""
        cx1, cy1, cx2, cy2, ex, ey = quadratic_to_cubic(0, 0, 50, 0, 100, 0)
        assert cy1 == pytest.approx(0)
        assert cy2 == pytest.approx(0)
        assert ey == pytest.approx(0)

    def test_symmetric(self):
        """Symmetric quadratic: CP1 and CP2 of cubic should be symmetric about midpoint."""
        cx1, cy1, cx2, cy2, ex, ey = quadratic_to_cubic(0, 0, 50, 100, 100, 0)
        mid_x = (cx1 + cx2) / 2
        mid_y = (cy1 + cy2) / 2
        assert mid_x == pytest.approx(50)
        assert mid_y == pytest.approx(200 / 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  format_svg_d
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatSvgD:

    def test_basic_roundtrip(self):
        cmds = [("M", [10.0, 20.0]), ("L", [30.0, 40.0]), ("Z", [])]
        assert format_svg_d(cmds) == "M 10.0 20.0 L 30.0 40.0 Z"

    def test_rounding(self):
        cmds = [("M", [10.12345, 20.6789])]
        assert format_svg_d(cmds, decimals=2) == "M 10.12 20.68"

    def test_rounding_zero_decimals(self):
        cmds = [("M", [10.7, 20.3])]
        result = format_svg_d(cmds, decimals=0)
        assert result == "M 11.0 20.0"

    def test_cubic_formatting(self):
        cmds = [("C", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
        assert format_svg_d(cmds) == "C 1.0 2.0 3.0 4.0 5.0 6.0"

    def test_z_no_trailing_space(self):
        cmds = [("M", [0.0, 0.0]), ("Z", [])]
        result = format_svg_d(cmds)
        assert result == "M 0.0 0.0 Z"
        assert not result.endswith(" ")

    def test_full_path(self):
        """Full parse → format round-trip produces clean output."""
        original = "M 0 0 L 100 0 L 100 100 L 0 100 Z"
        cmds = parse_and_normalize(original)
        result = format_svg_d(cmds, decimals=3)
        assert result == "M 0.0 0.0 L 100.0 0.0 L 100.0 100.0 L 0.0 100.0 Z"


# ═══════════════════════════════════════════════════════════════════════════════
#  format_scheme
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatScheme:

    def test_y_inversion(self):
        cmds = [("M", [0.0, 100.0])]
        result = format_scheme(cmds, scale=1.0)
        assert "(moveto 0.0 -100.0)" in result

    def test_scaling(self):
        cmds = [("M", [100.0, 200.0])]
        result = format_scheme(cmds, scale=0.01)
        assert "(moveto 1.0 -2.0)" in result

    def test_all_command_types(self):
        cmds = [
            ("M", [100.0, 200.0]),
            ("L", [300.0, 400.0]),
            ("C", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            ("Z", []),
        ]
        result = format_scheme(cmds, scale=0.01)
        assert "(moveto" in result
        assert "(lineto" in result
        assert "(curveto" in result
        assert "(closepath)" in result

    def test_scaling_and_inversion_on_cubic(self):
        cmds = [("C", [100.0, 200.0, 300.0, 400.0, 500.0, 600.0])]
        result = format_scheme(cmds, scale=0.01, decimals=2)
        assert "(curveto 1.0 -2.0 3.0 -4.0 5.0 -6.0)" in result

    def test_default_scale_is_001(self):
        cmds = [("M", [100.0, 0.0])]
        result = format_scheme(cmds)
        assert "(moveto 1.0 -0.0)" in result or "(moveto 1.0 0.0)" in result

    def test_custom_decimals(self):
        cmds = [("M", [100.0, 333.3333])]
        result = format_scheme(cmds, scale=0.01, decimals=2)
        assert "(moveto 1.0 -3.33)" in result

    def test_indentation(self):
        """Each line is indented with 4 spaces."""
        cmds = [("M", [0.0, 0.0]), ("L", [100.0, 100.0]), ("Z", [])]
        result = format_scheme(cmds, scale=0.01)
        for line in result.split("\n"):
            if line:  # skip empty lines
                assert line.startswith("    ")

    def test_negative_y_becomes_positive(self):
        """SVG negative Y → LilyPond positive Y after negation."""
        cmds = [("M", [0.0, -100.0])]
        result = format_scheme(cmds, scale=0.01)
        assert "(moveto 0.0 1.0)" in result


# ═══════════════════════════════════════════════════════════════════════════════
#  End-to-end pipeline sanity checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Full pipeline: raw SVG d → parse_and_normalize → formatter."""

    def test_complex_path_to_svg(self):
        """Path using many command types normalizes to clean MCLZ."""
        d = "M 0 0 h 100 v 100 H 0 V 0 Z"
        cmds = parse_and_normalize(d)
        svg_d = format_svg_d(cmds, decimals=1)
        # Should be a 100×100 square
        assert "M 0.0 0.0" in svg_d
        assert "L 100.0 0.0" in svg_d
        assert "L 100.0 100.0" in svg_d
        assert "L 0.0 100.0" in svg_d
        assert "L 0.0 0.0" in svg_d
        assert svg_d.endswith("Z")

    def test_complex_path_to_scheme(self):
        """Path with Q and S normalizes and exports to Scheme with scaling."""
        d = "M 0 0 Q 50 100 100 0 S 150 100 200 0"
        cmds = parse_and_normalize(d)
        scheme = format_scheme(cmds, scale=0.01)
        assert "(moveto 0.0" in scheme
        assert "(curveto" in scheme
        assert "(closepath)" not in scheme  # no Z in input

    def test_only_mclz_in_output(self):
        """Regardless of input, output commands are only M, C, L, Z."""
        d = "M 0 0 Q 50 100 100 0 T 200 0 S 300 100 400 0 A 25 25 0 0 1 450 0 H 500 V 100 Z"
        cmds = parse_and_normalize(d)
        allowed = {"M", "C", "L", "Z"}
        for cmd, _ in cmds:
            assert cmd in allowed, f"Unexpected command {cmd} in normalized output"

    def test_svg_roundtrip_stability(self):
        """Normalizing an already-normalized path produces identical output."""
        d = "M 0 0 L 100 0 C 100 50 50 100 0 100 Z"
        cmds1 = parse_and_normalize(d)
        svg1 = format_svg_d(cmds1, decimals=3)
        cmds2 = parse_and_normalize(svg1)
        svg2 = format_svg_d(cmds2, decimals=3)
        assert svg1 == svg2
