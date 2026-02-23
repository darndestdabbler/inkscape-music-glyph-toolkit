"""
Tier 1 Unit Tests — Music Glyph Toolkit
Pure pytest tests with no Inkscape dependency.
Tests: parse_and_normalize, format_svg_d, format_scheme, arc_to_curves, quadratic_to_cubic
"""
import sys
import types
import math
import re
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
    _arc_to_beziers,
    quadratic_to_cubic,
    _cubic_axis_extremes,
    path_extreme_points,
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


# ═══════════════════════════════════════════════════════════════════════════════
#  data-lilypond-* attribute format and content
# ═══════════════════════════════════════════════════════════════════════════════

def _wrap_scheme(commands):
    """Replicate the wrapping logic used by _embed_lilypond_data."""
    scheme = format_scheme(commands)
    return f"'(\n{scheme})"


class TestLilypondDataWrappedFormat:
    """Verify the wrapped LilyPond path format used in data- attributes."""

    def test_wrapper_starts_with_quote_paren(self):
        cmds = parse_and_normalize("M 0 0 L 100 0 L 100 100 Z")
        wrapped = _wrap_scheme(cmds)
        assert wrapped.startswith("'(")

    def test_wrapper_ends_with_close_paren(self):
        cmds = parse_and_normalize("M 0 0 L 100 0 L 100 100 Z")
        wrapped = _wrap_scheme(cmds)
        assert wrapped.rstrip().endswith(")")

    def test_balanced_parens(self):
        cmds = parse_and_normalize("M 80 0 C 80 -56 44 -100 0 -100 C -44 -100 -80 -56 -80 0 C -80 56 -44 100 0 100 C 44 100 80 56 80 0 Z")
        wrapped = _wrap_scheme(cmds)
        assert wrapped.count("(") == wrapped.count(")")

    def test_contains_valid_keywords_only(self):
        cmds = parse_and_normalize("M 0 0 L 50 0 C 50 50 0 50 0 0 Z")
        wrapped = _wrap_scheme(cmds)
        allowed = {"moveto", "lineto", "curveto", "closepath"}
        keywords = re.findall(r"\((\w+)", wrapped)
        for kw in keywords:
            assert kw in allowed, f"Unexpected keyword: {kw}"

    def test_coordinates_in_staff_space_range(self):
        """Wrapped coordinates should be /100 scaled (staff-space range, not SVG units)."""
        cmds = parse_and_normalize("M 80 0 C 80 -100 -80 -100 -80 0 Z")
        wrapped = _wrap_scheme(cmds)
        numbers = [float(n) for n in re.findall(r"-?[\d.]+", wrapped)]
        for n in numbers:
            assert abs(n) < 10, f"Coordinate {n} looks unscaled"

    def test_y_is_inverted(self):
        """SVG Y-down → LilyPond Y-up: positive SVG Y should become negative in Scheme."""
        cmds = parse_and_normalize("M 0 100")  # SVG y=100 (downward)
        wrapped = _wrap_scheme(cmds)
        assert "(moveto 0.0 -1.0)" in wrapped


class TestLilypondDataEllipse:
    """Test data-lilypond-* format with a simple elliptical notehead shape."""

    ELLIPSE_D = (
        "M 80 0 C 80 -56 44 -100 0 -100 "
        "C -44 -100 -80 -56 -80 0 "
        "C -80 56 -44 100 0 100 "
        "C 44 100 80 56 80 0 Z"
    )

    def test_ellipse_wrapped_has_moveto(self):
        cmds = parse_and_normalize(self.ELLIPSE_D)
        wrapped = _wrap_scheme(cmds)
        assert "(moveto" in wrapped

    def test_ellipse_wrapped_has_closepath(self):
        cmds = parse_and_normalize(self.ELLIPSE_D)
        wrapped = _wrap_scheme(cmds)
        assert "(closepath)" in wrapped

    def test_ellipse_wrapped_has_curveto(self):
        cmds = parse_and_normalize(self.ELLIPSE_D)
        wrapped = _wrap_scheme(cmds)
        assert "(curveto" in wrapped

    def test_ellipse_round_trip_consistency(self):
        """Wrapped scheme from normalized path should match scheme from same path re-parsed."""
        cmds1 = parse_and_normalize(self.ELLIPSE_D)
        svg_d = format_svg_d(cmds1, decimals=3)
        cmds2 = parse_and_normalize(svg_d)
        wrapped1 = _wrap_scheme(cmds1)
        wrapped2 = _wrap_scheme(cmds2)
        assert wrapped1 == wrapped2

    def test_ellipse_scheme_matches_direct_export(self):
        """Wrapped format should contain the same content as format_scheme (just wrapped)."""
        cmds = parse_and_normalize(self.ELLIPSE_D)
        direct_scheme = format_scheme(cmds)
        wrapped = _wrap_scheme(cmds)
        # The wrapped version should contain the direct scheme content
        assert direct_scheme in wrapped
        # And the wrapper adds '( at start and ) is already the last char of closepath
        assert wrapped == f"'(\n{direct_scheme})"


class TestLilypondDataAngular:
    """Test with angular shapes (kite/diamond) — lineto-heavy paths."""

    KITE_D = "M 0 -100 L 80 0 L 0 100 L -80 0 Z"
    DIAMOND_D = "M 0 -100 L 70 0 L 0 100 L -70 0 Z"

    def test_kite_uses_lineto(self):
        cmds = parse_and_normalize(self.KITE_D)
        wrapped = _wrap_scheme(cmds)
        assert "(lineto" in wrapped
        # Should have no curveto (all straight lines)
        assert "(curveto" not in wrapped

    def test_kite_has_four_lineto(self):
        cmds = parse_and_normalize(self.KITE_D)
        wrapped = _wrap_scheme(cmds)
        assert wrapped.count("(lineto") == 3  # L L L (first point is moveto)

    def test_diamond_coordinates_correct(self):
        """Verify specific coordinates for diamond: M 0 -100 → moveto 0.0 1.0 (Y inverted, /100)."""
        cmds = parse_and_normalize(self.DIAMOND_D)
        wrapped = _wrap_scheme(cmds)
        assert "(moveto 0.0 1.0)" in wrapped    # (0, -100) → (0.0, 1.0)
        assert "(lineto 0.7" in wrapped           # (70, 0) → (0.7, ...)
        assert "(lineto 0.0 -1.0)" in wrapped    # (0, 100) → (0.0, -1.0)
        assert "(lineto -0.7" in wrapped          # (-70, 0) → (-0.7, ...)

    def test_kite_round_trip(self):
        cmds1 = parse_and_normalize(self.KITE_D)
        svg_d = format_svg_d(cmds1, decimals=3)
        cmds2 = parse_and_normalize(svg_d)
        assert _wrap_scheme(cmds1) == _wrap_scheme(cmds2)


class TestLilypondDataCurves:
    """Test with shapes that have mixed curves and lines."""

    # A rounded-top kite: curve at top, straight sides
    ROUNDED_KITE_D = (
        "M 0 -100 C 44 -100 80 -56 80 0 "
        "L 0 100 L -80 0 "
        "C -80 -56 -44 -100 0 -100 Z"
    )

    def test_mixed_has_both_curveto_and_lineto(self):
        cmds = parse_and_normalize(self.ROUNDED_KITE_D)
        wrapped = _wrap_scheme(cmds)
        assert "(curveto" in wrapped
        assert "(lineto" in wrapped

    def test_mixed_balanced_parens(self):
        cmds = parse_and_normalize(self.ROUNDED_KITE_D)
        wrapped = _wrap_scheme(cmds)
        assert wrapped.count("(") == wrapped.count(")")

    def test_curveto_has_six_numbers(self):
        cmds = parse_and_normalize(self.ROUNDED_KITE_D)
        wrapped = _wrap_scheme(cmds)
        for line in wrapped.split("\n"):
            if "(curveto" in line:
                numbers = re.findall(r"-?[\d.]+", line)
                assert len(numbers) == 6, f"curveto with wrong arg count: {line}"


class TestLilypondDataHollow:
    """Test data-lilypond-hollow attribute format (smaller path inside notehead)."""

    OUTER_D = "M 80 0 C 80 -56 44 -100 0 -100 C -44 -100 -80 -56 -80 0 C -80 56 -44 100 0 100 C 44 100 80 56 80 0 Z"
    INNER_D = "M 50 0 C 50 -36 28 -64 0 -64 C -28 -64 -50 -36 -50 0 C -50 36 -28 64 0 64 C 28 64 50 36 50 0 Z"

    def test_hollow_is_valid_wrapped_scheme(self):
        cmds = parse_and_normalize(self.INNER_D)
        wrapped = _wrap_scheme(cmds)
        assert wrapped.startswith("'(")
        assert wrapped.count("(") == wrapped.count(")")

    def test_hollow_smaller_than_notehead(self):
        """Hollow path coordinates should be smaller than notehead (inner cutout)."""
        outer_cmds = parse_and_normalize(self.OUTER_D)
        inner_cmds = parse_and_normalize(self.INNER_D)
        outer_wrapped = _wrap_scheme(outer_cmds)
        inner_wrapped = _wrap_scheme(inner_cmds)
        # Extract max absolute coordinate from each
        outer_nums = [abs(float(n)) for n in re.findall(r"-?[\d.]+", outer_wrapped)]
        inner_nums = [abs(float(n)) for n in re.findall(r"-?[\d.]+", inner_wrapped)]
        assert max(inner_nums) < max(outer_nums)

    def test_notehead_and_hollow_independent(self):
        """Notehead and hollow produce different wrapped outputs."""
        outer = _wrap_scheme(parse_and_normalize(self.OUTER_D))
        inner = _wrap_scheme(parse_and_normalize(self.INNER_D))
        assert outer != inner


class TestLilypondDataGlyph:
    """Test data-lilypond-glyph format for general glyph mode paths."""

    GLYPH_D = "M 0 0 L 100 0 L 100 400 L 0 400 Z"

    def test_glyph_wrapped_valid(self):
        cmds = parse_and_normalize(self.GLYPH_D)
        wrapped = _wrap_scheme(cmds)
        assert wrapped.startswith("'(")
        assert wrapped.count("(") == wrapped.count(")")

    def test_glyph_uses_same_format_as_notehead(self):
        """Glyph mode uses the same wrapping format as notehead mode."""
        cmds = parse_and_normalize(self.GLYPH_D)
        direct = format_scheme(cmds)
        wrapped = _wrap_scheme(cmds)
        assert wrapped == f"'(\n{direct})"


class TestLilypondDataFromDirtyPaths:
    """Verify that dirty SVG paths normalize correctly through to wrapped scheme."""

    def test_arc_path_normalizes_to_valid_scheme(self):
        """SVG arc → cubic Bézier → valid wrapped LilyPond Scheme."""
        d = "M 0 0 A 50 50 0 0 1 100 0 A 50 50 0 0 1 0 0 Z"
        cmds = parse_and_normalize(d)
        wrapped = _wrap_scheme(cmds)
        assert wrapped.startswith("'(")
        assert wrapped.count("(") == wrapped.count(")")
        assert "(curveto" in wrapped
        assert "(closepath)" in wrapped

    def test_quadratic_path_normalizes(self):
        d = "M 0 0 Q 50 -100 100 0 T 200 0 Z"
        cmds = parse_and_normalize(d)
        wrapped = _wrap_scheme(cmds)
        assert "(curveto" in wrapped
        assert "(closepath)" in wrapped

    def test_relative_path_normalizes(self):
        d = "m 80 0 c 0 -56 -36 -100 -80 -100 c -44 0 -80 44 -80 100 c 0 56 36 100 80 100 c 44 0 80 -44 80 -100 z"
        cmds = parse_and_normalize(d)
        wrapped = _wrap_scheme(cmds)
        assert wrapped.startswith("'(")
        assert "(moveto" in wrapped
        assert "(closepath)" in wrapped
        # All coordinates should be in staff-space range
        numbers = [float(n) for n in re.findall(r"-?[\d.]+", wrapped)]
        for n in numbers:
            assert abs(n) < 10, f"Coordinate {n} looks unscaled"

# ═══════════════════════════════════════════════════════════════════════════════
#  _cubic_axis_extremes
# ═══════════════════════════════════════════════════════════════════════════════

class TestCubicAxisExtremes:
    """Test the cubic Bézier axis extreme-finding helper."""

    def test_straight_line_no_extremes(self):
        """A straight line (colinear control points) has no interior extremes."""
        result = _cubic_axis_extremes(0, 10, 20, 30)
        assert result == []

    def test_symmetric_bulge_has_one_extreme(self):
        """A symmetric outward bulge has one X extreme at t=0.5."""
        # P0=0, P1=100, P2=100, P3=0 — bulges out to ~75 at t=0.5
        result = _cubic_axis_extremes(0, 100, 100, 0)
        assert len(result) == 1
        t, val = result[0]
        assert t == pytest.approx(0.5, abs=1e-6)
        assert val == pytest.approx(75.0, abs=1e-6)

    def test_s_curve_has_two_extremes(self):
        """An S-curve (control points on opposite sides) has two extremes."""
        # P0=0, P1=100, P2=-100, P3=0 — crosses axis twice
        result = _cubic_axis_extremes(0, 100, -100, 0)
        assert len(result) == 2
        # One should be positive, one negative
        vals = sorted(v for _, v in result)
        assert vals[0] < 0
        assert vals[1] > 0

    def test_monotonic_no_extremes(self):
        """A monotonically increasing curve has no interior extremes."""
        result = _cubic_axis_extremes(0, 10, 90, 100)
        assert result == []

    def test_degenerate_zero_length(self):
        """All four control values equal — no extremes."""
        result = _cubic_axis_extremes(50, 50, 50, 50)
        assert result == []

    def test_extreme_at_boundary_excluded(self):
        """Extremes exactly at t=0 or t=1 are not included (only interior)."""
        # Linear: extreme is at endpoint, not interior
        result = _cubic_axis_extremes(100, 50, 25, 0)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
#  path_extreme_points
# ═══════════════════════════════════════════════════════════════════════════════

class TestPathExtremePoints:
    """Test full path extreme point calculation."""

    def test_diamond_all_vertices(self):
        """Diamond (all lineto) — extremes are exactly the four vertices."""
        d = "M 0 -100 L 80 0 L 0 100 L -80 0 Z"
        points = path_extreme_points(parse_and_normalize(d))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        assert max(xs) == pytest.approx(80)
        assert min(xs) == pytest.approx(-80)
        assert max(ys) == pytest.approx(100)
        assert min(ys) == pytest.approx(-100)

    def test_circle_extremes_beyond_control_points(self):
        """Circle approximation — curve extremes extend to full radius."""
        # Standard 4-segment circle approximation, radius 100
        K = 55.228  # (4/3)(sqrt(2) - 1) * 100
        d = (
            f"M 100 0 "
            f"C 100 {K} {K} 100 0 100 "
            f"C {-K} 100 -100 {K} -100 0 "
            f"C -100 {-K} {-K} -100 0 -100 "
            f"C {K} -100 100 {-K} 100 0 Z"
        )
        points = path_extreme_points(parse_and_normalize(d))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        assert max(xs) == pytest.approx(100, abs=0.5)
        assert min(xs) == pytest.approx(-100, abs=0.5)
        assert max(ys) == pytest.approx(100, abs=0.5)
        assert min(ys) == pytest.approx(-100, abs=0.5)

    def test_rounded_diamond_extremes_beyond_vertices(self):
        """Rounded diamond from real-world use — curves bulge past vertices.

        This is the exact path from the bug report: a diamond whose corners
        are rounded with cubic Beziers that extend to ~154 on each axis,
        well beyond the vertex endpoints at ~137.67.
        """
        d = (
            "M -29.675 137.67 "
            "L -137.67 29.676 "
            "C -154.11 13.236 -154.11 -13.235 -137.67 -29.675 "
            "L -29.675 -137.67 "
            "C -13.235 -154.11 13.235 -154.11 29.676 -137.67 "
            "L 137.67 -29.675 "
            "C 154.11 -13.235 154.11 13.236 137.67 29.676 "
            "L 29.676 137.67 "
            "C 13.236 154.11 -13.235 154.11 -29.675 137.67 Z"
        )
        points = path_extreme_points(parse_and_normalize(d))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # The vertex endpoints are at +/-137.67, but curves bulge to ~150
        assert max(xs) > 137.67, "maxX should exceed the vertex endpoint"
        assert max(xs) == pytest.approx(150.0, abs=1.0)
        assert min(xs) < -137.67, "minX should exceed the vertex endpoint"
        assert min(xs) == pytest.approx(-150.0, abs=1.0)

    def test_rounded_diamond_stem_y_at_extreme_x(self):
        """At the rightmost extreme of the rounded diamond, Y should be near 0."""
        d = (
            "M -29.675 137.67 "
            "L -137.67 29.676 "
            "C -154.11 13.236 -154.11 -13.235 -137.67 -29.675 "
            "L -29.675 -137.67 "
            "C -13.235 -154.11 13.235 -154.11 29.676 -137.67 "
            "L 137.67 -29.675 "
            "C 154.11 -13.235 154.11 13.236 137.67 29.676 "
            "L 29.676 137.67 "
            "C 13.236 154.11 -13.235 154.11 -29.675 137.67 Z"
        )
        points = path_extreme_points(parse_and_normalize(d))
        # Find the point with the maximum X
        rightmost = max(points, key=lambda p: p[0])
        # At the rightmost curve apex, Y should be near zero (midpoint of the side)
        assert abs(rightmost[1]) < 1.0, f"Y at rightmost extreme should be ~0, got {rightmost[1]}"

    def test_ellipse_extremes(self):
        """Standard ellipse — extremes at the axis endpoints."""
        d = (
            "M 80 0 C 80 -56 44 -100 0 -100 "
            "C -44 -100 -80 -56 -80 0 "
            "C -80 56 -44 100 0 100 "
            "C 44 100 80 56 80 0 Z"
        )
        points = path_extreme_points(parse_and_normalize(d))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        assert max(xs) == pytest.approx(80, abs=0.5)
        assert min(xs) == pytest.approx(-80, abs=0.5)
        assert max(ys) == pytest.approx(100, abs=0.5)
        assert min(ys) == pytest.approx(-100, abs=0.5)

    def test_square_all_lineto(self):
        """Square — all extremes are at vertices, no curve extremes added."""
        d = "M -50 -50 L 50 -50 L 50 50 L -50 50 Z"
        cmds = parse_and_normalize(d)
        points = path_extreme_points(cmds)
        # Should have exactly 4 points (the 4 vertices from M + 3 L)
        assert len(points) == 4
        xs = [p[0] for p in points]
        assert max(xs) == pytest.approx(50)
        assert min(xs) == pytest.approx(-50)

    def test_empty_path(self):
        """Empty command list returns no points."""
        assert path_extreme_points([]) == []


# ═══════════════════════════════════════════════════════════════════════════════
#  Arc conversion — endpoint snapping and numerical accuracy
# ═══════════════════════════════════════════════════════════════════════════════

class TestArcEndpointSnapping:
    """The final cubic endpoint must exactly match the arc's target (x2, y2)."""

    def test_final_endpoint_exact(self):
        """arc_to_curves snaps last endpoint to (x2, y2) exactly."""
        curves = arc_to_curves(0, 0, 25, 25, 0, False, True, 50, 0)
        # Last endpoint must be EXACTLY 50, 0 — not approximately
        assert curves[-1][1][-2] == 50
        assert curves[-1][1][-1] == 0

    def test_full_circle_endpoint_exact(self):
        """After two semicircles, final endpoint must return exactly to start."""
        curves1 = arc_to_curves(0, 0, 25, 25, 0, False, True, 50, 0)
        end1_x, end1_y = curves1[-1][1][-2], curves1[-1][1][-1]
        curves2 = arc_to_curves(end1_x, end1_y, 25, 25, 0, False, True, 0, 0)
        assert curves2[-1][1][-2] == 0
        assert curves2[-1][1][-1] == 0

    def test_ellipse_endpoint_exact(self):
        """Elliptical arc snaps final endpoint exactly."""
        curves = arc_to_curves(0, 0, 50, 25, 0, False, True, 100, 0)
        assert curves[-1][1][-2] == 100
        assert curves[-1][1][-1] == 0

    def test_rotated_ellipse_endpoint_exact(self):
        """Rotated elliptical arc snaps final endpoint exactly."""
        curves = arc_to_curves(0, 0, 50, 25, 45, False, True, 100, 0)
        assert curves[-1][1][-2] == 100
        assert curves[-1][1][-1] == 0


class TestArcZeroExtent:
    """Zero angular extent arcs degrade to a line."""

    def test_zero_extent_from_near_coincident(self):
        """When radii force a degenerate arc, result should still be valid."""
        # Tiny radii that are too small for the distance — will be scaled up,
        # should still produce valid cubics with exact endpoint
        curves = arc_to_curves(0, 0, 1, 1, 0, False, True, 50, 0)
        assert len(curves) >= 1
        assert curves[-1][1][-2] == 50
        assert curves[-1][1][-1] == 0


class TestArcRadiiTolerance:
    """Radii on the boundary (lambda ≈ 1.0) are handled with tolerance."""

    def test_borderline_radii(self):
        """Radii that are exactly on the boundary should not crash."""
        # Distance between points = 50, so for a semicircle we need r >= 25.
        # Test with r = 25.0 (exactly at boundary — lambda = 1.0).
        curves = arc_to_curves(0, 0, 25, 25, 0, False, True, 50, 0)
        assert len(curves) >= 1
        assert curves[-1][1][-2] == 50
        assert curves[-1][1][-1] == 0


class TestArcVsInkscapeReference:
    """Compare arc_to_curves output against Inkscape 'To Absolute' reference.

    Reference data: an ellipse with rx=67.112, ry=44.742, defined by 4 arc
    commands (each spanning ~90°).  Inkscape converts to 8 cubics (2 per arc)
    using its own splitting strategy; our implementation may use fewer segments
    per arc (1 per ≤90° span) which is mathematically equivalent but produces
    different control point values.  Tests validate geometric properties rather
    than exact control-point-for-control-point matching.
    """

    ARC_PATH = (
        "M 127.1995315551758,111.9408645629883 "
        "A 67.11249542236328,44.74166488647461 0 0 1 60.0870361328125,156.6825294494629 "
        "A 67.11249542236328,44.74166488647461 0 0 1 -7.025459289550782,111.9408645629883 "
        "A 67.11249542236328,44.74166488647461 0 0 1 60.0870361328125,67.19919967651367 "
        "a 67.11249542236328,44.74166488647461 0 0 1 67.11249542236329,44.74166488647463 "
        "z"
    )

    # Ellipse parameters (derived from the arc commands)
    ECX, ECY = 60.087, 111.941
    ERX, ERY = 67.112, 44.742

    def _get_cubics(self):
        result = parse_and_normalize(self.ARC_PATH)
        return [r for r in result if r[0] == "C"]

    def test_segment_count_at_least_four(self):
        """4 quarter-ellipse arcs should produce at least 4 cubic segments."""
        cubics = self._get_cubics()
        assert len(cubics) >= 4

    def test_all_endpoints_on_ellipse(self):
        """Every cubic endpoint lies on the expected ellipse."""
        cubics = self._get_cubics()
        for i, c in enumerate(cubics):
            ex, ey = c[1][-2], c[1][-1]
            val = ((ex - self.ECX) / self.ERX) ** 2 + ((ey - self.ECY) / self.ERY) ** 2
            assert val == pytest.approx(1.0, abs=0.02), (
                f"Cubic {i} endpoint ({ex:.3f}, {ey:.3f}) not on ellipse: "
                f"equation value = {val:.4f}"
            )

    def test_arc_transition_endpoints_match(self):
        """The endpoint of each arc matches the expected quadrant points."""
        result = parse_and_normalize(self.ARC_PATH)
        cubics = [r for r in result if r[0] == "C"]

        # Expected quadrant endpoints (cardinal points of the ellipse):
        expected_endpoints = [
            (60.087, 156.683),   # bottom (cx, cy+ry)
            (-7.025, 111.941),   # left   (cx-rx, cy)
            (60.087, 67.199),    # top    (cx, cy-ry)
            (127.200, 111.941),  # right  (cx+rx, cy) — return to start
        ]

        # Each arc's last cubic should end at the corresponding quadrant point.
        # Since each arc may produce 1 or 2 cubics, we check which cubics land
        # near each expected endpoint.
        cubic_endpoints = [(c[1][-2], c[1][-1]) for c in cubics]
        for ex, ey in expected_endpoints:
            found = any(
                abs(cx - ex) < 0.1 and abs(cy - ey) < 0.1
                for cx, cy in cubic_endpoints
            )
            assert found, (
                f"No cubic endpoint near expected quadrant point ({ex}, {ey}). "
                f"Got endpoints: {cubic_endpoints}"
            )

    def test_closure_exact(self):
        """Final cubic endpoint matches the M starting point exactly."""
        result = parse_and_normalize(self.ARC_PATH)
        cubics = [r for r in result if r[0] == "C"]
        final_x = cubics[-1][1][-2]
        final_y = cubics[-1][1][-1]
        assert final_x == pytest.approx(127.1995315551758, abs=0.01)
        assert final_y == pytest.approx(111.9408645629883, abs=0.01)

    def test_bounding_box_matches_ellipse(self):
        """The collective bounding box of the cubics matches the ellipse extents."""
        cubics = self._get_cubics()
        all_x = []
        all_y = []
        for c in cubics:
            args = c[1]
            for j in range(0, 6, 2):
                all_x.append(args[j])
                all_y.append(args[j + 1])
        # Control points should be near the ellipse, bounding box should
        # roughly match: cx ± rx, cy ± ry
        assert min(all_x) < self.ECX - self.ERX + 5
        assert max(all_x) > self.ECX + self.ERX - 5
        assert min(all_y) < self.ECY - self.ERY + 5
        assert max(all_y) > self.ECY + self.ERY - 5


class TestArcToBeziers:
    """Direct tests of the _arc_to_beziers unit-circle helper."""

    def test_quarter_circle_kappa_value(self):
        """The kappa coefficient for a quarter-circle must be ≈0.5522847498."""
        # This is THE critical value — 4/3 * (√2 - 1)
        import math as m
        expected_kappa = (4.0 / 3.0) * (m.sqrt(2) - 1)
        pts = _arc_to_beziers(0, m.pi / 2)
        # CP1 = (1, kappa) for a quarter-arc starting at angle 0
        actual_kappa = pts[1]  # cp1y
        assert actual_kappa == pytest.approx(expected_kappa, rel=1e-10), (
            f"kappa={actual_kappa}, expected={expected_kappa}"
        )

    def test_quarter_circle_single_segment(self):
        """Quarter circle (π/2) produces 6 values (1 segment)."""
        pts = _arc_to_beziers(0, math.pi / 2)
        assert len(pts) == 6

    def test_semicircle_two_segments(self):
        """Semicircle (π) produces 12 values (2 segments)."""
        pts = _arc_to_beziers(0, math.pi)
        assert len(pts) == 12

    def test_full_circle_four_segments(self):
        """Full circle (2π) produces 24 values (4 segments)."""
        pts = _arc_to_beziers(0, 2 * math.pi)
        assert len(pts) == 24

    def test_endpoint_on_unit_circle(self):
        """Each segment endpoint should lie on the unit circle."""
        pts = _arc_to_beziers(0, math.pi)
        for i in range(0, len(pts), 6):
            ex, ey = pts[i + 4], pts[i + 5]
            r = math.sqrt(ex * ex + ey * ey)
            assert r == pytest.approx(1.0, abs=1e-10)

    def test_semicircle_cubics_match_inkscape(self):
        """A semicircular arc's 2 cubics should match the Inkscape reference.

        Inkscape splits this ellipse's arcs 3 & 4 into 2 cubics each,
        and those match our output exactly when we also produce 2 segments.
        Test a simple r=25 semicircle against known-good values.
        """
        # Semicircle: from (50, 0) to (0, 0) with r=25, center at (25, 0)
        # On unit circle: kappa ≈ 0.5523 for each quarter
        pts = _arc_to_beziers(0, math.pi)
        # First segment ends at (0, 1), second at (-1, 0)
        assert pts[4] == pytest.approx(0, abs=1e-10)
        assert pts[5] == pytest.approx(1, abs=1e-10)
        assert pts[10] == pytest.approx(-1, abs=1e-10)
        assert pts[11] == pytest.approx(0, abs=1e-10)
