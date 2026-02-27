#!/usr/bin/env python3
"""
Music Glyph Toolkit — Inkscape extension for music notation SVG normalization.

Pipeline architecture:
    parse_and_normalize(d_str)  →  List[Tuple[str, List[float]]]
                                          ↓
                             ┌────────────┴────────────┐
                             ↓                         ↓
                format_svg_d(commands)        format_scheme(commands)
                  → "M 10 20 C ..."             → "(moveto 0.1 -0.2)\\n..."

All SVG path commands (S, Q, T, A, H, V, relative) are normalized to
absolute M, C, L, Z only. The two formatters produce either an SVG `d`
attribute string or LilyPond Scheme path syntax from the same intermediate
representation.
"""
import inkex
import re
import math
import os
from inkex import PathElement


# IDs that the toolkit manages — used to skip during MCLZ normalization
STEM_IDS = {'stem-up', 'stem-down'}
GUIDELINES_LAYER_ID = 'guidelines-layer'
FULL_VIEW_ID = 'full-view'

# IDs managed by the toolkit that should be excluded from path processing
MANAGED_IDS = STEM_IDS | {GUIDELINES_LAYER_ID, FULL_VIEW_ID}

# Forbidden SVG path commands (anything not strict absolute M, C, L, Z)
FORBIDDEN_COMMANDS = re.compile(r'[AaSsQqTtHhVvmlcz]')

# Standard stem height in semitone-grid units
STEM_HEIGHT = 350.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Color helpers: extract existing colors and choose contrasting guideline pair
# ═══════════════════════════════════════════════════════════════════════════════

def parse_hex_color(style_str):
    """Extract hex color(s) from an SVG style string.
    
    Looks for fill: and stroke: properties and returns a list of
    (R, G, B) tuples (0–255) for any hex colors found.
    """
    colors = []
    if not style_str:
        return colors
    for match in re.finditer(r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b', style_str):
        h = match.group(1)
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
    return colors


def color_distance(c1, c2):
    """Perceptual color distance (weighted Euclidean in RGB).
    
    Uses human-eye weighting: green sensitivity > red > blue.
    """
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return math.sqrt(2 * dr*dr + 4 * dg*dg + 3 * db*db)


def rgb_to_hex(r, g, b):
    """Convert RGB tuple to hex string."""
    return f"#{r:02X}{g:02X}{b:02X}"


# Candidate guideline colors — bright, saturated, spread across the hue wheel
GUIDELINE_CANDIDATES = [
    (255,   0,   0),  # red
    (  0, 255,   0),  # green
    (  0,   0, 255),  # blue
    (255, 255,   0),  # yellow
    (  0, 255, 255),  # cyan
    (255,   0, 255),  # magenta
    (255, 128,   0),  # orange
    (128,   0, 255),  # purple
    (  0, 255, 128),  # spring green
    (255,   0, 128),  # rose
    (  0, 128, 255),  # sky blue
    (128, 255,   0),  # chartreuse
]


def pick_guideline_colors(existing_colors):
    """Choose two colors that contrast well against all existing colors.
    
    Selects the pair from GUIDELINE_CANDIDATES that maximizes the minimum
    perceptual distance to any existing color, while also being distinct
    from each other.
    
    Args:
        existing_colors: list of (R, G, B) tuples from the glyph elements.
        
    Returns:
        (hex1, hex2): Two hex color strings for the guideline line pair.
    """
    # Always include black and white as "existing" (canvas background + typical fill)
    all_existing = list(existing_colors) + [(0, 0, 0), (255, 255, 255)]
    
    candidates = GUIDELINE_CANDIDATES
    
    # Score each candidate by its minimum distance to any existing color
    def min_dist(c):
        return min(color_distance(c, e) for e in all_existing)
    
    # Sort candidates by contrast (best first)
    ranked = sorted(candidates, key=min_dist, reverse=True)
    
    # Pick the best candidate
    color_a = ranked[0]
    
    # Pick the second-best that is also distinct from color_a
    color_b = None
    for c in ranked[1:]:
        if color_distance(c, color_a) > 200:  # ensure they're visually distinct
            color_b = c
            break
    if color_b is None:
        color_b = ranked[1]
    
    return rgb_to_hex(*color_a), rgb_to_hex(*color_b)


# ═══════════════════════════════════════════════════════════════════════════════
#  Math helpers: arc and quadratic → cubic Bézier conversion
# ═══════════════════════════════════════════════════════════════════════════════
def _arc_to_beziers(angle_start, angle_extent):
    """Generate cubic Bézier control points for a unit-circle arc.

    Uses the standard Maisonobe/kappa tangent-based approximation for arcs
    up to π/2.  Returns a flat list of coordinates:
    [cp1x, cp1y, cp2x, cp2y, epx, epy, ...] with 6 values per segment.

    The control-point distance from each endpoint is::

        kappa = (4/3) * tan(segment_angle / 4)

    which is equivalent to ``(4/3) * sin(a/2) / (1 + cos(a/2))`` — the
    standard formula for cubic Bézier approximation of circular arcs.
    """
    n_segs = max(1, int(math.ceil(abs(angle_extent) / (math.pi / 2))))
    d_angle = angle_extent / n_segs
    points = []
    angle = angle_start

    # Standard kappa coefficient (Maisonobe formula).
    # For a quarter-circle (π/2) this gives the well-known
    # value ≈ 0.5522847498.
    kappa = (4.0 / 3.0) * math.tan(d_angle / 4.0)

    for _ in range(n_segs):
        cos1 = math.cos(angle)
        sin1 = math.sin(angle)
        cos2 = math.cos(angle + d_angle)
        sin2 = math.sin(angle + d_angle)

        # CP1 = P1 + kappa * tangent_at_P1;  tangent at (cos,sin) = (-sin,cos)
        cp1x = cos1 - kappa * sin1
        cp1y = sin1 + kappa * cos1
        # CP2 = P2 - kappa * tangent_at_P2
        cp2x = cos2 + kappa * sin2
        cp2y = sin2 - kappa * cos2
        # End point
        epx = cos2
        epy = sin2

        points.extend([cp1x, cp1y, cp2x, cp2y, epx, epy])
        angle += d_angle

    return points


def arc_to_curves(x1, y1, rx, ry, phi_deg, large_arc, sweep, x2, y2):
    """Convert an SVG elliptical arc to one or more cubic Bézier curves.

    Implements the SVG arc parameterization algorithm (F.6.5/F.6.6) to find
    the ellipse center, generates Bézier control points on a **unit circle**,
    then applies the full affine transform (scale → rotate → translate) to
    map them onto the actual ellipse.  This two-phase approach (matching the
    Android/Java SVG reference implementation) is more numerically stable
    than scaling during generation.

    The final endpoint of the last segment is snapped to the exact target
    coordinates ``(x2, y2)`` to eliminate accumulated floating-point drift.

    Edge cases:
        - Coincident endpoints → empty list
        - Zero radius → [('L', [x2, y2])]
        - Negative radii → treated as positive (per SVG spec)
        - Zero angular extent → [('L', [x2, y2])]
    """
    # --- Edge cases (per SVG spec) ---
    if x1 == x2 and y1 == y2:
        return []
    if rx == 0 or ry == 0:
        return [('L', [x2, y2])]
    rx, ry = abs(rx), abs(ry)

    # --- Step 1: Compute (x1', y1') ---
    phi = math.radians(phi_deg % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # --- Ensure radii are large enough (tolerance-based, per Java ref) ---
    rx_sq = rx * rx
    ry_sq = ry * ry
    x1p_sq = x1p * x1p
    y1p_sq = y1p * y1p
    radii_check = x1p_sq / rx_sq + y1p_sq / ry_sq
    if radii_check > 0.99999:
        radii_scale = math.sqrt(radii_check) * 1.00001
        rx *= radii_scale
        ry *= radii_scale
        rx_sq = rx * rx
        ry_sq = ry * ry

    # --- Step 2: Compute transformed centre (cxp, cyp) ---
    sign = -1.0 if (large_arc == sweep) else 1.0
    sq = ((rx_sq * ry_sq) - (rx_sq * y1p_sq) - (ry_sq * x1p_sq)) / \
         ((rx_sq * y1p_sq) + (ry_sq * x1p_sq))
    sq = max(0.0, sq)
    coef = sign * math.sqrt(sq)
    cxp = coef * (rx * y1p / ry)
    cyp = coef * -(ry * x1p / rx)

    # --- Step 3: Compute actual centre (cx, cy) ---
    sx2 = (x1 + x2) / 2.0
    sy2 = (y1 + y2) / 2.0
    cx = sx2 + (cos_phi * cxp - sin_phi * cyp)
    cy = sy2 + (sin_phi * cxp + cos_phi * cyp)

    # --- Step 4: Compute start angle and angular extent ---
    def _vec_angle(ux, uy, vx, vy):
        n = math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
        if n == 0:
            return 0.0
        c = max(-1.0, min(1.0, (ux * vx + uy * vy) / n))
        a = math.acos(c)
        return -a if (ux * vy - uy * vx < 0) else a

    theta1 = _vec_angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = _vec_angle(
        (x1p - cxp) / rx, (y1p - cyp) / ry,
        (-x1p - cxp) / rx, (-y1p - cyp) / ry,
    )

    # Zero extent — degenerate arc, emit a line (per Java ref)
    if dtheta == 0:
        return [('L', [x2, y2])]

    if not sweep and dtheta > 0:
        dtheta -= 2.0 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2.0 * math.pi

    # --- Generate unit-circle Béziers, then transform ---
    bezier_pts = _arc_to_beziers(theta1, dtheta)

    # Apply affine transform: scale(rx, ry) → rotate(phi) → translate(cx, cy)
    for i in range(0, len(bezier_pts), 2):
        px = bezier_pts[i] * rx
        py = bezier_pts[i + 1] * ry
        bezier_pts[i] = cos_phi * px - sin_phi * py + cx
        bezier_pts[i + 1] = sin_phi * px + cos_phi * py + cy

    # Snap final endpoint to exact target (eliminates accumulated FP drift)
    if len(bezier_pts) >= 2:
        bezier_pts[-2] = x2
        bezier_pts[-1] = y2

    # --- Build curve tuples ---
    curves = []
    for i in range(0, len(bezier_pts), 6):
        curves.append(('C', [
            bezier_pts[i],     bezier_pts[i + 1],
            bezier_pts[i + 2], bezier_pts[i + 3],
            bezier_pts[i + 4], bezier_pts[i + 5],
        ]))

    return curves

def quadratic_to_cubic(x0, y0, x1, y1, x2, y2):
    """Elevate a quadratic Bézier to a cubic using the 2/3 rule.

    Returns (cx1, cy1, cx2, cy2, x2, y2) — the cubic control points and endpoint.
    """
    return (x0 + 2/3 * (x1 - x0), y0 + 2/3 * (y1 - y0),
            x2 + 2/3 * (x1 - x2), y2 + 2/3 * (y1 - y2), x2, y2)


def _cubic_axis_extremes(p0, p1, p2, p3):
    """Find parameter values where a cubic Bézier has an extremum on one axis.

    Given four scalar control values (e.g. all X or all Y coordinates),
    solves dx/dt = 0 for the cubic Bézier and returns the (t, value) pairs
    for any roots in the open interval (0, 1).

    The derivative of B(t) = (1-t)³p0 + 3(1-t)²t·p1 + 3(1-t)t²·p2 + t³·p3
    is: B'(t) = 3[(1-t)²(p1-p0) + 2(1-t)t(p2-p1) + t²(p3-p2)]
    which is a quadratic: At² + Bt + C = 0 where
        a = p1-p0,  b = p2-p1,  c = p3-p2
        A = a - 2b + c,  B = 2(b - a),  C = a
    """
    a = p1 - p0
    b = p2 - p1
    c = p3 - p2
    A = a - 2*b + c
    B = 2*(b - a)
    C = a

    roots = []
    if abs(A) < 1e-12:
        # Linear: Bt + C = 0
        if abs(B) > 1e-12:
            t = -C / B
            if 0 < t < 1:
                roots.append(t)
    else:
        disc = B*B - 4*A*C
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            for t in [(-B + sqrt_disc) / (2*A), (-B - sqrt_disc) / (2*A)]:
                if 0 < t < 1:
                    roots.append(t)

    # Evaluate the cubic at each root
    results = []
    for t in roots:
        s = 1 - t
        val = s*s*s*p0 + 3*s*s*t*p1 + 3*s*t*t*p2 + t*t*t*p3
        results.append((t, val))
    return results


def path_extreme_points(commands):
    """Find the true geometric extreme points of a normalized MCLZ path.

    Examines every segment (lines and cubic Béziers) and returns
    (x, y) for every point that could be an X or Y extreme — including
    points along curves where the derivative is zero.

    Args:
        commands: List of (cmd, args) tuples from parse_and_normalize().

    Returns:
        List of (x, y) tuples: all endpoints plus all curve extremes.
    """
    points = []
    cx, cy = 0.0, 0.0  # current point

    for cmd, args in commands:
        if cmd == 'M':
            cx, cy = args[0], args[1]
            points.append((cx, cy))

        elif cmd == 'L':
            cx, cy = args[0], args[1]
            points.append((cx, cy))

        elif cmd == 'C':
            x1, y1, x2, y2, x3, y3 = args
            # Check for X extremes along the curve
            for _t, xval in _cubic_axis_extremes(cx, x1, x2, x3):
                s = 1 - _t
                yval = s*s*s*cy + 3*s*s*_t*y1 + 3*s*_t*_t*y2 + _t*_t*_t*y3
                points.append((xval, yval))
            # Check for Y extremes along the curve
            for _t, yval in _cubic_axis_extremes(cy, y1, y2, y3):
                s = 1 - _t
                xval = s*s*s*cx + 3*s*s*_t*x1 + 3*s*_t*_t*x2 + _t*_t*_t*x3
                points.append((xval, yval))
            # Always include the endpoint
            cx, cy = x3, y3
            points.append((cx, cy))

        elif cmd == 'Z':
            pass  # Z returns to start, no new extreme

    return points

# ═══════════════════════════════════════════════════════════════════════════════
#  Core pipeline: parse → normalize → format
# ═══════════════════════════════════════════════════════════════════════════════

def parse_and_normalize(d_str):
    """Parse SVG path `d` string and normalize to absolute M/C/L/Z commands.
    
    Returns:
        List[Tuple[str, List[float]]]: e.g. [('M', [x, y]), ('C', [x1,y1,x2,y2,x,y]), ('Z', [])]
    """
    token_pattern = re.compile(r'([MmZzLlHhVvCcSsQqTtAa])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')
    tokens = []
    for match in token_pattern.finditer(d_str):
        cmd, num = match.groups()
        if cmd: tokens.append(('cmd', cmd))
        elif num: tokens.append(('num', float(num)))
        
    commands = []
    current_x, current_y, start_x, start_y = 0.0, 0.0, 0.0, 0.0
    last_control_x, last_control_y = 0.0, 0.0
    last_command = None
    
    i = 0
    def get_nums(count):
        nonlocal i
        nums = []
        for _ in range(count):
            if i < len(tokens) and tokens[i][0] == 'num':
                nums.append(tokens[i][1])
                i += 1
            else: nums.append(0.0)
        return nums

    while i < len(tokens):
        if tokens[i][0] == 'cmd':
            cmd = tokens[i][1]
            i += 1
        else:
            cmd = 'L' if last_command in ('M', 'm') else last_command
            
        if cmd in ('M', 'm'):
            nums = get_nums(2)
            current_x = current_x + nums[0] if cmd == 'm' else nums[0]
            current_y = current_y + nums[1] if cmd == 'm' else nums[1]
            start_x, start_y = current_x, current_y
            commands.append(('M', [current_x, current_y]))
            last_command = cmd
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(2)
                current_x = current_x + nums[0] if cmd == 'm' else nums[0]
                current_y = current_y + nums[1] if cmd == 'm' else nums[1]
                commands.append(('L', [current_x, current_y]))
                
        elif cmd in ('L', 'l'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(2)
                current_x = current_x + nums[0] if cmd == 'l' else nums[0]
                current_y = current_y + nums[1] if cmd == 'l' else nums[1]
                commands.append(('L', [current_x, current_y]))
            last_command = cmd
            
        elif cmd in ('H', 'h'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(1)
                current_x = current_x + nums[0] if cmd == 'h' else nums[0]
                commands.append(('L', [current_x, current_y]))
            last_command = cmd
            
        elif cmd in ('V', 'v'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(1)
                current_y = current_y + nums[0] if cmd == 'v' else nums[0]
                commands.append(('L', [current_x, current_y]))
            last_command = cmd
            
        elif cmd in ('C', 'c'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(6)
                if cmd == 'c':
                    x1, y1 = current_x + nums[0], current_y + nums[1]
                    x2, y2 = current_x + nums[2], current_y + nums[3]
                    x, y = current_x + nums[4], current_y + nums[5]
                else:
                    x1, y1, x2, y2, x, y = nums
                commands.append(('C', [x1, y1, x2, y2, x, y]))
                last_control_x, last_control_y = x2, y2
                current_x, current_y = x, y
            last_command = cmd
            
        elif cmd in ('S', 's'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(4)
                if last_command in ('C', 'c', 'S', 's'):
                    x1, y1 = 2 * current_x - last_control_x, 2 * current_y - last_control_y
                else:
                    x1, y1 = current_x, current_y
                if cmd == 's':
                    x2, y2 = current_x + nums[0], current_y + nums[1]
                    x, y = current_x + nums[2], current_y + nums[3]
                else:
                    x2, y2, x, y = nums
                commands.append(('C', [x1, y1, x2, y2, x, y]))
                last_control_x, last_control_y = x2, y2
                current_x, current_y = x, y
                last_command = cmd
            
        elif cmd in ('Q', 'q'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(4)
                if cmd == 'q':
                    x1, y1 = current_x + nums[0], current_y + nums[1]
                    x, y = current_x + nums[2], current_y + nums[3]
                else:
                    x1, y1, x, y = nums
                cubic = quadratic_to_cubic(current_x, current_y, x1, y1, x, y)
                commands.append(('C', list(cubic)))
                last_control_x, last_control_y = x1, y1
                current_x, current_y = x, y
            last_command = cmd
            
        elif cmd in ('T', 't'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(2)
                if last_command in ('Q', 'q', 'T', 't'):
                    x1, y1 = 2 * current_x - last_control_x, 2 * current_y - last_control_y
                else:
                    x1, y1 = current_x, current_y
                if cmd == 't':
                    x, y = current_x + nums[0], current_y + nums[1]
                else:
                    x, y = nums
                cubic = quadratic_to_cubic(current_x, current_y, x1, y1, x, y)
                commands.append(('C', list(cubic)))
                last_control_x, last_control_y = x1, y1
                current_x, current_y = x, y
                last_command = cmd
            
        elif cmd in ('A', 'a'):
            while i < len(tokens) and tokens[i][0] == 'num':
                nums = get_nums(7)
                rx, ry, phi, large_arc, sweep = nums[0], nums[1], nums[2], nums[3] != 0, nums[4] != 0
                if cmd == 'a':
                    x, y = current_x + nums[5], current_y + nums[6]
                else:
                    x, y = nums[5], nums[6]
                arc_curves = arc_to_curves(current_x, current_y, rx, ry, phi, large_arc, sweep, x, y)
                commands.extend(arc_curves)
                current_x, current_y = x, y
            last_command = cmd
            
        elif cmd in ('Z', 'z'):
            commands.append(('Z', []))
            current_x, current_y = start_x, start_y
            last_command = cmd
        else:
            i += 1

    return commands


def format_svg_d(commands, decimals=3):
    """Format normalized commands as an SVG `d` attribute string.
    
    Args:
        commands: Output of parse_and_normalize().
        decimals: Decimal places for rounding coordinates.
    """
    parts = []
    for cmd, args in commands:
        rounded = [str(round(a, decimals)) for a in args]
        parts.append(f"{cmd} {' '.join(rounded)}" if rounded else cmd)
    return " ".join(parts)


def format_scheme(commands, scale=0.01, decimals=4):
    """Format normalized commands as LilyPond Scheme path syntax.
    
    Applies coordinate scaling (default /100 for staff-space units)
    and negates Y axis (SVG Y-down → LilyPond Y-up).
    
    Args:
        commands: Output of parse_and_normalize().
        scale: Coordinate scale factor (default 0.01).
        decimals: Decimal places for rounding coordinates.
    """
    lines = []
    for cmd, args in commands:
        if cmd == 'M':
            x, y = round(args[0] * scale, decimals), round(args[1] * scale, decimals)
            lines.append(f"    (moveto {x} {-y})")
        elif cmd == 'L':
            x, y = round(args[0] * scale, decimals), round(args[1] * scale, decimals)
            lines.append(f"    (lineto {x} {-y})")
        elif cmd == 'C':
            x1 = round(args[0] * scale, decimals)
            y1 = round(args[1] * scale, decimals)
            x2 = round(args[2] * scale, decimals)
            y2 = round(args[3] * scale, decimals)
            x  = round(args[4] * scale, decimals)
            y  = round(args[5] * scale, decimals)
            lines.append(f"    (curveto {x1} {-y1} {x2} {-y2} {x} {-y})")
        elif cmd == 'Z':
            lines.append("    (closepath)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def validate_for_export(svg):
    """Validate that the SVG canvas is ready for LilyPond export.
    
    Checks:
        1. All path elements (except stems) contain only M/C/L/Z commands.
        2. The notehead (if present) bounding box is centered at 0,0.
    
    Raises inkex.AbortExtension with a descriptive message on failure.
    """
    errors = []
    
    for elem in svg.descendants().filter(PathElement):
        eid = elem.get('id', '')
        if eid in STEM_IDS:
            continue
        d_str = elem.get('d', '')
        forbidden = FORBIDDEN_COMMANDS.findall(d_str)
        if forbidden:
            unique = sorted(set(forbidden))
            errors.append(
                f"Path '{eid}' contains forbidden commands: {', '.join(unique)}.\n"
                "Run the Notehead or Glyph formatter first to normalize paths to strict M/C/L/Z."
            )
    
    # Check notehead centering
    noteheads = [e for e in svg.descendants().filter(PathElement)
                 if 'notehead' in e.get('id', '').lower()]
    if len(noteheads) == 1:
        bbox = noteheads[0].bounding_box()
        if bbox is not None:
            cx, cy = bbox.center.x, bbox.center.y
            if abs(cx) > 1.0 or abs(cy) > 1.0:
                errors.append(
                    f"Notehead bounding box center is at ({cx:.1f}, {cy:.1f}), not at (0, 0).\n"
                    "Run the Notehead formatter first to center the glyph."
                )
    
    if errors:
        raise inkex.AbortExtension(
            "Export validation failed:\n\n" + "\n\n".join(errors)
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Inkscape extension class
# ═══════════════════════════════════════════════════════════════════════════════

def parse_semitones(value, label, min_val, max_val):
    """Parse a semitone height string to float with range validation.

    Args:
        value: String from the INX text field.
        label: Human-readable field name for error messages.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        float: The parsed value.

    Raises:
        inkex.AbortExtension: If the value is not a valid number or is out of range.
    """
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise inkex.AbortExtension(
            f"{label}: '{value}' is not a valid number. Please enter a numeric value (e.g. 2.0, 2.72, 1.5)."
        )
    if result < min_val or result > max_val:
        raise inkex.AbortExtension(
            f"{label}: {result} is out of range. Please enter a value between {min_val} and {max_val}."
        )
    return result


class MusicGlyphToolkit(inkex.EffectExtension):
    def add_arguments(self, pars):
        pars.add_argument("--active_tab", type=str, default="tab_notehead")
        pars.add_argument("--nh_semitones", type=str, default="2.0")
        pars.add_argument("--stem_placemarkers", type=inkex.Boolean, default=True)
        pars.add_argument("--color_stem_up", type=str, default="#FF00FF")
        pars.add_argument("--color_stem_down", type=str, default="#00FFFF")
        pars.add_argument("--glyph_semitones", type=str, default="4.0")
        pars.add_argument("--output_format", type=str, default="svg")
        pars.add_argument("--strict_mclz", type=inkex.Boolean, default=True)
        pars.add_argument("--round_decimals", type=int, default=3)
        pars.add_argument("--show_guidelines", type=inkex.Boolean, default=True)
        pars.add_argument("--include_lilypond_data", type=inkex.Boolean, default=True)

    def effect(self):
        output_format = self.options.output_format

        # ── Export mode: validate and export without processing ──
        if output_format == "lilypond":
            validate_for_export(self.svg)
            self.export_lilypond()
            return

        # ── SVG processing mode ──
        tab = self.options.active_tab

        if tab == "tab_notehead":
            self.process_notehead()
        elif tab == "tab_glyph":
            self.process_glyph()

        # Apply MCLZ strict formatting and rounding to path elements only
        # (skips stem rectangles automatically since they aren't PathElements)
        if self.options.strict_mclz:
            for elem in self.svg.descendants().filter(PathElement):
                eid = elem.get('id', '')
                if eid not in STEM_IDS:
                    commands = parse_and_normalize(elem.get('d'))
                    baked_d = format_svg_d(commands, self.options.round_decimals)
                    elem.set('d', baked_d)

        # Embed pre-baked LilyPond path data after normalization
        if self.options.include_lilypond_data:
            self._embed_lilypond_data(tab)
        else:
            self._strip_lilypond_data()

    SVG_NS = '{http://www.w3.org/2000/svg}'
    INKSCAPE_NS = '{http://www.inkscape.org/namespaces/inkscape}'

    def _collect_glyph_colors(self):
        """Extract all fill/stroke colors from notehead, hollow, and stem elements."""
        colors = []
        for elem in self.svg.descendants():
            eid = elem.get('id', '')
            if eid in ('notehead', 'hollow', 'stem-up', 'stem-down'):
                style = elem.get('style', '')
                colors.extend(parse_hex_color(style))
                # Also check direct fill/stroke attributes
                for attr in ('fill', 'stroke'):
                    val = elem.get(attr, '')
                    colors.extend(parse_hex_color(val))
        return colors

    @staticmethod
    def _semitone_label_and_id(offset):
        """Generate the label and element ID for a semitone at the given offset.
        
        Convention:
            offset  0 → "semitone i"
            offset +3 → "semitone i + 3"
            offset -2 → "semitone i - 2"
        """
        if offset == 0:
            return "semitone i", "guideline-semi-i"
        elif offset > 0:
            return f"semitone i + {offset}", f"guideline-semi-iplus{offset}"
        else:
            return f"semitone i - {abs(offset)}", f"guideline-semi-iminus{abs(offset)}"

    def generate_guidelines(self, tnw, tnh, y_min, y_max, show=True):
        """Create semitone reference lines in a dedicated Inkscape layer.
        
        Generates enough guideline pairs to cover the full vertical extent
        of all visible elements (including stems). Each semitone is marked
        by a pair of abutting dashed lines whose colors are dynamically
        chosen to contrast against the notehead, hollow, and stem colors.
        
        Args:
            tnw: Target width in SVG units (viewport width).
            tnh: Target height in SVG units (viewport height).
            y_min: Topmost SVG Y coordinate of all visible elements (most negative).
            y_max: Bottommost SVG Y coordinate of all visible elements (most positive).
            show: If True the guidelines layer is visible; if False, hidden.
        """
        svg = self.svg
        SUPS = 100.0  # SVG units per semitone
        
        # Compute the range of semitone offsets needed to cover y_min to y_max.
        # SVG Y-down: semitone at offset N is at SVG y = -(N * SUPS).
        # So offset = -y / SUPS. We need offsets from floor(-y_max/SUPS) to ceil(-y_min/SUPS).
        offset_min = math.floor(-y_max / SUPS)
        offset_max = math.ceil(-y_min / SUPS)
        
        # Remove existing guidelines layer if present
        for elem in svg.descendants():
            if elem.get('id') == GUIDELINES_LAYER_ID:
                elem.getparent().remove(elem)
                break
        
        # Pick guideline colors that contrast against existing glyph colors
        existing_colors = self._collect_glyph_colors()
        color_a, color_b = pick_guideline_colors(existing_colors)
        
        # Create layer group
        layer_attribs = {
            'id': GUIDELINES_LAYER_ID,
            f'{self.INKSCAPE_NS}groupmode': 'layer',
            f'{self.INKSCAPE_NS}label': 'Semitone Guidelines',
            'style': f'display:{"inline" if show else "none"}',
        }
        guidelines_layer = inkex.etree.SubElement(svg, f'{self.SVG_NS}g', attrib=layer_attribs)
        
        # Shared line properties
        stroke_w = 1.0
        half_stroke = stroke_w / 2.0
        dash = '6,4'
        half_w = tnw / 2.0
        
        for offset in range(offset_min, offset_max + 1):
            label, guide_id = self._semitone_label_and_id(offset)
            
            # SVG Y-down: higher pitch = more negative Y
            y = -(offset * SUPS)
            
            # Group for this semitone pair
            pair_group = inkex.etree.SubElement(guidelines_layer, f'{self.SVG_NS}g', attrib={
                'id': guide_id,
                f'{self.INKSCAPE_NS}label': label,
            })
            
            # Upper line (color A) just above the semitone Y (inner edge touches y)
            inkex.etree.SubElement(pair_group, f'{self.SVG_NS}line', attrib={
                'id': f'{guide_id}-upper',
                'x1': str(-half_w), 'y1': str(y - half_stroke),
                'x2': str(half_w),  'y2': str(y - half_stroke),
                'style': f'stroke:{color_a};stroke-width:{stroke_w};stroke-dasharray:{dash};stroke-opacity:0.8',
            })
            
            # Lower line (color B) just below the semitone Y (inner edge touches y)
            inkex.etree.SubElement(pair_group, f'{self.SVG_NS}line', attrib={
                'id': f'{guide_id}-lower',
                'x1': str(-half_w), 'y1': str(y + half_stroke),
                'x2': str(half_w),  'y2': str(y + half_stroke),
                'style': f'stroke:{color_b};stroke-width:{stroke_w};stroke-dasharray:{dash};stroke-opacity:0.8',
            })

    def generate_full_view(self, tnw, tnh, stem_coords):
        """Create or update a <view> element whose viewBox encompasses all objects
        including stems, so a consuming program can optionally show everything.
        
        Args:
            tnw: Target notehead/glyph width in SVG units.
            tnh: Target notehead/glyph height in SVG units.
            stem_coords: dict with keys 'x_up', 'y_up', 'x_down', 'y_down',
                         'w_stem' describing the stem rectangles, or None if
                         no stems (glyph mode).
        """
        svg = self.svg
        DEC_PLACES = self.options.round_decimals
        
        # Remove existing <view> if present
        for elem in svg:
            if elem.get('id') == FULL_VIEW_ID:
                svg.remove(elem)
                break
        
        if stem_coords is not None:
            # Compute extent that includes stems
            x_left = min(-tnw / 2.0, stem_coords['x_down'])
            x_right = max(tnw / 2.0, stem_coords['x_up'] + stem_coords['w_stem'])
            y_top = min(-tnh / 2.0, stem_coords['y_up'])
            y_bottom = max(tnh / 2.0, stem_coords['y_down'] + STEM_HEIGHT)
        else:
            # Glyph mode — no stems, just use the glyph dimensions
            x_left = -tnw / 2.0
            x_right = tnw / 2.0
            y_top = -tnh / 2.0
            y_bottom = tnh / 2.0
        
        full_w = x_right - x_left
        full_h = y_bottom - y_top
        
        vb = (f"{round(x_left, DEC_PLACES)} {round(y_top, DEC_PLACES)} "
              f"{round(full_w, DEC_PLACES)} {round(full_h, DEC_PLACES)}")
        
        inkex.etree.SubElement(svg, f'{self.SVG_NS}view', attrib={
            'id': FULL_VIEW_ID,
            'viewBox': vb,
        })

    def _bake_transforms(self):
        """Normalize path data and apply any pre-existing transforms.

        Must run before any bounding box calculations or processing.

        Normalizes arcs/quadratics to cubic Béziers BEFORE applying
        transforms, because inkex.apply_transform() cannot correctly
        apply rotation/skew transforms to elliptical arc parameters.
        Cubic Béziers transform correctly under any affine transformation.
        """
        for elem in self.svg.descendants().filter(PathElement):
            eid = elem.get('id', '')
            if eid in STEM_IDS:
                continue
            # Normalize arcs → cubics so the transform applies cleanly
            d_str = elem.get('d', '')
            if d_str:
                commands = parse_and_normalize(d_str)
                elem.set('d', format_svg_d(commands, decimals=6))
            elem.apply_transform()

    def _find_existing_stem(self, stem_id):
        """Find an existing stem element (rect or path) by ID."""
        for elem in self.svg.descendants():
            if elem.get('id') == stem_id:
                return elem
        return None

    def _embed_lilypond_data(self, tab):
        """Embed pre-baked LilyPond path data as data- attributes on SVG elements.

        Called after MCLZ normalization to ensure paths are fully normalized.
        The attribute value is the complete LilyPond path string in wrapped
        format: '((moveto x y) (curveto ...) ... (closepath))

        In notehead mode: sets data-lilypond-notehead on #notehead,
                          data-lilypond-hollow on #hollow (if present).
        In glyph mode:    sets data-lilypond-glyph on each non-stem path element.
        """
        if tab == "tab_notehead":
            for elem in self.svg.descendants().filter(PathElement):
                eid = elem.get('id', '')
                if eid == 'notehead':
                    commands = parse_and_normalize(elem.get('d', ''))
                    scheme = format_scheme(commands)
                    elem.set('data-lilypond-notehead', f"'(\n{scheme})")
                elif eid == 'hollow':
                    commands = parse_and_normalize(elem.get('d', ''))
                    scheme = format_scheme(commands)
                    elem.set('data-lilypond-hollow', f"'(\n{scheme})")

        elif tab == "tab_glyph":
            for elem in self.svg.descendants().filter(PathElement):
                eid = elem.get('id', '')
                if eid in STEM_IDS:
                    continue
                commands = parse_and_normalize(elem.get('d', ''))
                scheme = format_scheme(commands)
                elem.set('data-lilypond-glyph', f"'(\n{scheme})")

    def _strip_lilypond_data(self):
        """Remove any data-lilypond-* attributes from all elements.

        Called when the 'Include LilyPond path data' checkbox is unchecked,
        to clean up attributes from previous runs.
        """
        lilypond_attrs = ('data-lilypond-notehead', 'data-lilypond-hollow', 'data-lilypond-glyph')
        for elem in self.svg.descendants().filter(PathElement):
            for attr in lilypond_attrs:
                if elem.get(attr) is not None:
                    del elem.attrib[attr]

    def process_notehead(self):
        svg = self.svg
        
        # --- BAKE PRE-EXISTING TRANSFORMS ---
        # Normalize arcs to cubics, then apply any transforms (flips,
        # rotations, etc.) into path data before processing.
        self._bake_transforms()
        
        # --- SAFETY CHECK ---
        # Check for unconverted objects that should be paths
        # (exclude rect elements with stem IDs — those are our managed stems)
        for elem in svg.descendants():
            eid = elem.get('id', '')
            if eid and ('notehead' in eid.lower() or 'hollow' in eid.lower()):
                if not isinstance(elem, inkex.PathElement):
                    # Before raising, check if this non-path element (e.g. a <g>
                    # group) contains a descendant PathElement with the same
                    # keyword.  If so, silently skip — the path will be found
                    # later by the filter(PathElement) lookup.
                    keyword = 'notehead' if 'notehead' in eid.lower() else 'hollow'
                    has_descendant_path = False
                    if hasattr(elem, 'descendants'):
                        for child in elem.descendants():
                            if isinstance(child, inkex.PathElement) and \
                               keyword in child.get('id', '').lower():
                                has_descendant_path = True
                                break
                    if has_descendant_path:
                        continue
                    tag_name = elem.TAG.split('}')[-1]
                    raise inkex.AbortExtension(
                        f"Wait! The object '{eid}' is a <{tag_name}>, not a Path.\n\n"
                        "Please select it and press Ctrl+Shift+C (Path > Object to Path) before running this tool."
                    )
        
        layer = svg.get_current_layer()
        S = parse_semitones(self.options.nh_semitones, "Notehead Height", 0.5, 20.0)
        TARGET_HEIGHT = S * 100.0
        
        noteheads = [e for e in svg.descendants().filter(PathElement) if 'notehead' in e.get('id', '').lower()]
        hollows = [e for e in svg.descendants().filter(PathElement) if 'hollow' in e.get('id', '').lower()]

        # Find existing stems (could be rect or path from previous runs)
        existing_stem_up = self._find_existing_stem('stem-up')
        existing_stem_down = self._find_existing_stem('stem-down')

        if len(noteheads) != 1: raise inkex.AbortExtension("Exactly 1 'notehead' path is required.")
        notehead = noteheads[0]
        hollow = hollows[0] if len(hollows) > 0 else None

        notehead.set('id', 'notehead')
        notehead.set('data-semitone-height', str(S))
        if hollow is not None:
            hollow.set('id', 'hollow')

        # --- Center and scale path elements only (not stem rects) ---
        # Skip if already processed (idempotent re-runs)
        path_elements = [e for e in svg.descendants().filter(PathElement)]
        
        bbox = notehead.bounding_box()
        already_centered = abs(bbox.center.x) < 1.0 and abs(bbox.center.y) < 1.0
        already_scaled = abs(bbox.height - TARGET_HEIGHT) < 1.0
        
        if not (already_centered and already_scaled):
            cx, cy = bbox.center.x, bbox.center.y
            for elem in path_elements:
                elem.transform.add_translate(-cx, -cy)
                elem.apply_transform()

            bbox = notehead.bounding_box()
            scale_factor = TARGET_HEIGHT / bbox.height
            for elem in path_elements:
                elem.transform.add_scale(scale_factor)
                elem.apply_transform()

            bbox = notehead.bounding_box()

        # --- Set viewport to exact notehead dimensions ---
        # Do this early so Inkscape knows the document size before we add guides
        DEC_PLACES = self.options.round_decimals
        nw = bbox.width
        nh = bbox.height
        nwh = nw / nh  # width-to-height ratio
        tnh = TARGET_HEIGHT  # target SVG notehead height
        tnw = tnh * nwh      # target SVG notehead width
        
        vbMinX = round(-0.5 * tnw, DEC_PLACES)
        vbMinY = round(-0.5 * tnh, DEC_PLACES)
        
        svg.set('width', f"{round(tnw, DEC_PLACES)}")
        svg.set('height', f"{round(tnh, DEC_PLACES)}")
        svg.set('viewBox', f"{vbMinX:.{DEC_PLACES}f} {vbMinY:.{DEC_PLACES}f} {round(tnw, DEC_PLACES):.{DEC_PLACES}f} {round(tnh, DEC_PLACES):.{DEC_PLACES}f}")

        # --- Compute stem attachment points ---
        # Use true geometric extremes (including curve apexes) rather than
        # just path command endpoints, so stems attach at the actual widest
        # points of curved shapes like rounded diamonds and ovals.
        d_str = notehead.get('d', '')
        nh_commands = parse_and_normalize(d_str)
        points = path_extreme_points(nh_commands)

        maxX = max(p[0] for p in points)
        right_points = [p[1] for p in points if abs(p[0] - maxX) < 0.1]
        minY_at_maxX = min(right_points) if right_points else bbox.top

        minX = min(p[0] for p in points)
        left_points = [p[1] for p in points if abs(p[0] - minX) < 0.1]
        maxY_at_minX = max(left_points) if left_points else bbox.bottom

        w_stem = max(TARGET_HEIGHT / 9.0, 100.0 / 5.0)

        # --- Always remove old stem rects first ---
        if existing_stem_up is not None:
            parent = existing_stem_up.getparent()
            if parent is not None:
                parent.remove(existing_stem_up)
        if existing_stem_down is not None:
            parent = existing_stem_down.getparent()
            if parent is not None:
                parent.remove(existing_stem_down)

        # Clean up stale data-stem-* attributes from prior extension versions
        for attr in ('data-stem-up-x', 'data-stem-up-y',
                     'data-stem-down-x', 'data-stem-down-y'):
            if notehead.get(attr) is not None:
                del notehead.attrib[attr]

        if self.options.stem_placemarkers:
            # --- Generate stems as rectangles ---
            SVG_NS = '{http://www.w3.org/2000/svg}'
            x_up = maxX - w_stem
            y_up = minY_at_maxX - STEM_HEIGHT
            stem_up = inkex.etree.SubElement(layer, f'{SVG_NS}rect', attrib={
                'id': 'stem-up',
                'x': str(x_up), 'y': str(y_up),
                'width': str(w_stem), 'height': str(STEM_HEIGHT),
                'style': f'fill:{self.options.color_stem_up};stroke:none',
            })

            x_down = minX
            y_down = maxY_at_minX
            stem_down = inkex.etree.SubElement(layer, f'{SVG_NS}rect', attrib={
                'id': 'stem-down',
                'x': str(x_down), 'y': str(y_down),
                'width': str(w_stem), 'height': str(STEM_HEIGHT),
                'style': f'fill:{self.options.color_stem_down};stroke:none',
            })

            # Reorder: stems behind notehead/hollow for clean z-order
            for el in [stem_down, stem_up, notehead, hollow]:
                if el is not None:
                    parent = el.getparent()
                    if parent is not None:
                        parent.remove(el)
                    layer.append(el)

            # --- Semitone guidelines: notehead bbox + both stems ---
            all_y_min = min(-tnh / 2.0, y_up)
            all_y_max = max(tnh / 2.0, y_down + STEM_HEIGHT)
            self.generate_guidelines(tnw, tnh, all_y_min, all_y_max,
                                     show=self.options.show_guidelines)

            # --- Full view encompassing stems ---
            stem_coords = {
                'x_up': x_up, 'y_up': y_up,
                'x_down': x_down, 'y_down': y_down,
                'w_stem': w_stem,
            }
            self.generate_full_view(tnw, tnh, stem_coords)

        else:
            # --- No stems (semibreve / breve) ---
            # Semitone guidelines cover notehead only
            self.generate_guidelines(tnw, tnh, -tnh / 2.0, tnh / 2.0,
                                     show=self.options.show_guidelines)
            # Full view covers notehead only
            self.generate_full_view(tnw, tnh, stem_coords=None)

    def process_glyph(self):
        svg = self.svg
        
        # --- BAKE PRE-EXISTING TRANSFORMS ---
        # Normalize arcs to cubics, then apply any transforms (flips,
        # rotations, etc.) into path data before processing.
        self._bake_transforms()
        
        # --- SAFETY CHECK ---
        # Allow rect elements with stem IDs, line/g elements from guidelines layer,
        # and view elements — block everything else that isn't a path
        unconverted = []
        for e in svg.descendants():
            eid = e.get('id', '')
            tag = e.TAG.split('}')[-1]
            # Skip elements managed by the toolkit
            if eid in MANAGED_IDS or eid.startswith('guideline-semi-'):
                continue
            # Skip elements inside the guidelines layer
            parent = e.getparent()
            if parent is not None and parent.get('id') == GUIDELINES_LAYER_ID:
                continue
            if tag in ('text', 'circle', 'ellipse', 'polygon', 'polyline'):
                unconverted.append(e)
            elif tag == 'rect' and eid not in STEM_IDS:
                unconverted.append(e)
        if unconverted:
            raise inkex.AbortExtension(
                "Found unconverted text or basic shapes on the canvas.\n\n"
                "Please select all objects (Ctrl+A) and press Ctrl+Shift+C (Path > Object to Path), "
                "and ensure they are unioned (Ctrl++) before running this tool."
            )
            
        S = parse_semitones(self.options.glyph_semitones, "Target Height", 0.5, 50.0)
        TARGET_HEIGHT = S * 100.0
        
        all_paths = list(svg.descendants().filter(PathElement))
        if not all_paths: raise inkex.AbortExtension("No paths found.")
            
        collective_bbox = None
        for path in all_paths:
            pb = path.bounding_box()
            if pb is not None:
                if collective_bbox is None: collective_bbox = pb
                else: collective_bbox += pb
                
        if collective_bbox is None: raise inkex.AbortExtension("Could not calculate bounding box.")
        
        cx, cy = collective_bbox.center.x, collective_bbox.center.y
        for path in all_paths:
            path.transform.add_translate(-cx, -cy)
            path.apply_transform()
            
        scale_factor = TARGET_HEIGHT / collective_bbox.height
        for path in all_paths:
            path.transform.add_scale(scale_factor)
            path.apply_transform()
            path.set('data-semitone-height', str(S))
        
        # --- Set viewport to exact glyph dimensions ---
        DEC_PLACES = self.options.round_decimals
        nw = collective_bbox.width
        nh = collective_bbox.height
        nwh = nw / nh  # width-to-height ratio
        tnh = TARGET_HEIGHT
        tnw = tnh * nwh
        
        vbMinX = round(-0.5 * tnw, DEC_PLACES)
        vbMinY = round(-0.5 * tnh, DEC_PLACES)
        
        svg.set('width', f"{round(tnw, DEC_PLACES)}")
        svg.set('height', f"{round(tnh, DEC_PLACES)}")
        svg.set('viewBox', f"{vbMinX:.{DEC_PLACES}f} {vbMinY:.{DEC_PLACES}f} {round(tnw, DEC_PLACES):.{DEC_PLACES}f} {round(tnh, DEC_PLACES):.{DEC_PLACES}f}")
        
        # --- Semitone guidelines as SVG lines in a toggleable layer ---
        # Glyph mode: no stems, extent is just the glyph bbox
        self.generate_guidelines(tnw, tnh, -tnh / 2.0, tnh / 2.0,
                                 show=self.options.show_guidelines)
        
        # --- Full view (no stems in glyph mode) ---
        self.generate_full_view(tnw, tnh, stem_coords=None)

    def export_lilypond(self):
        """Export all path elements as LilyPond Scheme definitions.
        
        Produces:
            - \\version header for LilyPond compilation
            - Scheme path defines for each non-stem path element
            - Stem attachment point defines (computed from stem rect geometry,
              scaled to staff-space units with Y-axis inversion)
        """
        doc_name = self.svg.get('sodipodi:docname', 'music-glyph.svg')
        base_name = os.path.splitext(doc_name)[0]
        clean_base = re.sub(r'[^a-zA-Z0-9-]', '-', base_name).strip('-')

        out_lines = [
            '\\version "2.24.2"',
            "",
            f"%% Generated from: {doc_name}",
            "%% Coordinates: scaled to staff-space units (/100), Y-axis inverted for LilyPond.",
            "%% Usage: wrap path list in (make-path-stencil '(...) X Y) to create a stencil.",
            ""
        ]

        # Track notehead for stem attachment export
        notehead_elem = None

        for elem in self.svg.descendants().filter(PathElement):
            eid = elem.get('id', 'path')
            if eid in STEM_IDS:
                continue
            
            if 'notehead' in eid.lower():
                notehead_elem = elem
                
            clean_id = re.sub(r'[^a-zA-Z0-9-]', '-', eid).strip('-')
            var_name = f"{clean_base}-{clean_id}-path"

            d_str = elem.get('d')
            commands = parse_and_normalize(d_str)
            scheme_code = format_scheme(commands)

            out_lines.append(f"#(define {var_name} '(")
            out_lines.append(scheme_code)
            out_lines.append("))")
            out_lines.append("")

        # Generate stem attachment point defines (computed from stem rects)
        # Only for noteheads, and only when stem placemarkers are enabled.
        if notehead_elem is not None:
            eid = notehead_elem.get('id', 'notehead')
            clean_id = re.sub(r'[^a-zA-Z0-9-]', '-', eid).strip('-')
            
            if self.options.stem_placemarkers:
                # Find stem-up and stem-down rect elements by ID
                stem_up_elem = None
                stem_down_elem = None
                for elem in self.svg.descendants():
                    sid = elem.get('id', '')
                    if sid == 'stem-up':
                        stem_up_elem = elem
                    elif sid == 'stem-down':
                        stem_down_elem = elem
                
                # Validate: if stems are expected but missing, error
                missing = []
                if stem_up_elem is None:
                    missing.append('stem-up')
                if stem_down_elem is None:
                    missing.append('stem-down')
                if missing:
                    raise inkex.AbortExtension(
                        f"Missing stem placemarker(s): {', '.join(missing)}.\n\n"
                        "Either:\n"
                        "  • Uncheck 'Stem placemarkers' in the Notehead tab "
                        "(for stemless noteheads like semibreves/breves), or\n"
                        "  • Re-run 'Update SVG Canvas' with 'Stem placemarkers' "
                        "checked to regenerate them."
                    )
                
                if stem_up_elem is not None:
                    # Attachment point: center-x of rect, bottom edge (where stem meets notehead)
                    su_x = float(stem_up_elem.get('x', '0'))
                    su_y = float(stem_up_elem.get('y', '0'))
                    su_w = float(stem_up_elem.get('width', '0'))
                    su_h = float(stem_up_elem.get('height', '0'))
                    ax = round((su_x + su_w / 2.0) * 0.01, 4)
                    ay = round(-(su_y + su_h) * 0.01, 4)  # negate Y for LilyPond
                    var_stem_up = f"{clean_base}-{clean_id}-stem-up"
                    out_lines.append(f"#(define {var_stem_up} '({ax} . {ay}))")
                    
                if stem_down_elem is not None:
                    # Attachment point: center-x of rect, top edge (where stem meets notehead)
                    sd_x = float(stem_down_elem.get('x', '0'))
                    sd_y = float(stem_down_elem.get('y', '0'))
                    sd_w = float(stem_down_elem.get('width', '0'))
                    ax = round((sd_x + sd_w / 2.0) * 0.01, 4)
                    ay = round(-sd_y * 0.01, 4)  # negate Y for LilyPond
                    var_stem_down = f"{clean_base}-{clean_id}-stem-down"
                    out_lines.append(f"#(define {var_stem_down} '({ax} . {ay}))")
            
            out_lines.append("")

        try:
            out_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.exists(out_dir): out_dir = os.path.expanduser("~")
        except Exception:
            out_dir = "" 
            
        out_path = os.path.join(out_dir, f"{clean_base}.ly")
        with open(out_path, 'w') as f:
            f.write("\n".join(out_lines))
            
        raise inkex.AbortExtension(f"LilyPond export complete!\n\nFile saved to:\n{out_path}\n\n(Your SVG canvas was left unchanged.)")


if __name__ == '__main__':
    MusicGlyphToolkit().run()
