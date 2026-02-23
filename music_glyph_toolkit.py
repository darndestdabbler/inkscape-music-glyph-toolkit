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
def arc_to_curves(x1, y1, rx, ry, phi_deg, large_arc, sweep, x2, y2):
    """Convert an SVG elliptical arc to one or more cubic Bézier curves.

    Implements the SVG arc parameterization algorithm (F.6.5/F.6.6) to find
    the center, then approximates each ≤90° arc segment with a cubic Bézier.

    Edge cases:
        - Coincident endpoints → empty list
        - Zero radius → [('L', [x2, y2])]
        - Negative radii → treated as positive (per SVG spec)
    """
    if x1 == x2 and y1 == y2: return []
    if rx == 0 or ry == 0: return [('L', [x2, y2])]
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi_deg)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    lambda_sq = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lambda_sq > 1:
        lambda_val = math.sqrt(lambda_sq)
        rx *= lambda_val
        ry *= lambda_val
    rx_sq, ry_sq = rx * rx, ry * ry
    x1p_sq, y1p_sq = x1p * x1p, y1p * y1p
    numerator = rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq
    denominator = rx_sq * y1p_sq + ry_sq * x1p_sq
    factor = 0 if denominator == 0 or numerator < 0 else math.sqrt(numerator / denominator)
    if large_arc == sweep: factor = -factor
    cxp = factor * rx * y1p / ry
    cyp = -factor * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2

    def angle(ux, uy, vx, vy):
        n = math.sqrt(ux * ux + uy * uy) * math.sqrt(vx * vx + vy * vy)
        if n == 0: return 0
        c = max(-1, min(1, (ux * vx + uy * vy) / n))
        a = math.acos(c)
        return -a if ux * vy - uy * vx < 0 else a

    theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0: dtheta -= 2 * math.pi
    elif sweep and dtheta < 0: dtheta += 2 * math.pi
    
    n_segments = max(1, int(math.ceil(abs(dtheta) / (math.pi / 2))))
    d_theta = dtheta / n_segments
    curves = []
    theta = theta1
    
    for _ in range(n_segments):
        t = math.tan(d_theta / 4)
        alpha = math.sin(d_theta) * (math.sqrt(4 + 3 * t * t) - 1) / 3
        cos_t1, sin_t1 = math.cos(theta), math.sin(theta)
        cos_t2, sin_t2 = math.cos(theta + d_theta), math.sin(theta + d_theta)
        
        p1x, p1y = rx * cos_t1, ry * sin_t1
        p2x, p2y = rx * cos_t2, ry * sin_t2
        d1x, d1y = -rx * sin_t1, ry * cos_t1
        d2x, d2y = -rx * sin_t2, ry * cos_t2
        
        cp1x, cp1y = p1x + alpha * d1x, p1y + alpha * d1y
        cp2x, cp2y = p2x - alpha * d2x, p2y - alpha * d2y
        
        def transform_point(px, py):
            return cos_phi * px - sin_phi * py + cx, sin_phi * px + cos_phi * py + cy
            
        x_cp1, y_cp1 = transform_point(cp1x, cp1y)
        x_cp2, y_cp2 = transform_point(cp2x, cp2y)
        x_end, y_end = transform_point(p2x, p2y)
        curves.append(('C', [x_cp1, y_cp1, x_cp2, y_cp2, x_end, y_end]))
        theta += d_theta
        
    return curves

def quadratic_to_cubic(x0, y0, x1, y1, x2, y2):
    """Elevate a quadratic Bézier to a cubic using the 2/3 rule.

    Returns (cx1, cy1, cx2, cy2, x2, y2) — the cubic control points and endpoint.
    """
    return (x0 + 2/3 * (x1 - x0), y0 + 2/3 * (y1 - y0),
            x2 + 2/3 * (x1 - x2), y2 + 2/3 * (y1 - y2), x2, y2)

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

class MusicGlyphToolkit(inkex.EffectExtension):
    def add_arguments(self, pars):
        pars.add_argument("--active_tab", type=str, default="tab_notehead")
        pars.add_argument("--nh_semitones", type=float, default=2.0)
        pars.add_argument("--color_stem_up", type=str, default="#FF00FF")
        pars.add_argument("--color_stem_down", type=str, default="#00FFFF")
        pars.add_argument("--glyph_semitones", type=float, default=4.0)
        pars.add_argument("--output_format", type=str, default="svg")
        pars.add_argument("--strict_mclz", type=inkex.Boolean, default=True)
        pars.add_argument("--round_decimals", type=int, default=3)
        pars.add_argument("--show_guidelines", type=inkex.Boolean, default=True)

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

    def _find_existing_stem(self, stem_id):
        """Find an existing stem element (rect or path) by ID."""
        for elem in self.svg.descendants():
            if elem.get('id') == stem_id:
                return elem
        return None

    def process_notehead(self):
        svg = self.svg
        
        # --- SAFETY CHECK ---
        # Check for unconverted objects that should be paths
        # (exclude rect elements with stem IDs — those are our managed stems)
        for elem in svg.descendants():
            eid = elem.get('id', '')
            if eid and ('notehead' in eid.lower() or 'hollow' in eid.lower()):
                if not isinstance(elem, inkex.PathElement):
                    tag_name = elem.TAG.split('}')[-1]
                    raise inkex.AbortExtension(
                        f"Wait! The object '{eid}' is a <{tag_name}>, not a Path.\n\n"
                        "Please select it and press Ctrl+Shift+C (Path > Object to Path) before running this tool."
                    )
        
        layer = svg.get_current_layer()
        S = self.options.nh_semitones
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
        notehead.path = notehead.path.to_absolute()
        points = []
        for cmd in notehead.path:
            if len(cmd.args) >= 2: points.append((cmd.args[-2], cmd.args[-1]))

        maxX = max(p[0] for p in points)
        right_points = [p[1] for p in points if abs(p[0] - maxX) < 0.1]
        minY_at_maxX = min(right_points) if right_points else bbox.top

        minX = min(p[0] for p in points)
        left_points = [p[1] for p in points if abs(p[0] - minX) < 0.1]
        maxY_at_minX = max(left_points) if left_points else bbox.bottom

        w_stem = max(TARGET_HEIGHT / 9.0, 100.0 / 5.0)

        # --- Generate stems as rectangles ---
        # If stems already exist, remove old ones and regenerate
        if existing_stem_up is not None:
            parent = existing_stem_up.getparent()
            if parent is not None:
                parent.remove(existing_stem_up)
        if existing_stem_down is not None:
            parent = existing_stem_down.getparent()
            if parent is not None:
                parent.remove(existing_stem_down)

        # Stem-up: right side of notehead, extends upward
        # Rect x is flush with right edge minus stem width
        SVG_NS = '{http://www.w3.org/2000/svg}'
        x_up = maxX - w_stem
        y_up = minY_at_maxX - STEM_HEIGHT
        stem_up = inkex.etree.SubElement(layer, f'{SVG_NS}rect', attrib={
            'id': 'stem-up',
            'x': str(x_up), 'y': str(y_up),
            'width': str(w_stem), 'height': str(STEM_HEIGHT),
            'style': f'fill:{self.options.color_stem_up};stroke:none',
        })

        # Stem-down: left side of notehead, extends downward
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

        # Store stem attachment points as data attributes for export
        notehead.set('data-stem-up-x', str(round(x_up + w_stem / 2.0, 3)))
        notehead.set('data-stem-up-y', str(round(minY_at_maxX, 3)))
        notehead.set('data-stem-down-x', str(round(x_down + w_stem / 2.0, 3)))
        notehead.set('data-stem-down-y', str(round(maxY_at_minX, 3)))
        
        # --- Semitone guidelines as SVG lines in a toggleable layer ---
        # Vertical extent: notehead bbox + both stems
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

    def process_glyph(self):
        svg = self.svg
        
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
            
        S = self.options.glyph_semitones
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
            - Stem attachment point defines (x, y pairs in staff-space units)
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

        # Generate stem attachment point defines
        if notehead_elem is not None:
            eid = notehead_elem.get('id', 'notehead')
            clean_id = re.sub(r'[^a-zA-Z0-9-]', '-', eid).strip('-')
            
            stem_up_x = notehead_elem.get('data-stem-up-x')
            stem_up_y = notehead_elem.get('data-stem-up-y')
            stem_down_x = notehead_elem.get('data-stem-down-x')
            stem_down_y = notehead_elem.get('data-stem-down-y')
            
            if stem_up_x is not None and stem_up_y is not None:
                # Scale /100 and negate Y for LilyPond coordinates
                sx = round(float(stem_up_x) * 0.01, 4)
                sy = round(-float(stem_up_y) * 0.01, 4)
                var_stem_up = f"{clean_base}-{clean_id}-stem-up"
                out_lines.append(f"#(define {var_stem_up} '({sx} . {sy}))")
                
            if stem_down_x is not None and stem_down_y is not None:
                sx = round(float(stem_down_x) * 0.01, 4)
                sy = round(-float(stem_down_y) * 0.01, 4)
                var_stem_down = f"{clean_base}-{clean_id}-stem-down"
                out_lines.append(f"#(define {var_stem_down} '({sx} . {sy}))")
            
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
