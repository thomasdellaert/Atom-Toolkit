"""Turn a Grotrian spec into positioned geometry.

Everything here is a pure function over plain dataclasses: no SVG, no matplotlib, no I/O.
That is what makes layout testable -- assertions run against coordinates rather than
pixels.

Coordinates are SVG user units with y increasing downwards, so higher energy means
*smaller* y.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .scale import EnergyScale, LinearScale
from .spec import ColumnPolicy, LevelSpec, Theme

__all__ = ['Scene', 'LevelLine', 'Label', 'Leader', 'AxisTick', 'Anchor', 'ColumnHeader',
           'Bracket', 'ScaleBar', 'Arrow', 'layout', 'resolve_overlaps',
           'hyperfine_scale', 'hyperfine_reference', 'format_splitting',
           'transition_wavelength_nm']


# ---------------------------------------------------------------------------
# Scene primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Anchor:
    """The drawn extent of a level. Transitions attach to these, never to raw energies."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class LevelLine:
    x0: float
    x1: float
    y: float
    color: str
    linewidth: float
    opacity: float = 1.0
    dashes: Optional[str] = None
    title: str = ''          #: SVG <title>, shown as a native tooltip


@dataclass(frozen=True)
class Label:
    x: float
    y: float
    runs: Tuple[Tuple[str, str], ...]    #: (text, role) from Term.markup_runs
    color: str
    size: float
    anchor: str = 'start'                #: SVG text-anchor


@dataclass(frozen=True)
class Leader:
    x0: float
    y0: float
    x1: float
    y1: float
    color: str


@dataclass(frozen=True)
class AxisTick:
    y: float
    label: str


@dataclass(frozen=True)
class ColumnHeader:
    x: float
    y: float
    text: str


@dataclass(frozen=True)
class Bracket:
    """A square bracket grouping a manifold's hyperfine sublevels."""
    x: float
    y0: float
    y1: float
    width: float
    color: str
    dashed: bool = False     #: dashed means the spacing is schematic, not physical


@dataclass(frozen=True)
class Arrow:
    """A transition. ``control`` is None for a straight line, else a quadratic control."""
    x0: float
    y0: float
    x1: float
    y1: float
    color: str
    width: float
    control: Optional[Tuple[float, float]] = None
    polygon: Optional[Tuple[Tuple[float, float], ...]] = None   #: set for block arrows
    dashed: bool = False
    wavy: bool = False
    head: str = 'end'
    opacity: float = 1.0
    title: str = ''


@dataclass(frozen=True)
class ScaleBar:
    """Annotates the single global hyperfine magnification."""
    x: float
    y0: float
    y1: float
    label: str
    color: str


@dataclass
class Scene:
    """Everything needed to draw, and nothing about how to draw it."""
    width: float
    height: float
    theme: Theme
    lines: List[LevelLine] = field(default_factory=list)
    labels: List[Label] = field(default_factory=list)
    leaders: List[Leader] = field(default_factory=list)
    ticks: List[AxisTick] = field(default_factory=list)
    headers: List[ColumnHeader] = field(default_factory=list)
    brackets: List[Bracket] = field(default_factory=list)
    arrows: List[Arrow] = field(default_factory=list)
    scalebar: Optional[ScaleBar] = None
    anchors: Dict[object, Anchor] = field(default_factory=dict)
    axis_label: str = ''
    title: str = ''
    plot_top: float = 0.0
    plot_bottom: float = 0.0
    axis_x: float = 0.0


# ---------------------------------------------------------------------------
# Label de-overlap
# ---------------------------------------------------------------------------

def resolve_overlaps(desired: Sequence[float], min_gap: float) -> List[float]:
    """Spread values apart to at least ``min_gap``, moving them as little as possible.

    Minimises the sum of squared displacement subject to ``y[i+1] - y[i] >= min_gap``.
    Substituting ``z[i] = y[i] - i*min_gap`` turns that into an isotonic regression, which
    pool-adjacent-violators solves exactly in linear time -- much better than nudging
    greedily, which piles up error at one end.

    :param desired: preferred positions, ascending
    :param min_gap: minimum separation
    :return: adjusted positions, same order as the input
    """
    n = len(desired)
    if n == 0:
        return []
    z = [value - i * min_gap for i, value in enumerate(desired)]

    # PAVA: maintain blocks of (weighted mean, count), merging while decreasing
    means: List[float] = []
    counts: List[int] = []
    for value in z:
        means.append(value)
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            total = means[-2] * counts[-2] + means[-1] * counts[-1]
            count = counts[-2] + counts[-1]
            means[-2:], counts[-2:] = [total / count], [count]

    flat: List[float] = []
    for mean, count in zip(means, counts):
        flat.extend([mean] * count)
    return [value + i * min_gap for i, value in enumerate(flat)]


def estimate_text_width(runs: Sequence[Tuple[str, str]], size: float) -> float:
    """Rough advance width for a run sequence.

    SVG has no layout engine, so exact metrics would need font tables. Phase 1 reserves a
    fixed label gutter instead (see the plan, §7), and this estimate is only used for
    canvas sizing -- being a little generous is harmless.
    """
    width = 0.0
    for text, role in runs:
        factor = 0.62 if role == 'normal' else 0.62 * 0.72
        width += len(text) * size * factor
    return width


# ---------------------------------------------------------------------------
# Hyperfine structure
# ---------------------------------------------------------------------------

def hyperfine_span(level) -> float:
    """Total hyperfine spread of a manifold, in Hz. Zero if it has no splitting data."""
    subs = level.sublevels()
    if len(subs) < 2:
        return 0.0
    shifts = [s.shift_Hz for s in subs]
    return max(shifts) - min(shifts)


def hyperfine_reference(levels: Sequence, mode: str = 'auto') -> float:
    """Pick the splitting, in Hz, that the hyperfine band should be sized against.

    Never the smallest *gap*: near-degeneracies make that explode. 137Ba+ 5d 2D5/2 has
    F=4 and F=3 just 0.496 MHz apart inside a 135 MHz span, a ratio of 272.

    ``'auto'`` uses the largest span among the **excited** manifolds, ignoring the
    lowest-energy one. Ground-state hyperfine splittings run several times larger than
    anything else -- measured across the bundled species the ratio is 5.5x in 43Ca+,
    5.4x in 137Ba+, 8.4x in 87Rb and 7.9x in 133Cs -- so scaling to the ground state
    squashes every other manifold to a sliver. Scaling to the excited states instead
    keeps them legible and lets the ground state overflow proportionally, which is both
    honest and the usual convention.

    ``'widest'`` restores sizing to the largest span present, ground state included.

    :param levels: manifolds being drawn with substructure, any order
    :param mode: ``'auto'`` or ``'widest'``
    :return: the reference splitting in Hz, or 0.0 if nothing has a splitting
    """
    with_data = [(level.level_Hz, hyperfine_span(level)) for level in levels]
    with_data = [(energy, span) for energy, span in with_data if span > 0]
    if not with_data:
        return 0.0
    if mode == 'auto' and len(with_data) >= 2:
        with_data.sort(key=lambda item: item[0])
        return max(span for _, span in with_data[1:])     # skip the lowest-lying manifold
    return max(span for _, span in with_data)


def hyperfine_scale(levels: Sequence, band_px: float, mode: str = 'widest') -> float:
    """Global hyperfine magnification in px per Hz. See :func:`hyperfine_reference`."""
    reference = hyperfine_reference(levels, mode)
    return band_px / reference if reference > 0 else 0.0


def zeeman_only_spread(level, b_gauss: float) -> float:
    """Zeeman spread of a manifold with the hyperfine offset removed, in Hz.

    ``level_hz_at`` returns hyperfine *plus* Zeeman; subtracting the parent F-level's
    zero-field shift isolates the Zeeman part, which is what an exaggerated Zeeman scale
    has to act on.
    """
    shifts = []
    for hf in level.sublevels():
        for z in hf.sublevels():
            shifts.append(z.level_hz_at(b_gauss) - level.level_Hz - hf.shift_Hz)
    if len(shifts) < 2:
        return 0.0
    return max(shifts) - min(shifts)


def zeeman_spread(level, b_gauss: float) -> float:
    """Total Zeeman spread of a manifold at ``b_gauss``, in Hz.

    Routes through ``ZLevel.level_hz_at``, so manifolds with
    ``use_nonlinear_zeeman = True`` get exact H_HF + H_Z diagonalisation for free.
    """
    shifts = []
    for hf in level.sublevels():
        for z in hf.sublevels():
            shifts.append(z.level_hz_at(b_gauss) - level.level_Hz)
    if len(shifts) < 2:
        return 0.0
    return max(shifts) - min(shifts)


def resolve_level_scale(style_value, exaggerate: float, level, band_px: float,
                        inherited: float) -> float:
    """Per-manifold magnification, falling back to the diagram-wide one.

    :param style_value: ``None`` to inherit, ``'fit'`` to make this manifold fill the
        band, a px/Hz float, or a splitting ``Quantity``
    :param exaggerate: multiplier applied on top
    :param level: the manifold, needed for ``'fit'``
    :param band_px: the nominal band height
    :param inherited: the diagram-wide scale
    """
    if style_value is None:
        base = inherited
    elif style_value == 'fit':
        span = hyperfine_span(level)
        base = band_px / span if span > 0 else inherited
    else:
        base = _as_scale(style_value, band_px)
    return base * exaggerate


def _as_scale(value, band_px: float) -> float:
    """Interpret a user-supplied scale: px per Hz, or a splitting that fills the band."""
    if hasattr(value, 'to'):                      # a pint Quantity, e.g. Q_(1, 'GHz')
        hz = float(value.to('Hz', 'spectroscopy').magnitude)
        return band_px / hz if hz else 0.0
    return float(value)


def format_splitting(hz: float) -> str:
    """Human-readable frequency for annotations."""
    for threshold, unit, scale in ((1e12, 'THz', 1e12), (1e9, 'GHz', 1e9),
                                   (1e6, 'MHz', 1e6), (1e3, 'kHz', 1e3)):
        if abs(hz) >= threshold:
            return f'{hz / scale:.3g} {unit}'
    return f'{hz:.3g} Hz'


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

_C_NM_HZ = 2.99792458e17      # speed of light in nm/s


def transition_wavelength_nm(lower, upper) -> float:
    """Vacuum wavelength of a transition between two levels, in nm.

    Computed straight from the level energies rather than from a Transition object, so it
    works for any pair -- including sublevel pairs that were never added to the atom.
    """
    delta = abs(upper.level_Hz - lower.level_Hz)
    return _C_NM_HZ / delta if delta > 0 else float('inf')


def _resolve_anchor(scene: Scene, level):
    """Find the drawn box for a level, falling back up the hierarchy.

    A transition may point at a sublevel of a manifold that was drawn collapsed, or at a
    manifold whose sublevels were drawn individually. Walking up to the nearest drawn
    ancestor is what makes both cases land on something sensible.
    """
    if level in scene.anchors:
        return scene.anchors[level]
    for attr in ('parent', 'manifold'):
        ancestor = getattr(level, attr, None)
        if ancestor is not None and ancestor in scene.anchors:
            return scene.anchors[ancestor]
    return None


def _port(anchor: Anchor, mode: str, toward_right: Optional[bool],
          partner_y: Optional[float] = None) -> Tuple[float, float]:
    """Pick the attachment point on a level.

    For a manifold drawn as a hyperfine fan the anchor spans the whole band, and its
    centre falls in the gap *between* sublevel lines -- an arrow drawn from there appears
    to start in mid-air. Attach to the edge of the band facing the partner instead, so
    the arrow always meets a drawn line.
    """
    x_mid, y_mid = anchor.center
    if partner_y is not None and anchor.y1 > anchor.y0:
        y_mid = anchor.y0 if partner_y < anchor.y0 else (
            anchor.y1 if partner_y > anchor.y1 else y_mid)
    if mode == 'left':
        return anchor.x0, y_mid
    if mode == 'right':
        return anchor.x1, y_mid
    if mode == 'center' or toward_right is None:
        return x_mid, y_mid
    # 'auto' across columns: attach on the side facing the partner, a third of the way in
    third = (anchor.x1 - anchor.x0) / 3.0
    return (x_mid + third if toward_right else x_mid - third), y_mid


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def layout(specs: Sequence[LevelSpec],
           theme: Theme,
           policy: ColumnPolicy,
           transitions: Sequence['TransitionSpec'] = (),
           scale: Optional[EnergyScale] = None,
           title: str = '',
           hf_scale: Union[str, float] = 'auto',
           b_gauss: float = 0.0,
           z_scale: Union[str, float] = 'match') -> Scene:
    """Position every level, sublevel, label and tick.

    :param specs: what to draw
    :param theme: sizes and colours
    :param policy: column assignment
    :param transitions: transitions to draw between them
    :param scale: gross energy transform; defaults to :class:`LinearScale`
    :param title: optional diagram title
    :param hf_scale: hyperfine magnification in px per Hz, or ``'auto'`` to fit the
        largest span present into ``theme.hf_band``. One value is shared by the whole
        diagram so hyperfine sizes stay comparable between manifolds.
    :param b_gauss: magnetic field for ``detail='zeeman'`` levels. At B=0 the Zeeman fan
        is purely horizontal, which is the conventional Grotrian look.
    :param z_scale: ``'match'`` (default) to keep Zeeman on the hyperfine scale,
        ``'auto'`` to exaggerate it so the largest Zeeman spread fills ``theme.z_band``,
        or an explicit px/Hz value or splitting Quantity.
    :return: a :class:`Scene`
    """
    if not specs:
        raise ValueError('Nothing to draw: add at least one level.')

    scale = scale or LinearScale()
    scale.prepare([spec.level.level_Hz for spec in specs])

    hf_specs = [spec for spec in specs if spec.style.detail == 'hyperfine']
    z_specs = [spec for spec in specs if spec.style.detail == 'zeeman']
    any_hf = bool(hf_specs) or bool(z_specs)

    # One vertical magnification for *all* substructure, hyperfine and Zeeman alike.
    # Zeeman sublevels are displaced vertically by the hyperfine shift plus the (much
    # smaller) Zeeman shift, so giving Zeeman its own scale would silently rescale the
    # hyperfine splitting that dominates it. Sharing one scale keeps every substructure
    # in the figure directly comparable, which is the whole point of a global scale.
    sub_levels = [spec.level for spec in hf_specs + z_specs]
    hf_reference = 0.0
    if isinstance(hf_scale, str):
        hf_reference = hyperfine_reference(sub_levels, hf_scale)
        m_hf = theme.hf_band / hf_reference if hf_reference > 0 else 0.0
    else:
        m_hf = _as_scale(hf_scale, theme.hf_band)

    # Zeeman may run on its own, exaggerated scale. 'match' keeps it faithful to the
    # hyperfine scale, which is honest but usually renders the Zeeman splitting invisible
    # -- it is orders of magnitude smaller at ordinary fields.
    z_reference = 0.0
    if z_scale == 'match':
        m_z = m_hf
    elif z_scale == 'auto':
        spreads = [zeeman_only_spread(spec.level, b_gauss) for spec in z_specs]
        z_reference = max(spreads) if spreads else 0.0
        m_z = theme.z_band / z_reference if z_reference > 0 else m_hf
    else:
        m_z = _as_scale(z_scale, theme.z_band)

    # --- columns -----------------------------------------------------------
    keyed = []
    for spec in specs:
        sort_key, label = policy(spec.level)
        if spec.column is not None:
            label = spec.column
        keyed.append((sort_key, label, spec))
    column_order = sorted({(k, lbl) for k, lbl, _ in keyed})
    column_index = {lbl: i for i, (_, lbl) in enumerate(column_order)}

    # hyperfine diagrams need a sub-gutter for 'F = n' labels, left of the term symbol
    hf_gutter = theme.hf_label_gutter if any_hf else 0.0
    n_columns = len(column_order)
    col_pitch = theme.column_width + hf_gutter + theme.label_gutter + theme.column_gap
    plot_left = theme.margin_left
    plot_top = theme.margin_top
    plot_bottom = plot_top + theme.plot_height
    width = plot_left + n_columns * col_pitch - theme.column_gap + theme.margin_right

    bar_strip = (theme.hf_band + 30.0) if (any_hf and theme.show_hf_scalebar and m_hf > 0) else 0.0

    # Substructure fans stick out above and below their parent line. With hf_scale='auto'
    # the ground state overflows the nominal band by its ratio to the excited states --
    # 5-8x for most species -- so the energy mapping is inset by the worst overhang rather
    # than letting a fan run off the canvas.
    over_up = over_down = 0.0
    for spec in hf_specs + z_specs:
        shifts = [sub.shift_Hz for sub in spec.level.sublevels()]
        if len(shifts) < 2 or max(shifts) - min(shifts) <= 0:
            n = max(len(shifts), 1)
            reach = (n - 1) / 2 * theme.hf_schematic_pitch
            over_up = max(over_up, reach)
            over_down = max(over_down, reach)
            continue
        local = resolve_level_scale(spec.style.hf_scale, spec.style.hf_exaggerate,
                                    spec.level, theme.hf_band, m_hf)
        over_up = max(over_up, local * max(shifts))
        over_down = max(over_down, -local * min(shifts))
    if z_specs and m_z != m_hf:
        extra = max((zeeman_only_spread(spec.level, b_gauss) for spec in z_specs),
                    default=0.0) * m_z / 2
        over_up += extra
        over_down += extra

    # Grow the canvas around the fans rather than squeezing the energy span into them --
    # shrinking the span would silently compress the gross axis whenever a manifold is
    # exaggerated, which is exactly the sort of quiet misrepresentation this design is
    # trying to avoid. plot_height therefore always means the *energy* span.
    band_top_limit = plot_top + over_up
    usable_height = theme.plot_height
    band_bottom_limit = band_top_limit + usable_height
    plot_bottom = band_bottom_limit + over_down
    height = plot_bottom + theme.margin_bottom + bar_strip

    scene = Scene(width=width, height=height, theme=theme,
                  axis_label=scale.axis_label, title=title,
                  plot_top=plot_top, plot_bottom=plot_bottom,
                  axis_x=plot_left - 10.0)

    if theme.show_axis:
        for frac, text in scale.ticks():
            scene.ticks.append(AxisTick(y=band_bottom_limit - frac * usable_height,
                                        label=text))

    if theme.show_headers:
        for label, index in column_index.items():
            scene.headers.append(ColumnHeader(
                x=plot_left + index * col_pitch + theme.column_width / 2,
                y=plot_top - 18.0, text=label))

    # --- levels ------------------------------------------------------------
    per_column: Dict[int, List[Tuple[float, LevelSpec, float]]] = {}

    for _, label, spec in keyed:
        index = column_index[label]
        style = spec.style
        line_width = (style.width or theme.level_width) * theme.column_width
        cx = plot_left + index * col_pitch + theme.column_width / 2
        y_parent = band_bottom_limit - scale.normalize(spec.level.level_Hz) * usable_height
        x0, x1 = cx - line_width / 2, cx + line_width / 2
        color = style.color or theme.level_color
        term_x = x1 + theme.label_gap + hf_gutter

        if style.detail == 'zeeman':
            local_hf = resolve_level_scale(style.hf_scale, style.hf_exaggerate,
                                           spec.level, theme.hf_band, m_hf)
            local_z = resolve_level_scale(style.z_scale, style.z_exaggerate,
                                          spec.level, theme.z_band, m_z)
            band_top, band_bottom = _draw_zeeman(scene, spec, cx, y_parent, theme,
                                                 local_hf, local_z, b_gauss, color, style)
            scene.anchors[spec.level] = Anchor(x0=cx - theme.column_width / 2,
                                               y0=band_top,
                                               x1=cx + theme.column_width / 2,
                                               y1=band_bottom)
            if style.bracket:
                scene.brackets.append(Bracket(
                    x=cx + theme.column_width / 2 + theme.bracket_gap,
                    y0=band_top, y1=band_bottom, width=theme.bracket_width,
                    color=theme.leader_color, dashed=False))
            term_y = (band_top + band_bottom) / 2
            if style.label:
                per_column.setdefault(index, []).append((term_y, spec, term_x))
            continue

        sublevels = spec.level.sublevels() if style.detail == 'hyperfine' else []
        if style.detail == 'hyperfine' and len(sublevels) >= 2:
            local_hf = resolve_level_scale(style.hf_scale, style.hf_exaggerate,
                                           spec.level, theme.hf_band, m_hf)
            placed = _hyperfine_positions(sublevels, y_parent, local_hf, theme)
            schematic = hyperfine_span(spec.level) <= 0 or local_hf <= 0

            if style.show_parent:
                scene.lines.append(LevelLine(
                    x0=x0, x1=x1, y=y_parent, color=color,
                    linewidth=theme.level_linewidth, opacity=0.22,
                    title=_tooltip(spec.level)))

            for sublevel, y in placed:
                scene.lines.append(LevelLine(
                    x0=x0, x1=x1, y=y, color=color,
                    linewidth=style.linewidth or theme.hf_linewidth,
                    opacity=style.opacity, dashes=style.dashes,
                    title=_tooltip(sublevel)))
                scene.anchors[sublevel] = Anchor(x0=x0, y0=y, x1=x1, y1=y)

            band_top = min(y for _, y in placed)
            band_bottom = max(y for _, y in placed)
            # the manifold anchor spans the whole fan, so a transition drawn to the gross
            # level still lands on something sensible
            scene.anchors[spec.level] = Anchor(x0=x0, y0=band_top, x1=x1, y1=band_bottom)

            if style.bracket:
                scene.brackets.append(Bracket(
                    x=x1 + theme.bracket_gap, y0=band_top, y1=band_bottom,
                    width=theme.bracket_width, color=theme.leader_color,
                    dashed=schematic))
            if style.hf_labels:
                _place_hf_labels(scene, placed, x1, theme, color)
            # drawn off the shared scale? say so, or the figure misleads
            if not schematic and abs(local_hf - m_hf) > 1e-12 * max(local_hf, m_hf, 1.0):
                scene.labels.append(Label(
                    x=x1 + theme.bracket_gap, y=band_top - 4.0,
                    runs=((format_splitting(hyperfine_span(spec.level)), 'normal'),),
                    color=theme.axis_color, size=theme.hf_label_size))
            term_y = (band_top + band_bottom) / 2
        else:
            scene.lines.append(LevelLine(
                x0=x0, x1=x1, y=y_parent, color=color,
                linewidth=style.linewidth or theme.level_linewidth,
                opacity=style.opacity, dashes=style.dashes,
                title=_tooltip(spec.level)))
            scene.anchors[spec.level] = Anchor(x0=x0, y0=y_parent, x1=x1, y1=y_parent)
            term_y = y_parent

        if style.label:
            per_column.setdefault(index, []).append((term_y, spec, term_x))

    # --- term labels, de-overlapped within each column ---------------------
    max_label_right = 0.0
    for entries in per_column.values():
        entries.sort(key=lambda item: item[0])
        adjusted = resolve_overlaps([e[0] for e in entries], theme.min_label_spacing)
        for (y, spec, text_x), new_y in zip(entries, adjusted):
            runs = _label_runs(spec)
            scene.labels.append(Label(
                x=text_x, y=new_y + theme.label_size * 0.35, runs=tuple(runs),
                color=spec.style.color or theme.label_color, size=theme.label_size))
            if abs(new_y - y) > theme.leader_threshold:
                scene.leaders.append(Leader(x0=text_x - hf_gutter * 0.3, y0=y,
                                            x1=text_x - 2.0, y1=new_y,
                                            color=theme.leader_color))
            max_label_right = max(max_label_right,
                                  text_x + estimate_text_width(runs, theme.label_size))

    _place_transitions(scene, transitions, theme)

    if bar_strip:
        bar_bottom = height - theme.margin_bottom
        text = f'hyperfine scale: {format_splitting(theme.hf_band / m_hf)}'
        if z_specs and m_z != m_hf and m_z > 0:
            text += (f'   ·   Zeeman exaggerated '
                     f'{m_z / m_hf:.0f}x ({format_splitting(theme.z_band / m_z)})')
        scene.scalebar = ScaleBar(
            x=plot_left + 6.0, y0=bar_bottom - theme.hf_band, y1=bar_bottom,
            label=text, color=theme.axis_color)

    scene.width = max(scene.width, max_label_right + theme.margin_right)
    return scene


def _place_transitions(scene: Scene, transitions: Sequence['TransitionSpec'],
                       theme: Theme) -> None:
    """Route each transition between the *drawn* boxes of its endpoints.

    Endpoints resolve through :func:`_resolve_anchor`, never through raw energies -- an
    arrow into a magnified hyperfine fan has to land on the sublevel line actually drawn,
    which may be far from where its true energy would put it.
    """
    from .color import get_spectrum, is_visible
    spectrum = get_spectrum(theme.spectrum)

    for spec in transitions:
        lower = _resolve_anchor(scene, spec.lower)
        upper = _resolve_anchor(scene, spec.upper)
        if lower is None or upper is None:
            warnings.warn(
                f'Skipping a transition: {"lower" if lower is None else "upper"} level is '
                f'not drawn in this diagram. Add it with add_level() first.',
                stacklevel=3)
            continue

        style = spec.style
        nm = transition_wavelength_nm(spec.lower, spec.upper)
        visible = is_visible(nm)
        color = style.color or spectrum(nm)
        # 'auto' dashes non-visible lines -- but a wavy line already reads as "different",
        # and dashing a squiggle just looks noisy.
        dashed = ((not visible) and not style.wavy) if style.dash == 'auto' \
            else (style.dash == 'dashed')

        same_column = abs(lower.center[0] - upper.center[0]) < 1e-6
        toward_right = None if same_column else upper.center[0] > lower.center[0]
        x0, y0 = _port(lower, style.port, toward_right, upper.center[1])
        x1, y1 = _port(upper, style.port, None if same_column else not toward_right,
                       lower.center[1])

        control = None
        arc = style.route == 'arc' or (style.route == 'auto' and same_column
                                       and abs(y1 - y0) > 1e-6 and _crowded(scene, x0, y0, y1))
        # unit normal to the chord: an arc must bulge *perpendicular*, since offsetting
        # along +x barely bends a diagonal
        dx, dy = x1 - x0, y1 - y0
        chord = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / chord, dx / chord
        if nx < 0:                       # keep the bulge on a consistent side
            nx, ny = -nx, -ny
        if arc:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            control = (mx + nx * style.curvature * chord,
                       my + ny * style.curvature * chord)

        polygon = None
        if style.block:
            polygon = _block_polygon(x0, y0, x1, y1, lower, style)

        scene.arrows.append(Arrow(
            x0=x0, y0=y0, x1=x1, y1=y1, control=control, polygon=polygon,
            color=color, width=style.width or theme.transition_width,
            dashed=dashed, wavy=style.wavy, head=style.head,
            opacity=style.block_opacity if style.block else style.opacity,
            title=f'{_level_name(spec.lower)} -> {_level_name(spec.upper)}   '
                  f'{nm:.2f} nm' if nm != float('inf') else ''))

        if style.label:
            text = style.label
            if style.label == 'wavelength':
                text = f'{nm:.1f} nm' if nm < 1e4 else f'{nm / 1000:.2f} µm'
            elif style.label == 'frequency':
                text = format_splitting(abs(spec.upper.level_Hz - spec.lower.level_Hz))
            mx = (x0 + x1) / 2 if control is None else (x0 + 2 * control[0] + x1) / 4
            my = (y0 + y1) / 2 if control is None else (y0 + 2 * control[1] + y1) / 4
            offset = theme.transition_label_gap + theme.transition_label_size * 0.3
            scene.labels.append(Label(
                x=mx + nx * offset, y=my + ny * offset + theme.transition_label_size * 0.35,
                runs=((text, 'normal'),), color=color,
                size=theme.transition_label_size))


def _block_polygon(x0: float, y0: float, x1: float, y1: float,
                   source: Anchor, style) -> Tuple[Tuple[float, float], ...]:
    """Outline of a wide "fancy" arrow rising from a level line.

    Two things make this read correctly, both taken from how these are drawn by hand:

    * the shaft is as wide as the **source level line**, so the arrow visibly grows out of
      that level rather than being an abstract connector;
    * the tail is cut **flat** -- horizontally, flush with the level line -- rather than
      square to the arrow's own axis, so it sits on the level instead of hovering at an
      angle to it.

    The head, by contrast, is built square to the axis and sized from the shaft width, not
    from the arrow's length: a fraction-of-length head turns into an enormous wedge on a
    long diagonal, and flat shoulders shear visibly on a steep one.
    """
    width = style.block_width if style.block_width is not None else (source.x1 - source.x0)
    width = max(width, 4.0)

    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux                      # unit normal

    # Flat (horizontal) tail cut, unless the arrow is so shallow that a horizontal cut
    # would degenerate, in which case fall back to a square one.
    if abs(uy) < 0.2:
        base = [(x0 + nx * width / 2, y0 + ny * width / 2),
                (x0 - nx * width / 2, y0 - ny * width / 2)]
    else:
        base = [(x0 - width / 2, y0), (x0 + width / 2, y0)]

    # perpendicular offsets of the two tail corners, so the shaft sides stay parallel
    offsets = [(bx - x0) * nx + (by - y0) * ny for bx, by in base]

    head_len = min(max(style.block_head * width, 7.0), 0.45 * length)
    px, py = x1 - ux * head_len, y1 - uy * head_len
    shoulder = [(px + nx * o, py + ny * o) for o in offsets]
    flare = [(px + nx * o * style.block_flare, py + ny * o * style.block_flare)
             for o in offsets]

    return (base[0], shoulder[0], flare[0], (x1, y1), flare[1], shoulder[1], base[1])


def _crowded(scene: Scene, x: float, y_from: float, y_to: float) -> bool:
    """Would a vertical line at x cross a level line between these two heights?"""
    lo, hi = sorted((y_from, y_to))
    for line in scene.lines:
        if line.x0 - 1 <= x <= line.x1 + 1 and lo + 1 < line.y < hi - 1:
            return True
    return False


def _level_name(level) -> str:
    return getattr(level, 'name', str(level))


def _hyperfine_positions(sublevels: Sequence, y_parent: float, m_hf: float,
                         theme: Theme) -> List[Tuple[object, float]]:
    """Vertical positions for a manifold's hyperfine sublevels.

    Ordering is by *energy*, not by F: a negative hyperfine A inverts the ladder, as in
    137Ba+ 5d 2D5/2 where F runs 4, 3, 2, 1 from the bottom.

    The fan is deliberately not symmetric about ``y_parent`` -- the hyperfine centre of
    gravity is the degeneracy-weighted mean, so shifts are applied exactly as they come.
    """
    shifts = [s.shift_Hz for s in sublevels]
    if max(shifts) - min(shifts) > 0 and m_hf > 0:
        ordered = sorted(sublevels, key=lambda s: s.shift_Hz)
        return [(s, y_parent - m_hf * s.shift_Hz) for s in ordered]
    # No splitting data at all -- the common case, since hyperfine coefficients are only
    # entered for a few levels per species. Fall back to an evenly spaced schematic fan,
    # which the dashed bracket marks as not-to-scale.
    ordered = sorted(sublevels, key=lambda s: (s.term.F if s.term.F is not None else 0))
    n = len(ordered)
    return [(s, y_parent + ((n - 1) / 2 - i) * theme.hf_schematic_pitch)
            for i, s in enumerate(ordered)]


def _draw_zeeman(scene: Scene, spec: LevelSpec, cx: float, y_parent: float,
                 theme: Theme, m_hf: float, m_z: float, b_gauss: float, color: str,
                 style) -> Tuple[float, float]:
    """Draw a manifold's Zeeman sublevels as a horizontal fan of short lines.

    Hyperfine structure stacks vertically; Zeeman structure *fans horizontally* by m_F,
    which is the conventional look and a different packing problem -- it competes for
    column width rather than for vertical room. At B = 0 every shift is zero and the fan
    is purely horizontal.
    """
    hf_sublevels = spec.level.sublevels()
    zlevels = [(hf, z) for hf in hf_sublevels for z in hf.sublevels()]
    if not zlevels:
        return y_parent, y_parent

    max_mF = max(abs(z.term.mF) for _, z in zlevels if z.term.mF is not None) or 0.5
    slots = 2 * max_mF + 1
    pitch = theme.column_width / slots
    half = pitch * theme.z_fill / 2

    ys = []
    for hf, z in zlevels:
        # split the offset so hyperfine and Zeeman can carry different magnifications
        hf_shift = hf.shift_Hz
        z_shift = z.level_hz_at(b_gauss) - spec.level.level_Hz - hf_shift
        y = y_parent - m_hf * hf_shift - m_z * z_shift
        x_mid = cx + (z.term.mF or 0.0) * pitch
        scene.lines.append(LevelLine(
            x0=x_mid - half, x1=x_mid + half, y=y, color=color,
            linewidth=style.linewidth or theme.z_linewidth,
            opacity=style.opacity, title=_tooltip(z)))
        scene.anchors[z] = Anchor(x0=x_mid - half, y0=y, x1=x_mid + half, y1=y)
        ys.append(y)
    return min(ys), max(ys)


def _place_hf_labels(scene: Scene, placed: Sequence[Tuple[object, float]], x1: float,
                     theme: Theme, color: str) -> None:
    """'F = n' labels in the sub-gutter, de-overlapped without moving the lines.

    Lines stay exactly where the physics puts them; only the text is relaxed, with a
    leader when it has to travel. That keeps near-degenerate manifolds honest and still
    readable -- 137Ba+ 5d 2D5/2 has two sublevels 0.5 MHz apart that would otherwise
    print on top of each other.
    """
    x = x1 + theme.bracket_gap + theme.bracket_width + 4.0
    ordered = sorted(placed, key=lambda item: item[1])
    adjusted = resolve_overlaps([y for _, y in ordered], theme.hf_min_label_spacing)
    for (sublevel, y), new_y in zip(ordered, adjusted):
        F = sublevel.term.F
        text = f'{F:g}' if F is not None else '?'
        scene.labels.append(Label(
            x=x, y=new_y + theme.hf_label_size * 0.35,
            runs=(('F', 'normal'), (text, 'sub')),
            color=color, size=theme.hf_label_size))
        if abs(new_y - y) > theme.leader_threshold:
            scene.leaders.append(Leader(x0=x1 + 1.0, y0=y, x1=x - 2.0, y1=new_y,
                                        color=theme.leader_color))


def _label_runs(spec: LevelSpec):
    if spec.style.label_text is not None:
        return [(spec.style.label_text, 'normal')]
    return spec.level.term.markup_runs(include_conf=spec.style.show_conf, level='J')


def _tooltip(level) -> str:
    """Native SVG tooltip text: the full name plus the energy."""
    try:
        return f'{level.name}  —  {level.level_Hz / 2.99792458e10:.2f} cm⁻¹'
    except Exception:      # pragma: no cover - defensive, tooltips are cosmetic
        return str(level)
