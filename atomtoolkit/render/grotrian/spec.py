"""Declarative inputs to a Grotrian diagram: what to draw, and how it should look.

Nothing here computes geometry. A spec is cheap, inspectable and immutable, which is what
lets ``Grotrian.render()`` be pure and repeatable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Tuple

from ... import util

__all__ = ['LevelStyle', 'LevelSpec', 'TransitionStyle', 'TransitionSpec',
           'Theme', 'ColumnPolicy', 'LIGHT', 'DARK']


@dataclass(frozen=True)
class LevelStyle:
    """Per-level appearance. ``None`` means "inherit from the theme"."""

    color: Optional[str] = None
    linewidth: Optional[float] = None
    width: Optional[float] = None        #: line length, as a fraction of the column width
    opacity: float = 1.0
    dashes: Optional[str] = None         #: SVG stroke-dasharray, e.g. '4 2'
    label: bool = True
    label_text: Optional[str] = None     #: override the auto term symbol
    show_conf: bool = False              #: prefix the configuration in the label

    # --- substructure (phase 2) ---
    detail: str = 'gross'                #: 'gross', 'hyperfine' or 'zeeman'
    show_parent: bool = False            #: also draw a faint line at the manifold centroid
    bracket: bool = True                 #: bracket grouping the hyperfine sublevels
    hf_labels: bool = True               #: label each sublevel with its F value
    z_labels: bool = True                #: label m_F sublevels, subject to there being room

    # Per-manifold scale overrides. None means "inherit the diagram-wide scale", which is
    # the default and what keeps manifolds comparable. Override one when you need its
    # internal structure resolved -- 137Ba+ 5d 2D5/2 is ~1 px tall on a global scale.
    # Any manifold drawn off-scale is annotated with its true span, so the figure still
    # tells the reader how big the splitting actually is.
    hf_scale: Optional[object] = None    #: 'fit', px per Hz, or a splitting Quantity
    hf_exaggerate: float = 1.0           #: multiplier on whatever hyperfine scale applies
    z_scale: Optional[object] = None     #: 'fit', px per Hz, or a splitting Quantity
    z_exaggerate: float = 1.0            #: multiplier on whatever Zeeman scale applies

    def merged(self, **overrides) -> 'LevelStyle':
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


@dataclass(frozen=True)
class LevelSpec:
    """One requested level, plus how to draw it."""

    level: object                        #: an EnergyLevel (BaseLevel, kept loose to avoid a cycle)
    style: LevelStyle = field(default_factory=LevelStyle)
    column: Optional[str] = None         #: force a column key, bypassing the policy


@dataclass(frozen=True)
class TransitionStyle:
    """Per-transition appearance. ``None`` means "work it out"."""

    color: Optional[str] = None          #: default: derived from the wavelength
    width: Optional[float] = None
    opacity: float = 1.0
    route: str = 'auto'                  #: 'auto', 'straight' or 'arc'
    curvature: float = 0.16              #: arc bulge, as a fraction of the span
    dash: str = 'auto'                   #: 'auto' dashes non-visible lines, or 'solid'/'dashed'
    wavy: bool = False                   #: squiggle, the convention for spontaneous decay
    head: str = 'end'                    #: 'end', 'both' or 'none'
    label: Optional[str] = None          #: 'wavelength', 'frequency', or literal text
    port: str = 'auto'                   #: 'auto', 'left', 'center' or 'right'

    # --- block ("fancy") arrows ---
    block: bool = False                  #: wide filled arrow instead of a stroked line
    block_width: Optional[float] = None  #: default: the width of the source level line
    block_flare: float = 1.6             #: head width, as a multiple of the shaft width
    block_head: float = 1.15             #: head length, also as a multiple of the shaft
                                         #: width -- a fraction of the arrow's *length*
                                         #: makes a wedge out of any long diagonal
    block_opacity: float = 0.42          #: block arrows read best translucent


@dataclass(frozen=True)
class TransitionSpec:
    """One requested transition, lower level first."""

    lower: object
    upper: object
    style: TransitionStyle = field(default_factory=TransitionStyle)


@dataclass(frozen=True)
class Theme:
    """Global appearance. Sizes are in SVG user units (nominally px)."""

    # typography
    font_family: str = 'Helvetica, Arial, sans-serif'
    label_size: float = 13.0
    axis_size: float = 11.0
    header_size: float = 14.0
    title_size: float = 18.0

    # colours
    background: Optional[str] = '#ffffff'
    level_color: str = '#1a1a1a'
    label_color: str = '#1a1a1a'
    axis_color: str = '#8a8a8a'
    header_color: str = '#4a4a4a'
    leader_color: str = '#b0b0b0'
    grid_color: str = '#ececec'

    # geometry
    level_linewidth: float = 2.2
    level_width: float = 0.86            #: fraction of the line region actually drawn
    column_width: float = 118.0          #: width of the level-line region
    label_gutter: float = 88.0           #: reserved for labels, so they never intrude
                                         #: on the next column (plan §7: fixed gutters
                                         #: sidestep needing real font metrics)
    column_gap: float = 16.0             #: breathing room between columns
    label_gap: float = 7.0               #: line end -> label text
    min_label_spacing: float = 15.0      #: minimum vertical pitch between label baselines
    leader_threshold: float = 3.0        #: draw a leader once a label moves this far

    # --- hyperfine structure ---
    hf_band: float = 46.0                #: px given to the *largest* hyperfine span present
    hf_schematic_pitch: float = 7.0      #: pitch used when a manifold has no splitting data
    hf_linewidth: float = 1.6
    hf_label_size: float = 10.0
    hf_label_gutter: float = 40.0        #: sub-gutter for 'F = n' labels, left of the term
    hf_min_label_spacing: float = 11.0
    bracket_width: float = 5.0
    bracket_gap: float = 5.0
    show_hf_scalebar: bool = True        #: one bar annotating the global hyperfine scale

    # --- transitions ---
    spectrum: str = 'bruton'             #: wavelength -> colour model; see color.SPECTRA
    transition_width: float = 1.9
    transition_head: float = 7.0         #: arrowhead length in px
    transition_label_size: float = 10.5
    transition_label_gap: float = 4.0    #: text offset perpendicular to the arrow
    wavy_amplitude: float = 2.6
    wavy_period: float = 11.0

    # --- Zeeman structure ---
    z_band: float = 30.0                 #: px given to the largest Zeeman spread present
    z_fill: float = 0.74                 #: fraction of each m_F slot actually drawn
    z_linewidth: float = 1.3
    z_label_size: float = 8.5
    z_label_gap: float = 3.0             #: m_F label offset below its line
    min_label_pitch: float = 15.0        #: below this per-slot width, m_F labels are
                                         #: dropped and the row is summarised instead

    # canvas
    margin_top: float = 54.0
    margin_bottom: float = 34.0
    margin_left: float = 66.0
    margin_right: float = 26.0
    plot_height: float = 620.0

    show_axis: bool = True
    show_grid: bool = False
    show_headers: bool = True


LIGHT = Theme()

DARK = Theme(
    background='#15181c',
    level_color='#e8e8e8',
    label_color='#e8e8e8',
    axis_color='#6d757e',
    header_color='#9aa3ad',
    leader_color='#454b52',
    grid_color='#23272d',
)


# ---------------------------------------------------------------------------
# Column policies
# ---------------------------------------------------------------------------

class ColumnPolicy:
    """Assigns levels to columns and orders them left to right.

    A policy maps a level to ``(sort_key, display_label)``. Columns are laid out in
    ascending ``sort_key``.
    """

    def __init__(self, by='J', key: Optional[Callable] = None):
        self.by = by
        self._key = key
        if key is None:
            try:
                self._key = _BUILTIN[by]
            except KeyError:
                raise ValueError(
                    f'Unknown column policy {by!r}. '
                    f'Expected one of {sorted(_BUILTIN)}, or pass a callable.'
                ) from None

    def __call__(self, level) -> Tuple[float, str]:
        return self._key(level)

    def __repr__(self):
        return f'ColumnPolicy({self.by!r})'


def _by_J(level) -> Tuple[float, str]:
    J = level.term.J
    return (J if J is not None else -1.0), f'J = {util.float_to_frac(J)}'


def _by_parity(level) -> Tuple[float, str]:
    even = level.term.parity == 1
    return (0.0 if even else 1.0), ('even' if even else 'odd')


def _by_L(level) -> Tuple[float, str]:
    """Group by orbital angular momentum letter of the term (LS terms only)."""
    term = (level.term.term or '').replace('*', '')
    letter = next((c for c in term if c.isalpha()), '?')
    order = 'SPDFGHIKLMNOQRTUVWXYZ'
    return float(order.find(letter)) if letter in order else 99.0, letter


def _by_config(level) -> Tuple[float, str]:
    conf = level.term.conf or ''
    return 0.0, conf


_BUILTIN = {
    'J': _by_J,
    'parity': _by_parity,
    'L': _by_L,
    'term': _by_L,
    'config': _by_config,
}
