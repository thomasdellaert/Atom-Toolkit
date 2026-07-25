"""
SVG Grotrian diagrams.

A Grotrian diagram is a *diagram*, not a plot, so this renders SVG directly rather than
going through matplotlib. See ``docs/grotrian-svg-plan.md`` for the design and its
rationale.

Quick start::

    from atomtoolkit.render.grotrian import Grotrian
    import atomtoolkit

    yb = atomtoolkit.load_species('171Yb II')
    Grotrian.auto(yb, n_levels=14).save('yb171.svg')

or build one up explicitly::

    g = (Grotrian(yb, title='¹⁷¹Yb⁺')
         .add_levels(yb.levels[:14])
         .style_levels(lambda l: l.term.parity == -1, color='#b3541e'))
    g          # renders inline in Jupyter
    g.save('yb171.svg')

The pipeline is ``spec -> layout -> render``: :class:`Grotrian` collects immutable
:class:`~.spec.LevelSpec` objects, :func:`~.layout.layout` positions them into a
:class:`~.layout.Scene`, and :func:`~.svg.render_svg` emits the document. ``render()`` is
pure, so calling it repeatedly (as a notebook does) is always safe.

**Phase 1** draws gross structure: level lines, term-symbol labels, columns and an energy
axis. Hyperfine/Zeeman insets and transitions are phases 2 and 3.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import replace
from typing import Callable, Iterable, List, Optional, Sequence, Union

from .layout import Anchor, Scene, layout, resolve_overlaps
from .scale import BrokenScale, CompressedScale, EnergyScale, LinearScale
from .color import (SPECTRA, BrutonSpectrum, InterpolatedSpectrum, SpectralMap,
                    get_spectrum, is_visible, wavelength_to_hex)
from .spec import (DARK, LIGHT, ColumnPolicy, LevelSpec, LevelStyle, Theme,
                   TransitionSpec, TransitionStyle)
from .svg import render_svg

__all__ = ['Grotrian', 'Theme', 'LevelStyle', 'TransitionStyle', 'ColumnPolicy',
           'wavelength_to_hex', 'is_visible', 'get_spectrum', 'SPECTRA',
           'SpectralMap', 'BrutonSpectrum', 'InterpolatedSpectrum',
           'LinearScale', 'CompressedScale', 'BrokenScale', 'EnergyScale',
           'Scene', 'Anchor', 'layout', 'render_svg', 'resolve_overlaps',
           'LIGHT', 'DARK']


class Grotrian:
    """Builder for an SVG Grotrian diagram.

    Methods that configure the diagram return ``self``, so calls chain. Nothing is
    computed until :meth:`render`.
    """

    def __init__(self,
                 atom=None,
                 theme: Theme = LIGHT,
                 columns: Union[str, ColumnPolicy, Callable] = 'J',
                 scale: Optional[EnergyScale] = None,
                 title: str = '',
                 hf_scale: Union[str, float] = 'auto',
                 B=None,
                 z_scale: Union[str, float] = 'match',
                 spectrum=None):
        """
        :param atom: the Atom these levels belong to. Optional, but lets
            :meth:`add_levels` accept level *names*.
        :param theme: a :class:`~.spec.Theme`; :data:`~.spec.LIGHT` or :data:`~.spec.DARK`
        :param columns: ``'J'``, ``'parity'``, ``'L'``, ``'config'``, a
            :class:`~.spec.ColumnPolicy`, or a callable ``level -> (sort_key, label)``
        :param scale: energy transform; defaults to :class:`~.scale.LinearScale` in cm⁻¹
        :param title: drawn above the diagram
        :param spectrum: wavelength-to-colour model for transitions -- a name from
            :data:`~.color.SPECTRA` (``'bruton'``, ``'rainbow'``, ``'rainbow-saturated'``),
            a :class:`~.color.SpectralMap`, or any callable ``nm -> '#rrggbb'``. Overrides
            ``theme.spectrum``.
        :param z_scale: ``'match'`` (default) keeps Zeeman on the hyperfine scale;
            ``'auto'`` exaggerates it so the largest Zeeman spread fills ``theme.z_band``.
            Also accepts px/Hz or a splitting ``Quantity``. Exaggeration is usually needed
            to see Zeeman structure at all, since at ordinary fields it is orders of
            magnitude smaller than the hyperfine splitting.
        :param hf_scale: hyperfine magnification. ``'auto'`` (default) sizes to the largest
            *excited-state* splitting, because ground-state hyperfine runs 5-8x larger than
            anything else and would otherwise squash every other manifold; ``'widest'``
            sizes to the largest splitting present. Also accepts px/Hz or a ``Quantity``. One
            value is shared by the whole diagram, so hyperfine splittings stay comparable
            between manifolds and a near-degeneracy cannot blow the scale up.
        """
        self.atom = atom
        self.theme = theme
        self.title = title
        self.scale = scale or LinearScale()
        self.hf_scale = hf_scale
        self.z_scale = z_scale
        if spectrum is not None:
            get_spectrum(spectrum)          # fail fast on a bad name
            self.theme = replace(self.theme, spectrum=spectrum)
        # B may be a pint Quantity or a bare number of gauss
        self.b_gauss = 0.0 if B is None else float(getattr(B, 'to', lambda _: B)('gauss').magnitude
                                                   if hasattr(B, 'to') else B)
        self.columns = (columns if isinstance(columns, ColumnPolicy)
                        else ColumnPolicy(by=columns) if isinstance(columns, str)
                        else ColumnPolicy(by='custom', key=columns))
        self._specs: List[LevelSpec] = []
        self._transitions: List[TransitionSpec] = []

    # -- building ----------------------------------------------------------

    def add_level(self, level, **style) -> 'Grotrian':
        """Add one level.

        :param level: an ``EnergyLevel``, or a level name if this Grotrian has an atom
        :param style: any :class:`~.spec.LevelStyle` field -- ``color``, ``linewidth``,
            ``width``, ``opacity``, ``dashes``, ``label``, ``label_text``, ``show_conf``
        """
        column = style.pop('column', None)
        self._specs.append(LevelSpec(level=self._resolve(level),
                                     style=LevelStyle(**style),
                                     column=column))
        return self

    def add_levels(self, levels: Iterable, **style) -> 'Grotrian':
        """Add several levels with a shared style."""
        for level in levels:
            self.add_level(level, **style)
        return self

    def add_transition(self, transition, **style) -> 'Grotrian':
        """Draw a transition between two levels.

        :param transition: a ``BaseTransition``, or a ``(lower, upper)`` pair of levels
            or level names. Order is normalised, so either way round works.
        :param style: any :class:`~.spec.TransitionStyle` field -- ``color`` (defaults to
            the wavelength's own colour), ``width``, ``route`` (``'auto'``/``'straight'``/
            ``'arc'``), ``wavy`` (the convention for spontaneous decay), ``dash``,
            ``head``, ``label`` (``'wavelength'``, ``'frequency'`` or literal text),
            ``port``

        ::

            g.add_transition(('4f14.6s 2S1/2', '4f14.6p 2P*1/2'), label='wavelength')
            g.add_transition(atom.transitions['cooling'], wavy=True, head='none')
        """
        if hasattr(transition, 'E_lower') and hasattr(transition, 'E_upper'):
            lower, upper = transition.E_lower, transition.E_upper
        else:
            lower, upper = transition
        lower, upper = self._resolve(lower), self._resolve(upper)
        if upper.level_Hz < lower.level_Hz:
            lower, upper = upper, lower
        self._transitions.append(
            TransitionSpec(lower=lower, upper=upper, style=TransitionStyle(**style)))
        return self

    def style_levels(self, predicate: Callable, **style) -> 'Grotrian':
        """Restyle every already-added level for which ``predicate(level)`` is true.

        ::

            g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
        """
        self._specs = [
            LevelSpec(level=spec.level,
                      style=spec.style.merged(**style),
                      column=spec.column)
            if predicate(spec.level) else spec
            for spec in self._specs
        ]
        return self

    def _resolve(self, level):
        if isinstance(level, str):
            if self.atom is None:
                raise ValueError(
                    f'Cannot look up level {level!r}: this Grotrian has no atom. '
                    f'Pass Grotrian(atom) or an EnergyLevel object.')
            resolved = self.atom.levels[level]
            if resolved is None:
                raise KeyError(f'No level named {level!r} in {self.atom.name}.')
            return resolved
        return level

    # -- output ------------------------------------------------------------

    @property
    def levels(self) -> List:
        """The levels currently in the diagram, in insertion order."""
        return [spec.level for spec in self._specs]

    def build(self) -> Scene:
        """Run layout and return the positioned :class:`~.layout.Scene`.

        Useful for inspecting geometry, or for feeding an alternative renderer.
        """
        return layout(self._specs, self.theme, self.columns, self._transitions,
                      self.scale, self.title,
                      hf_scale=self.hf_scale, b_gauss=self.b_gauss,
                      z_scale=self.z_scale)

    def render(self, responsive: bool = False) -> str:
        """Render to an SVG document string. Pure -- safe to call repeatedly.

        :param responsive: scale to the container width rather than a fixed pixel size.
            :meth:`_repr_svg_` uses this so wide diagrams fit a notebook cell.
        """
        return render_svg(self.build(), responsive=responsive)

    def save(self, path: Union[str, pathlib.Path]) -> pathlib.Path:
        """Write the SVG to ``path`` and return it."""
        path = pathlib.Path(path)
        path.write_text(self.render(), encoding='utf-8')
        return path

    def _repr_svg_(self) -> str:
        """Display inline in Jupyter with no boilerplate.

        Rendered responsively, so a diagram wider than the notebook cell scales down
        instead of overflowing it. Tooltips (the SVG ``<title>`` on every level and
        transition) work in the notebook too.
        """
        return self.render(responsive=True)

    def __repr__(self) -> str:
        return f'<Grotrian: {len(self._specs)} levels, columns={self.columns.by!r}>'

    # -- convenience -------------------------------------------------------

    @classmethod
    def auto(cls, atom, n_levels: int = 12, aspect: float = 1.6, **kwargs) -> 'Grotrian':
        """Build a reasonable diagram with no tuning.

        This is the honest test of the defaults: if ``auto`` needs adjusting to look
        right, the defaults are wrong.

        Levels are taken in energy order, coloured by parity so the E1-connected ladder
        reads at a glance, and titled with the atom's name.

        Unless ``columns`` is given, the column policy is **chosen** rather than assumed.
        Grouping by J is the conventional default but degenerates on atoms with high-J
        levels -- 171Yb+ over 16 levels spreads across 7 columns and 1630 px, several
        holding a single level, where grouping by parity needs 520 px. Each candidate
        policy is laid out and the one landing closest to ``aspect`` wins, with a
        tie-break toward J so the familiar layout is preferred when it is not much worse.

        :param atom: the atom to draw
        :param n_levels: how many of the lowest levels to include
        :param aspect: preferred width-to-height ratio
        """
        kwargs.setdefault('title', atom.name)
        levels = sorted(atom.levels.values(), key=lambda l: l.level_Hz)[:n_levels]

        def build(policy):
            grotrian = cls(atom, columns=policy, **kwargs)
            grotrian.add_levels(levels)
            grotrian.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
            return grotrian

        if 'columns' in kwargs:
            return build(kwargs.pop('columns'))

        best, best_cost = None, None
        for rank, policy in enumerate(('J', 'L', 'parity')):
            candidate = build(policy)
            scene = candidate.build()
            ratio = scene.width / max(scene.height, 1.0)
            # log-distance from the target aspect, nudged by preference order
            cost = abs(math.log(max(ratio, 1e-6) / aspect)) + 0.12 * rank
            if best_cost is None or cost < best_cost:
                best, best_cost = candidate, cost
        return best
