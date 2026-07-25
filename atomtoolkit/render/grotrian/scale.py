"""Energy axis transforms for Grotrian diagrams.

A scale maps an absolute energy in Hz onto the unit interval, where 0 is the bottom of
the plotting area and 1 the top. Converting that to pixels is the layout engine's job, so
scales stay independent of canvas size and are trivial to test.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from ... import Q_

__all__ = ['EnergyScale', 'LinearScale', 'CompressedScale', 'BrokenScale', 'UNITS']

#: Display units, as (pint unit string, axis label). Energies are stored in Hz.
UNITS = {
    'cm-1': ('cm**-1', 'Energy (cm⁻¹)'),
    'THz': ('THz', 'Frequency (THz)'),
    'eV': ('eV', 'Energy (eV)'),
    'Hz': ('Hz', 'Frequency (Hz)'),
}


def _to_display(energy_hz: float, unit: str) -> float:
    """Convert an energy in Hz to the display unit."""
    pint_unit = UNITS[unit][0]
    return float(Q_(energy_hz, 'Hz').to(pint_unit, 'spectroscopy').magnitude)


def _nice_step(rough: float) -> float:
    """Round a step size up to the nearest 1, 2, 2.5 or 5 times a power of ten."""
    if rough <= 0:
        return 1.0
    exponent = math.floor(math.log10(rough))
    base = rough / 10 ** exponent
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if base <= candidate:
            return candidate * 10 ** exponent
    return 10 ** (exponent + 1)


@dataclass
class EnergyScale:
    """Base class: a monotonic map from energy (Hz) to the unit interval."""

    unit: str = 'cm-1'
    lo_hz: float = 0.0
    hi_hz: float = 1.0
    pad: float = 0.05

    def prepare(self, energies_hz: Sequence[float]) -> 'EnergyScale':
        """Fit the scale to the energies that will be drawn. Returns self."""
        if not energies_hz:
            raise ValueError('Cannot build an energy scale with no levels.')
        lo, hi = min(energies_hz), max(energies_hz)
        if hi == lo:                       # a single level, or several degenerate ones
            hi = lo + 1.0
        span = hi - lo
        self.lo_hz, self.hi_hz = lo - span * self.pad, hi + span * self.pad
        return self

    def normalize(self, energy_hz: float) -> float:
        raise NotImplementedError

    @property
    def axis_label(self) -> str:
        return UNITS[self.unit][1]

    def ticks(self, count: int = 6) -> List[Tuple[float, str]]:
        """Tick positions as ``(normalized, label)``, in display units."""
        lo, hi = _to_display(self.lo_hz, self.unit), _to_display(self.hi_hz, self.unit)
        if hi < lo:                        # wavelength-like units invert the ordering
            lo, hi = hi, lo
        step = _nice_step((hi - lo) / max(count, 1))
        first = math.ceil(lo / step) * step
        out = []
        value = first
        while value <= hi + step * 1e-9:
            # invert the display conversion by interpolating on the Hz axis
            frac = (value - lo) / (hi - lo) if hi != lo else 0.0
            energy = self.lo_hz + frac * (self.hi_hz - self.lo_hz)
            label = f'{value:g}'
            out.append((self.normalize(energy), label))
            value += step
        return [(f, s) for f, s in out if -0.001 <= f <= 1.001]


@dataclass
class LinearScale(EnergyScale):
    """Linear in energy -- the default, and the literal reading of the axis."""

    def normalize(self, energy_hz: float) -> float:
        if self.hi_hz == self.lo_hz:
            return 0.5
        return (energy_hz - self.lo_hz) / (self.hi_hz - self.lo_hz)


@dataclass
class CompressedScale(EnergyScale):
    """Piecewise-linear scale that allocates space by level *density*.

    Grotrian diagrams in the literature usually do this: a large empty gap between the
    ground state and the first excited state is squeezed so that crowded regions higher up
    get room to breathe. ``strength`` blends between linear (0.0) and pure
    rank-ordering (1.0).

    The transform stays monotonic, so ordering is always preserved, but vertical distances
    are no longer proportional to energy differences -- which is why it is not the default.
    """

    strength: float = 0.7
    _knots_hz: List[float] = field(default_factory=list)
    _knots_frac: List[float] = field(default_factory=list)

    def prepare(self, energies_hz: Sequence[float]) -> 'CompressedScale':
        super().prepare(energies_hz)
        unique = sorted(set(energies_hz))
        if len(unique) < 2:
            self._knots_hz, self._knots_frac = [self.lo_hz, self.hi_hz], [0.0, 1.0]
            return self
        knots = [self.lo_hz] + unique + [self.hi_hz]
        span = self.hi_hz - self.lo_hz
        fracs = []
        for i, energy in enumerate(knots):
            linear = (energy - self.lo_hz) / span
            rank = i / (len(knots) - 1)
            fracs.append((1 - self.strength) * linear + self.strength * rank)
        # guard monotonicity against float noise
        for i in range(1, len(fracs)):
            fracs[i] = max(fracs[i], fracs[i - 1] + 1e-12)
        self._knots_hz, self._knots_frac = knots, fracs
        return self

    def normalize(self, energy_hz: float) -> float:
        if not self._knots_hz:
            return 0.5
        knots, fracs = self._knots_hz, self._knots_frac
        if energy_hz <= knots[0]:
            return fracs[0]
        if energy_hz >= knots[-1]:
            return fracs[-1]
        i = bisect.bisect_right(knots, energy_hz) - 1
        i = min(i, len(knots) - 2)
        lo, hi = knots[i], knots[i + 1]
        t = 0.0 if hi == lo else (energy_hz - lo) / (hi - lo)
        return fracs[i] + t * (fracs[i + 1] - fracs[i])


@dataclass
class BrokenScale(EnergyScale):
    """Linear, except that named energy ranges are collapsed to a thin gap.

    The discrete cousin of :class:`CompressedScale`. Where that one smoothly reallocates
    space by level density, this one keeps every retained region *exactly* linear and
    simply cuts out the stretches you nominate -- which is the convention in published
    figures, and easier to read a value off, at the cost of having to choose the breaks.

    ::

        BrokenScale(breaks=[(Q_(5, 'THz'), Q_(600, 'THz'))])
        BrokenScale.spanning(energies, n_breaks=1)      # pick the emptiest stretch

    :param breaks: ``(low, high)`` pairs in Hz, or pint Quantities, to collapse
    :param gap: fraction of the axis each collapsed range keeps
    """

    breaks: List[Tuple[float, float]] = field(default_factory=list)
    gap: float = 0.035
    _knots_hz: List[float] = field(default_factory=list)
    _knots_frac: List[float] = field(default_factory=list)

    @classmethod
    def spanning(cls, energies_hz: Sequence[float], n_breaks: int = 1,
                 min_fraction: float = 0.15, anomaly: float = 3.0,
                 **kwargs) -> 'BrokenScale':
        """Build a scale that breaks the ``n_breaks`` emptiest stretches.

        A gap qualifies only if it is both a decent fraction of the whole range *and*
        anomalous compared with the typical gap. The second condition is what stops evenly
        spread levels being broken: with N levels spaced evenly, every gap is 1/(N-1) of
        the range -- 25% for five levels -- so a fraction test alone would always fire.

        :param n_breaks: at most this many breaks
        :param min_fraction: a gap must exceed this fraction of the full range
        :param anomaly: ...and this multiple of the median gap
        """
        ordered = sorted(set(energies_hz))
        if len(ordered) < 3:
            return cls(breaks=[], **kwargs)
        total = ordered[-1] - ordered[0]
        widths = [b - a for a, b in zip(ordered, ordered[1:])]
        median = sorted(widths)[len(widths) // 2]
        threshold = max(min_fraction * total, anomaly * median)
        gaps = sorted(((b - a, a, b) for a, b in zip(ordered, ordered[1:])), reverse=True)
        chosen = [(a, b) for width, a, b in gaps[:n_breaks] if width > threshold]
        return cls(breaks=sorted(chosen), **kwargs)

    def _as_hz(self, value) -> float:
        return float(value.to('Hz', 'spectroscopy').magnitude) if hasattr(value, 'to') \
            else float(value)

    def prepare(self, energies_hz: Sequence[float]) -> 'BrokenScale':
        super().prepare(energies_hz)
        spans = []
        for low, high in self.breaks:
            low, high = self._as_hz(low), self._as_hz(high)
            low, high = max(min(low, high), self.lo_hz), min(max(low, high), self.hi_hz)
            if high > low:
                spans.append((low, high))
        spans.sort()

        # walk the axis, giving retained regions space proportional to energy and each
        # collapsed region a fixed sliver
        edges, kept = [self.lo_hz], 0.0
        for low, high in spans:
            edges += [low, high]
            kept += low - edges[-3] if len(edges) > 3 else low - self.lo_hz
        edges.append(self.hi_hz)
        retained = sum(edges[i + 1] - edges[i] for i in range(0, len(edges) - 1, 2))
        budget = 1.0 - self.gap * len(spans)
        if retained <= 0 or budget <= 0:
            self._knots_hz, self._knots_frac = [self.lo_hz, self.hi_hz], [0.0, 1.0]
            return self

        knots_hz, knots_frac, frac = [edges[0]], [0.0], 0.0
        for i in range(len(edges) - 1):
            width = edges[i + 1] - edges[i]
            frac += (width / retained) * budget if i % 2 == 0 else self.gap
            knots_hz.append(edges[i + 1])
            knots_frac.append(frac)
        self._knots_hz, self._knots_frac = knots_hz, knots_frac
        return self

    @property
    def break_positions(self) -> List[Tuple[float, float]]:
        """Normalized ``(start, end)`` of each collapsed region, for drawing markers."""
        out = []
        for i in range(1, len(self._knots_hz) - 1, 2):
            out.append((self._knots_frac[i], self._knots_frac[i + 1]))
        return out

    def normalize(self, energy_hz: float) -> float:
        if not self._knots_hz:
            return 0.5
        knots, fracs = self._knots_hz, self._knots_frac
        if energy_hz <= knots[0]:
            return fracs[0]
        if energy_hz >= knots[-1]:
            return fracs[-1]
        i = min(bisect.bisect_right(knots, energy_hz) - 1, len(knots) - 2)
        span = knots[i + 1] - knots[i]
        t = 0.0 if span == 0 else (energy_hz - knots[i]) / span
        return fracs[i] + t * (fracs[i + 1] - fracs[i])
