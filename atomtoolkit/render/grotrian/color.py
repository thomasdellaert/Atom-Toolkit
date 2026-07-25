"""Mapping transition wavelengths to colour.

Colouring an arrow by its own wavelength is the one piece of styling a Grotrian diagram
can do that is genuinely physical, so it is worth doing properly. There is no single right
answer, though: a physically-derived spectral fit and a hand-tuned "what this laser
actually looks like on the bench" table disagree, and both are legitimate. So the mapping
is pluggable.

::

    Grotrian(atom, spectrum='rainbow')          # a bundled model
    Grotrian(atom, spectrum=my_map)             # any callable nm -> '#rrggbb'

Bundled models:

``bruton``
    Piecewise-linear fit to the visible spectrum (Dan Bruton's widely used approximation),
    with eye-response roll-off at both ends and a fade to grey outside the visible band.
``rainbow``
    Control points hand-picked from the perceived colour of real laser lines, linearly
    interpolated, fading to black rather than grey outside the visible band. Derived from
    analysis/Rainbow.ipynb.
``rainbow-saturated``
    As above with saturation pushed to full, which keeps deep reds and violets legible as
    thin lines on white.
"""

from __future__ import annotations

import bisect
import colorsys
from typing import Callable, Dict, Sequence, Tuple, Union

__all__ = ['SpectralMap', 'BrutonSpectrum', 'InterpolatedSpectrum',
           'wavelength_to_hex', 'is_visible', 'get_spectrum', 'SPECTRA',
           'VISIBLE_MIN_NM', 'VISIBLE_MAX_NM']

VISIBLE_MIN_NM = 380.0
VISIBLE_MAX_NM = 780.0


def is_visible(wavelength_nm: float) -> bool:
    """Whether a wavelength falls in the visible band."""
    return VISIBLE_MIN_NM <= wavelength_nm <= VISIBLE_MAX_NM


def _hex(rgb: Tuple[float, float, float]) -> str:
    return '#{:02x}{:02x}{:02x}'.format(
        *[int(round(255 * min(max(c, 0.0), 1.0))) for c in rgb])


class SpectralMap:
    """Base class: maps a wavelength in nm to an ``#rrggbb`` string."""

    name = 'spectral-map'

    def rgb(self, wavelength_nm: float) -> Tuple[float, float, float]:
        raise NotImplementedError

    def __call__(self, wavelength_nm: float) -> str:
        return _hex(self.rgb(wavelength_nm))

    def __repr__(self):
        return f'<{type(self).__name__} {self.name!r}>'


class BrutonSpectrum(SpectralMap):
    """Piecewise-linear fit to the visible spectrum, fading to grey outside it.

    Outside 380-780 nm the colour blends toward neutral grey over a few hundred nm rather
    than jumping, so a 370 nm line still reads violet-ish and a far-infrared one is
    plainly grey.
    """

    name = 'bruton'
    _FADE_NM = 260.0
    _GREY = (0.42, 0.42, 0.45)

    def __init__(self, darken: float = 0.82):
        self.darken = darken

    @staticmethod
    def _spectral(nm: float) -> Tuple[float, float, float]:
        if nm < 440:
            r, g, b = -(nm - 440) / 60.0, 0.0, 1.0
        elif nm < 490:
            r, g, b = 0.0, (nm - 440) / 50.0, 1.0
        elif nm < 510:
            r, g, b = 0.0, 1.0, -(nm - 510) / 20.0
        elif nm < 580:
            r, g, b = (nm - 510) / 70.0, 1.0, 0.0
        elif nm < 645:
            r, g, b = 1.0, -(nm - 645) / 65.0, 0.0
        else:
            r, g, b = 1.0, 0.0, 0.0
        if nm < 420:
            factor = 0.3 + 0.7 * (nm - 380) / 40.0
        elif nm > 700:
            factor = 0.3 + 0.7 * (780 - nm) / 80.0
        else:
            factor = 1.0
        return r * factor, g * factor, b * factor

    def rgb(self, wavelength_nm: float) -> Tuple[float, float, float]:
        if wavelength_nm <= 0:
            return (0.42, 0.42, 0.44)
        clamped = min(max(wavelength_nm, VISIBLE_MIN_NM), VISIBLE_MAX_NM)
        r, g, b = self._spectral(clamped)
        if not is_visible(wavelength_nm):
            distance = (VISIBLE_MIN_NM - wavelength_nm if wavelength_nm < VISIBLE_MIN_NM
                        else wavelength_nm - VISIBLE_MAX_NM)
            blend = min(distance / self._FADE_NM, 1.0)
            r, g, b = (channel * (1 - blend) + grey * blend
                       for channel, grey in zip((r, g, b), self._GREY))
        return r * self.darken, g * self.darken, b * self.darken


class InterpolatedSpectrum(SpectralMap):
    """Linear interpolation between hand-picked ``(wavelength, r, g, b)`` control points.

    This is the shape of model you want when the colours should match what a set of real
    laser lines *look like*, rather than what a spectral fit says they should be. Add or
    move control points to taste; values outside the tabulated range are clamped to the
    endpoints, so tabulating 0 and some far-infrared wavelength as black makes
    non-visible transitions fade out naturally.

    :param wavelengths: control wavelengths in nm, ascending
    :param reds: red channel at each control point, 0-255
    :param greens: green channel, 0-255
    :param blues: blue channel, 0-255
    :param saturate: push saturation to full, which keeps deep reds and violets legible
        as thin strokes on white
    :param name: identifier, used in repr and the registry
    """

    def __init__(self, wavelengths: Sequence[float],
                 reds: Sequence[float], greens: Sequence[float], blues: Sequence[float],
                 saturate: bool = False, name: str = 'interpolated'):
        if not (len(wavelengths) == len(reds) == len(greens) == len(blues)):
            raise ValueError('wavelengths, reds, greens and blues must be the same length')
        if len(wavelengths) < 2:
            raise ValueError('need at least two control points')
        if list(wavelengths) != sorted(wavelengths):
            raise ValueError('control wavelengths must be ascending')
        self.wavelengths = list(map(float, wavelengths))
        self.channels = [list(map(float, reds)), list(map(float, greens)),
                         list(map(float, blues))]
        self.saturate = saturate
        self.name = name

    def rgb(self, wavelength_nm: float) -> Tuple[float, float, float]:
        xs = self.wavelengths
        nm = min(max(wavelength_nm, xs[0]), xs[-1])
        i = min(max(bisect.bisect_right(xs, nm) - 1, 0), len(xs) - 2)
        span = xs[i + 1] - xs[i]
        t = 0.0 if span == 0 else (nm - xs[i]) / span
        out = tuple((channel[i] + t * (channel[i + 1] - channel[i])) / 255.0
                    for channel in self.channels)
        if self.saturate:
            h, _, v = colorsys.rgb_to_hsv(*out)
            out = colorsys.hsv_to_rgb(h, 1.0, v)
        return out


#: Control points measured by eye against real laser lines; see analysis/Rainbow.ipynb.
_RAINBOW_NM = [0, 200, 280, 300, 350, 400, 445, 475, 493, 510, 570, 650, 780, 1000, 1500]
_RAINBOW_R = [0, 0, 10, 50, 120, 80, 110, 0, 40, 40, 255, 235, 90, 50, 0]
_RAINBOW_G = [0, 0, 0, 0, 120, 40, 0, 0, 255, 255, 230, 30, 30, 20, 0]
_RAINBOW_B = [0, 10, 30, 100, 255, 120, 255, 255, 250, 0, 0, 30, 30, 20, 0]

SPECTRA: Dict[str, SpectralMap] = {
    'bruton': BrutonSpectrum(),
    'rainbow': InterpolatedSpectrum(_RAINBOW_NM, _RAINBOW_R, _RAINBOW_G, _RAINBOW_B,
                                    name='rainbow'),
    'rainbow-saturated': InterpolatedSpectrum(_RAINBOW_NM, _RAINBOW_R, _RAINBOW_G,
                                              _RAINBOW_B, saturate=True,
                                              name='rainbow-saturated'),
}

_DEFAULT = SPECTRA['bruton']


def get_spectrum(spectrum: Union[str, SpectralMap, Callable, None]) -> Callable[[float], str]:
    """Resolve a spectrum argument to a callable ``nm -> '#rrggbb'``.

    Accepts a registry name, a :class:`SpectralMap`, any callable, or ``None`` for the
    default.
    """
    if spectrum is None:
        return _DEFAULT
    if isinstance(spectrum, str):
        try:
            return SPECTRA[spectrum]
        except KeyError:
            raise ValueError(
                f'Unknown spectrum {spectrum!r}. Available: {sorted(SPECTRA)}, '
                f'or pass a SpectralMap or any callable nm -> "#rrggbb".') from None
    if callable(spectrum):
        return spectrum
    raise TypeError(f'spectrum must be a name, a SpectralMap or a callable, '
                    f'not {type(spectrum).__name__}')


def wavelength_to_hex(wavelength_nm: float, darken: float = 0.82) -> str:
    """Colour for a wavelength under the default (Bruton) model.

    Kept as a module-level convenience; prefer :func:`get_spectrum` when the model should
    be selectable.
    """
    return _hex(BrutonSpectrum(darken=darken).rgb(wavelength_nm))
