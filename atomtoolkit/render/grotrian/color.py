"""Mapping transition wavelengths to colour.

Colouring an arrow by its own wavelength is the one piece of styling a Grotrian diagram
can do that is genuinely physical, so it is worth doing properly rather than cycling a
palette.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ['wavelength_to_hex', 'is_visible', 'VISIBLE_MIN_NM', 'VISIBLE_MAX_NM']

VISIBLE_MIN_NM = 380.0
VISIBLE_MAX_NM = 780.0

#: How far past the visible range a colour keeps fading before it is fully desaturated.
_FADE_NM = 260.0
#: Desaturated endpoint for wavelengths well outside the visible range.
_GREY = (0.42, 0.42, 0.45)


def is_visible(wavelength_nm: float) -> bool:
    """Whether a wavelength falls in the visible band."""
    return VISIBLE_MIN_NM <= wavelength_nm <= VISIBLE_MAX_NM


def _spectral_rgb(nm: float) -> Tuple[float, float, float]:
    """Approximate sRGB for a visible wavelength (Bruton's piecewise fit)."""
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

    # the eye's response rolls off at both ends of the visible band
    if nm < 420:
        factor = 0.3 + 0.7 * (nm - 380) / 40.0
    elif nm > 700:
        factor = 0.3 + 0.7 * (780 - nm) / 80.0
    else:
        factor = 1.0
    return r * factor, g * factor, b * factor


def wavelength_to_hex(wavelength_nm: float, darken: float = 0.82) -> str:
    """Colour for a transition of the given wavelength.

    Inside the visible band this is the spectral colour. Outside it, the colour fades
    toward neutral grey over a few hundred nm rather than jumping, so that a 370 nm UV
    line still reads as violet-ish and a 1 µm line as dull red, while a far-infrared line
    is plainly grey. Callers should also dash non-visible transitions -- colour alone is
    not a reliable signal.

    :param wavelength_nm: transition wavelength in nanometres
    :param darken: multiplier applied so lines read against a white background
    :return: an ``#rrggbb`` string
    """
    if wavelength_nm <= 0:
        return '#6b6b6f'
    clamped = min(max(wavelength_nm, VISIBLE_MIN_NM), VISIBLE_MAX_NM)
    r, g, b = _spectral_rgb(clamped)

    if not is_visible(wavelength_nm):
        distance = (VISIBLE_MIN_NM - wavelength_nm if wavelength_nm < VISIBLE_MIN_NM
                    else wavelength_nm - VISIBLE_MAX_NM)
        blend = min(distance / _FADE_NM, 1.0)
        r = r * (1 - blend) + _GREY[0] * blend
        g = g * (1 - blend) + _GREY[1] * blend
        b = b * (1 - blend) + _GREY[2] * blend

    values = [int(round(255 * min(max(channel * darken, 0.0), 1.0))) for channel in (r, g, b)]
    return '#{:02x}{:02x}{:02x}'.format(*values)
