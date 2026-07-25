#!/usr/bin/env python3
"""
Render a gallery of Grotrian diagrams to ``examples/output/``.

This is the replacement for the old ``grotrians.ipynb`` scratchpad: a general visual test
that exercises the renderer across couplings, column policies, energy scales and themes.
Run it after touching anything under ``atomtoolkit/render/grotrian/`` and look at the
results.

    python examples/grotrian_gallery.py

Every diagram here is produced with the public API and no magic numbers.
"""

import pathlib

import atomtoolkit
from atomtoolkit import Q_
from atomtoolkit.render.grotrian import (DARK, CompressedScale, Grotrian, LinearScale,
                                         Theme)

OUT = pathlib.Path(__file__).parent / 'output'
OUT.mkdir(exist_ok=True)


def flagship() -> Grotrian:
    """¹⁷¹Yb⁺ — the reference diagram, and the one to eyeball first.

    Exercises LS and JK term symbols in one figure, both parities, and the compressed
    energy scale that keeps the crowded 20000–35000 cm⁻¹ region legible without
    misrepresenting the ordering.
    """
    yb = atomtoolkit.load_species('171Yb II')
    levels = sorted(yb.levels.values(), key=lambda l: l.level_Hz)[:16]

    g = Grotrian(yb, title='¹⁷¹Yb⁺  ·  low-lying structure',
                 scale=CompressedScale(strength=0.72))
    g.add_levels(levels)
    # odd-parity levels in warm orange: the E1-connected ladder then reads at a glance
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    # the qubit and shelving states, called out
    g.style_levels(lambda l: l.name == '4f14.6s 2S1/2', color='#1b6ca8', linewidth=3.6)
    g.style_levels(lambda l: l.name == '4f14.5d 2D3/2', color='#1b6ca8', linewidth=3.0)
    g.style_levels(lambda l: l.name == '4f13.(2F*).6s2 2F*7/2', color='#1b6ca8',
                   linewidth=3.0)
    return g


def calcium() -> Grotrian:
    """⁴³Ca⁺ on a literal linear axis — the honest view, dead space and all."""
    ca = atomtoolkit.load_species('43Ca II')
    levels = sorted(ca.levels.values(), key=lambda l: l.level_Hz)[:9]
    g = Grotrian(ca, title='⁴³Ca⁺  ·  linear energy axis', scale=LinearScale())
    g.add_levels(levels)
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    return g


def strontium_by_parity() -> Grotrian:
    """⁸⁷Sr — columns by parity instead of J, which suits the two-electron structure."""
    sr = atomtoolkit.load_species('87Sr I')
    levels = sorted(sr.levels.values(), key=lambda l: l.level_Hz)[:14]
    g = Grotrian(sr, title='⁸⁷Sr  ·  grouped by parity',
                 columns='parity', scale=CompressedScale())
    g.add_levels(levels)
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    return g


def neon_dark() -> Grotrian:
    """²⁰Ne* on the dark theme — JK-coupled terms, ²[3/2]° and friends."""
    ne = atomtoolkit.load_species('20Ne I')
    levels = sorted(ne.levels.values(), key=lambda l: l.level_Hz)[:12]
    g = Grotrian(ne, title='²⁰Ne*  ·  JK coupling, dark theme',
                 theme=DARK, scale=CompressedScale())
    g.add_levels(levels)
    g.style_levels(lambda l: l.term.parity == -1, color='#e08a4b')
    return g


def holmium() -> Grotrian:
    """¹⁶⁵Ho — the pathological case: JJ-coupled terms and half-integer J up to 21/2."""
    ho = atomtoolkit.load_species('165Ho I')
    levels = sorted(ho.levels.values(), key=lambda l: l.level_Hz)[:12]
    g = Grotrian(ho, title='¹⁶⁵Ho  ·  JJ coupling stress test',
                 scale=CompressedScale(strength=0.85))
    g.add_levels(levels)
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    return g


def gridded() -> Grotrian:
    """⁸⁸Sr⁺ with gridlines and configurations shown in the labels."""
    sr = atomtoolkit.load_species('88Sr II')
    levels = sorted(sr.levels.values(), key=lambda l: l.level_Hz)[:8]
    theme = Theme(show_grid=True, label_gutter=150.0)
    g = Grotrian(sr, title='⁸⁸Sr⁺  ·  gridlines and configurations', theme=theme)
    g.add_levels(levels, show_conf=True)
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    return g


def barium_hyperfine() -> Grotrian:
    """¹³⁷Ba⁺ hyperfine structure on one global scale — the phase-2 reference.

    Shows why the scale must come from the largest span rather than the smallest gap:
    5d ²D₅/₂ has F=4 and F=3 only 0.496 MHz apart inside a 135 MHz span (ratio 272).
    Sized from that gap, one manifold would need a band thousands of pixels tall. Sized
    globally, ²D₅/₂ correctly collapses to almost a single line while the 8.04 GHz ground
    state sets the scale — and the F labels still fan out on leaders so it stays readable.
    """
    ba = atomtoolkit.load_species('137Ba II')
    g = Grotrian(ba, title='¹³⁷Ba⁺  ·  hyperfine structure, one global scale',
                 scale=CompressedScale(strength=0.6))
    for name in ('6s 2S1/2', '5d 2D3/2', '5d 2D5/2', '6p 2P*1/2', '6p 2P*3/2'):
        g.add_level(name, detail='hyperfine')
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    return g


def schematic_gross_real_hyperfine() -> Grotrian:
    """⁴³Ca⁺: schematic gross structure, quantitative hyperfine.

    The combination that is usually wanted for a talk figure — the gross axis compressed
    so everything is visible, while the hyperfine splittings stay on one true, comparable
    scale. The two transforms are independent, so this costs nothing extra.
    """
    ca = atomtoolkit.load_species('43Ca II')
    g = Grotrian(ca, title='⁴³Ca⁺  ·  schematic gross axis, true hyperfine scale',
                 scale=CompressedScale(strength=0.9))
    for name in ('3p6.4s 2S1/2', '3p6.3d 2D3/2', '3p6.3d 2D5/2',
                 '3p6.4p 2P*1/2', '3p6.4p 2P*3/2'):
        g.add_level(name, detail='hyperfine')
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    return g


def ytterbium_zeeman() -> Grotrian:
    """¹⁷¹Yb⁺ Zeeman fans at 5 G — hyperfine stacks in y, m_F fans in x."""
    yb = atomtoolkit.load_species('171Yb II')
    yb.levels['4f14.6s 2S1/2'].use_nonlinear_zeeman = True
    theme = Theme(column_width=200.0, plot_height=420.0)
    g = Grotrian(yb, title='¹⁷¹Yb⁺  ·  Zeeman fans at 5 G, exaggerated',
                 B=Q_(5.0, 'G'), theme=theme, scale=CompressedScale(),
                 z_scale='auto')
    for name in ('4f14.6s 2S1/2', '4f14.5d 2D3/2', '4f14.6p 2P*1/2', '4f14.5d 2D5/2', '4f13.(2F*).6s2 2F*7/2'):
        g.add_level(name, detail='zeeman')
    g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
    return g


def ytterbium_scheme() -> Grotrian:
    """¹⁷¹Yb⁺ with its actual laser scheme — the phase-3 reference diagram.

    Arrows are coloured by their own wavelength, which is the one piece of styling a
    Grotrian diagram can do that is genuinely physical. Non-visible lines dash themselves;
    spontaneous decay is drawn wavy and headless by convention.
    """
    yb = atomtoolkit.load_species('171Yb II')
    levels = ('4f14.6s 2S1/2', '4f14.5d 2D3/2', '4f14.5d 2D5/2',
              '4f13.(2F*).6s2 2F*7/2', '4f14.6p 2P*1/2', '4f14.6p 2P*3/2',
              '4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2')
    g = Grotrian(yb, title='¹⁷¹Yb⁺  ·  laser scheme',
                 scale=CompressedScale(strength=0.72))
    for name in levels:
        g.add_level(name, color='#5a5a5a')
    g.style_levels(lambda l: l.name in ('4f14.6s 2S1/2', '4f14.5d 2D3/2',
                                        '4f13.(2F*).6s2 2F*7/2'),
                   color='#1a1a1a', linewidth=3.2)

    # 369.5 nm cooling/detection, and its spontaneous decay back down
    g.add_transition(('4f14.6s 2S1/2', '4f14.6p 2P*1/2'), label='wavelength')
    g.add_transition(('4f14.6p 2P*1/2', '4f14.5d 2D3/2'), wavy=True, head='none')
    # 935 nm repump out of the D3/2 shelf
    g.add_transition(('4f14.5d 2D3/2', '4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2'),
                     label='wavelength')
    # 411 nm electric-quadrupole clock line
    g.add_transition(('4f14.6s 2S1/2', '4f14.5d 2D5/2'), route='arc', label='wavelength')
    # 3.4 um F7/2 -> D5/2, deep in the infrared so it dashes itself
    g.add_transition(('4f13.(2F*).6s2 2F*7/2', '4f14.5d 2D5/2'), label='wavelength')
    return g


def barium_hyperfine_transitions() -> Grotrian:
    """¹³⁷Ba⁺: a microwave transition *inside* one hyperfine manifold, plus an optical one.

    The F=1 -> F=2 ground-state transition lives entirely within a single magnified fan,
    which is the case that only works because arrows resolve against drawn anchors rather
    than true energies.
    """
    ba = atomtoolkit.load_species('137Ba II')
    g = Grotrian(ba, title='¹³⁷Ba⁺  ·  microwave and optical transitions',
                 scale=CompressedScale(strength=0.6))
    for name in ('6s 2S1/2', '5d 2D3/2', '6p 2P*1/2'):
        g.add_level(name, detail='hyperfine')
    ground = ba.levels['6s 2S1/2']
    g.add_transition((ground['F=1'], ground['F=2']), label='frequency',
                     route='arc', color='#1b6ca8')
    g.add_transition(('6s 2S1/2', '6p 2P*1/2'), label='wavelength')
    g.add_transition(('6p 2P*1/2', '5d 2D3/2'), wavy=True, head='none', label='wavelength')
    return g


GALLERY = {
    'yb171_flagship': flagship,
    'ca43_linear': calcium,
    'sr87_by_parity': strontium_by_parity,
    'ne20_dark': neon_dark,
    'ho165_stress': holmium,
    'sr88_grid': gridded,
    'ba137_hyperfine': barium_hyperfine,
    'ca43_schematic_gross': schematic_gross_real_hyperfine,
    'yb171_zeeman': ytterbium_zeeman,
    'yb171_scheme': ytterbium_scheme,
    'ba137_transitions': barium_hyperfine_transitions,
}


def main() -> None:
    for name, builder in GALLERY.items():
        grotrian = builder()
        scene = grotrian.build()
        path = grotrian.save(OUT / f'{name}.svg')
        print(f'{name:20s} {len(scene.lines):3d} levels  '
              f'{len(scene.headers)} columns  '
              f'{scene.width:5.0f}x{scene.height:.0f}  -> {path.relative_to(OUT.parent)}')


if __name__ == '__main__':
    main()
