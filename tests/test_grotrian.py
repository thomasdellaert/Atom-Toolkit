"""
Tests for the SVG Grotrian renderer.

The spec/layout/render split means almost everything here is a fast, pure test on
coordinates. Only the handful of end-to-end checks touch a real species.
"""

import xml.etree.ElementTree as ET

import pytest

from atomtoolkit import Hz, THz
from atomtoolkit.atom import Atom, EnergyLevel
from atomtoolkit.term import Term
from atomtoolkit.render.grotrian import (DARK, ColumnPolicy, CompressedScale, Grotrian,
                                         LinearScale, Theme)
from atomtoolkit.render.grotrian.layout import (estimate_text_width, layout,
                                                resolve_overlaps)
from atomtoolkit.render.grotrian.spec import LevelSpec, LevelStyle

SVG_NS = '{http://www.w3.org/2000/svg}'


@pytest.fixture
def toy_atom():
    """A small synthetic atom: cheap, and independent of the bundled species data."""
    a = Atom(name='toy')
    a.add_level(EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz))
    a.add_level(EnergyLevel(Term('1s2.2p', '2P*', 0.5), level=100 * THz))
    a.add_level(EnergyLevel(Term('1s2.2p', '2P*', 1.5), level=101 * THz))
    a.add_level(EnergyLevel(Term('1s2.3d', '2D', 1.5), level=180 * THz))
    a.add_level(EnergyLevel(Term('1s2.3d', '2D', 2.5), level=181 * THz))
    return a


# ---------------------------------------------------------------------------
# Label de-overlap
# ---------------------------------------------------------------------------

class TestResolveOverlaps:

    def test_empty(self):
        assert resolve_overlaps([], 10) == []

    def test_already_spaced_is_untouched(self):
        desired = [0.0, 20.0, 40.0]
        assert resolve_overlaps(desired, 10) == pytest.approx(desired)

    def test_coincident_points_are_spread(self):
        out = resolve_overlaps([50.0, 50.0, 50.0], 10)
        assert out == pytest.approx([40.0, 50.0, 60.0])   # symmetric about the mean

    def test_minimum_gap_is_honoured(self):
        out = resolve_overlaps([0.0, 1.0, 2.0, 3.0], 10)
        gaps = [b - a for a, b in zip(out, out[1:])]
        assert all(g >= 10 - 1e-9 for g in gaps)

    def test_order_is_preserved(self):
        out = resolve_overlaps([0.0, 0.5, 1.0, 30.0], 10)
        assert out == sorted(out)

    def test_centre_of_mass_is_preserved(self):
        """PAVA moves points minimally, so the mean should not drift."""
        desired = [10.0, 12.0, 13.0, 40.0]
        out = resolve_overlaps(desired, 10)
        assert sum(out) / len(out) == pytest.approx(sum(desired) / len(desired))

    def test_beats_greedy_on_displacement(self):
        """A greedy forward sweep piles all the error on one end; PAVA should not."""
        desired = [0.0] * 5
        out = resolve_overlaps(desired, 10)
        greedy = [i * 10.0 for i in range(5)]
        cost = sum((o - d) ** 2 for o, d in zip(out, desired))
        greedy_cost = sum((g - d) ** 2 for g, d in zip(greedy, desired))
        assert cost < greedy_cost


# ---------------------------------------------------------------------------
# Energy scales
# ---------------------------------------------------------------------------

class TestScales:

    def test_linear_is_monotonic_and_spans_the_unit_interval(self):
        scale = LinearScale().prepare([0.0, 1e14, 2e14])
        values = [scale.normalize(e) for e in (0.0, 1e14, 2e14)]
        assert values == sorted(values)
        assert 0.0 < values[0] < values[-1] < 1.0        # padding keeps it off the edges
        assert values[1] == pytest.approx(0.5)

    def test_linear_handles_a_single_level(self):
        scale = LinearScale().prepare([5e13])
        assert 0.0 <= scale.normalize(5e13) <= 1.0

    def test_compressed_is_monotonic(self):
        energies = [0.0, 1e14, 1.01e14, 1.02e14]
        scale = CompressedScale(strength=0.8).prepare(energies)
        values = [scale.normalize(e) for e in energies]
        assert values == sorted(values)

    def test_compressed_expands_crowded_regions(self):
        """Three levels bunched above a big gap should get more room than linearly."""
        energies = [0.0, 1e14, 1.01e14, 1.02e14]
        linear = LinearScale().prepare(energies)
        compressed = CompressedScale(strength=0.8).prepare(energies)
        lin_spread = linear.normalize(energies[3]) - linear.normalize(energies[1])
        com_spread = compressed.normalize(energies[3]) - compressed.normalize(energies[1])
        assert com_spread > lin_spread

    def test_ticks_lie_inside_the_plot(self):
        scale = LinearScale().prepare([0.0, 3e14])
        ticks = scale.ticks()
        assert ticks
        assert all(-0.001 <= frac <= 1.001 for frac, _ in ticks)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            LinearScale().prepare([])


# ---------------------------------------------------------------------------
# Column policies
# ---------------------------------------------------------------------------

class TestColumnPolicy:

    def test_by_J_sorts_ascending(self, toy_atom):
        policy = ColumnPolicy('J')
        keys = [policy(l)[0] for l in toy_atom.levels.values()]
        assert min(keys) == 0.5

    def test_by_parity_gives_two_columns(self, toy_atom):
        policy = ColumnPolicy('parity')
        assert {policy(l)[1] for l in toy_atom.levels.values()} == {'even', 'odd'}

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown column policy'):
            ColumnPolicy('nonsense')

    def test_callable_policy(self, toy_atom):
        g = Grotrian(toy_atom, columns=lambda l: (0.0, 'all'))
        g.add_levels(toy_atom.levels.values())
        assert len(g.build().headers) == 1


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class TestLayout:

    def _scene(self, atom, **kwargs):
        g = Grotrian(atom, **kwargs)
        g.add_levels(atom.levels.values())
        return g.build()

    def test_one_line_per_level(self, toy_atom):
        scene = self._scene(toy_atom)
        assert len(scene.lines) == len(toy_atom.levels)

    def test_every_level_gets_an_anchor(self, toy_atom):
        """Transitions will resolve against these, so none may be missing."""
        scene = self._scene(toy_atom)
        assert set(scene.anchors) == set(toy_atom.levels.values())

    def test_higher_energy_is_higher_on_the_page(self, toy_atom):
        scene = self._scene(toy_atom)
        ground = scene.anchors[toy_atom.levels['1s2.2s 2S1/2']]
        excited = scene.anchors[toy_atom.levels['1s2.3d 2D5/2']]
        assert excited.y0 < ground.y0          # SVG y grows downwards

    def test_labels_do_not_overlap_vertically(self, toy_atom):
        scene = self._scene(toy_atom)
        by_column = {}
        for label in scene.labels:
            by_column.setdefault(round(label.x, 3), []).append(label.y)
        for ys in by_column.values():
            ys.sort()
            gaps = [b - a for a, b in zip(ys, ys[1:])]
            assert all(g >= scene.theme.min_label_spacing - 1e-6 for g in gaps)

    def test_labels_stay_inside_the_canvas(self, toy_atom):
        scene = self._scene(toy_atom)
        for label in scene.labels:
            assert label.x + estimate_text_width(label.runs, label.size) <= scene.width + 1e-6

    def test_labels_do_not_intrude_on_the_next_column(self, toy_atom):
        """The reserved gutter is what makes this hold without real font metrics."""
        scene = self._scene(toy_atom)
        starts = sorted({round(line.x0, 3) for line in scene.lines})
        for label in scene.labels:
            following = [s for s in starts if s > label.x]
            if following:
                assert label.x + estimate_text_width(label.runs, label.size) <= following[0]

    def test_empty_diagram_is_rejected(self, toy_atom):
        with pytest.raises(ValueError, match='at least one level'):
            layout([], Theme(), ColumnPolicy('J'))

    def test_style_overrides_reach_the_scene(self, toy_atom):
        g = Grotrian(toy_atom)
        g.add_levels(toy_atom.levels.values(), color='#ff0000', linewidth=9.0)
        scene = g.build()
        assert all(line.color == '#ff0000' for line in scene.lines)
        assert all(line.linewidth == 9.0 for line in scene.lines)

    def test_label_can_be_suppressed(self, toy_atom):
        g = Grotrian(toy_atom)
        g.add_levels(toy_atom.levels.values(), label=False)
        assert g.build().labels == []


# ---------------------------------------------------------------------------
# The Grotrian facade
# ---------------------------------------------------------------------------

class TestGrotrian:

    def test_render_is_pure(self, toy_atom):
        """The matplotlib version doubled its state on every render; this must not."""
        g = Grotrian(toy_atom).add_levels(toy_atom.levels.values())
        first, second = g.render(), g.render()
        assert first == second
        assert len(g.build().lines) == len(toy_atom.levels)

    def test_chaining_returns_self(self, toy_atom):
        g = Grotrian(toy_atom)
        assert g.add_level(toy_atom.levels['1s2.2s 2S1/2']) is g
        assert g.style_levels(lambda l: True, color='#123456') is g

    def test_style_levels_is_selective(self, toy_atom):
        g = Grotrian(toy_atom).add_levels(toy_atom.levels.values())
        g.style_levels(lambda l: l.term.parity == -1, color='#b3541e')
        colors = {line.color for line in g.build().lines}
        assert '#b3541e' in colors and len(colors) == 2

    def test_lookup_by_name(self, toy_atom):
        g = Grotrian(toy_atom).add_level('1s2.2s 2S1/2')
        assert g.levels[0] is toy_atom.levels['1s2.2s 2S1/2']

    def test_lookup_by_name_without_atom_is_an_error(self):
        with pytest.raises(ValueError, match='no atom'):
            Grotrian().add_level('1s2.2s 2S1/2')

    def test_unknown_name_is_an_error(self, toy_atom):
        with pytest.raises(KeyError):
            Grotrian(toy_atom).add_level('not a level')

    def test_save_writes_a_file(self, toy_atom, tmp_path):
        path = Grotrian(toy_atom).add_levels(toy_atom.levels.values()).save(tmp_path / 'g.svg')
        assert path.exists()
        assert path.read_text(encoding='utf-8').startswith('<')

    def test_repr_svg_for_jupyter(self, toy_atom):
        g = Grotrian(toy_atom).add_levels(toy_atom.levels.values())
        assert g._repr_svg_().lstrip().startswith('<')

    def test_auto(self, toy_atom):
        g = Grotrian.auto(toy_atom, n_levels=3)
        assert len(g.levels) == 3
        assert g.title == 'toy'


# ---------------------------------------------------------------------------
# SVG output
# ---------------------------------------------------------------------------

class TestSvgOutput:

    @pytest.fixture
    def svg(self, toy_atom):
        return Grotrian.auto(toy_atom, n_levels=5).render()

    def test_is_well_formed_xml(self, svg):
        ET.fromstring(svg)

    def test_has_a_viewbox(self, svg):
        assert 'viewBox' in ET.fromstring(svg).attrib

    def test_draws_a_line_per_level(self, svg):
        root = ET.fromstring(svg)
        lines = root.findall(f'.//{SVG_NS}line')
        assert len(lines) >= 5

    def test_term_symbols_become_tspans(self, svg):
        """Superscript multiplicity and subscript J are the whole point of using SVG."""
        root = ET.fromstring(svg)
        spans = root.findall(f'.//{SVG_NS}tspan')
        assert spans
        texts = {s.text for s in spans if s.text}
        assert '2' in texts        # multiplicity
        assert '1/2' in texts      # J

    def test_levels_carry_tooltips(self, svg):
        root = ET.fromstring(svg)
        assert root.findall(f'.//{SVG_NS}title')

    def test_dark_theme_changes_the_background(self, toy_atom):
        light = Grotrian(toy_atom, theme=Theme()).add_levels(toy_atom.levels.values()).render()
        dark = Grotrian(toy_atom, theme=DARK).add_levels(toy_atom.levels.values()).render()
        assert DARK.background in dark
        assert dark != light


# ---------------------------------------------------------------------------
# End to end on real data
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRealSpecies:

    @pytest.mark.parametrize('name,n', [('171Yb II', 14), ('43Ca II', 10),
                                        ('165Ho I', 12), ('20Ne I', 12)])
    def test_species_render_cleanly(self, name, n):
        """Covers LS, JK and JJ term markup, and the awkward species."""
        import atomtoolkit
        atom = atomtoolkit.load_species(name)
        g = Grotrian.auto(atom, n_levels=n, scale=CompressedScale())
        scene, svg = g.build(), g.render()
        ET.fromstring(svg)
        assert len(scene.lines) == n
        assert all(0 <= line.y <= scene.height for line in scene.lines)
        for label in scene.labels:
            assert label.x + estimate_text_width(label.runs, label.size) <= scene.width + 1e-6


# ---------------------------------------------------------------------------
# Phase 2: hyperfine and Zeeman substructure
# ---------------------------------------------------------------------------

@pytest.fixture
def hf_atom():
    """A two-manifold atom with deliberately unequal hyperfine splittings.

    The excited manifold is 10x narrower than the ground one, which is what a *global*
    hyperfine scale has to represent faithfully.
    """
    from atomtoolkit import GHz
    a = Atom(name='hf-toy', I=1.5)
    ground = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz, hfA=1.0 * GHz)
    excited = EnergyLevel(Term('1s2.2p', '2P*', 0.5), level=100 * THz, hfA=0.1 * GHz)
    flat = EnergyLevel(Term('1s2.3d', '2D', 1.5), level=180 * THz)     # no hf data
    for level in (ground, excited, flat):
        a.add_level(level)
    return a


class TestHyperfineScale:

    def test_span_is_zero_without_data(self, hf_atom):
        from atomtoolkit.render.grotrian.layout import hyperfine_span
        assert hyperfine_span(hf_atom.levels['1s2.3d 2D3/2']) == 0.0

    def test_scale_is_driven_by_the_widest_span(self, hf_atom):
        from atomtoolkit.render.grotrian.layout import hyperfine_scale, hyperfine_span
        levels = [hf_atom.levels['1s2.2s 2S1/2'], hf_atom.levels['1s2.2p 2P*1/2']]
        m = hyperfine_scale(levels, band_px=46.0)
        widest = max(hyperfine_span(l) for l in levels)
        assert m * widest == pytest.approx(46.0)

    def test_scale_ignores_small_gaps(self):
        """The near-degeneracy case: a tiny internal gap must not inflate the scale.

        137Ba+ 5d 2D5/2 has F=4 and F=3 0.496 MHz apart inside a 135 MHz span. Sizing
        from the smallest gap would demand a band hundreds of times too tall.
        """
        from atomtoolkit.render.grotrian.layout import hyperfine_scale

        class FakeSub:
            def __init__(self, shift): self.shift_Hz = shift

        class FakeLevel:
            level_Hz = 0.0
            def sublevels(self):
                return [FakeSub(s) for s in (-30.225e6, -29.729e6, 33.139e6, 104.811e6)]

        m = hyperfine_scale([FakeLevel()], band_px=46.0)
        assert m * 135.036e6 == pytest.approx(46.0, rel=1e-3)

    def test_no_data_anywhere_gives_zero_scale(self, hf_atom):
        from atomtoolkit.render.grotrian.layout import hyperfine_scale
        assert hyperfine_scale([hf_atom.levels['1s2.3d 2D3/2']], 46.0) == 0.0

    def test_format_splitting(self):
        from atomtoolkit.render.grotrian.layout import format_splitting
        assert format_splitting(3.226e9).endswith('GHz')
        assert format_splitting(4.96e5).endswith('kHz')


class TestHyperfineLayout:

    def _scene(self, atom, **style):
        g = Grotrian(atom)
        g.add_level(atom.levels['1s2.2s 2S1/2'], detail='hyperfine', **style)
        g.add_level(atom.levels['1s2.2p 2P*1/2'], detail='hyperfine', **style)
        return g.build()

    def test_one_line_per_sublevel(self, hf_atom):
        scene = self._scene(hf_atom)
        expected = sum(len(hf_atom.levels[n].sublevels())
                       for n in ('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        assert len(scene.lines) == expected

    def test_sublevels_get_anchors(self, hf_atom):
        """Transitions in phase 3 will resolve against these."""
        scene = self._scene(hf_atom)
        for sublevel in hf_atom.levels['1s2.2s 2S1/2'].sublevels():
            assert sublevel in scene.anchors

    def test_manifold_anchor_spans_the_whole_fan(self, hf_atom):
        scene = self._scene(hf_atom)
        manifold = hf_atom.levels['1s2.2s 2S1/2']
        anchor = scene.anchors[manifold]
        sub_ys = [scene.anchors[s].y0 for s in manifold.sublevels()]
        assert anchor.y0 == pytest.approx(min(sub_ys))
        assert anchor.y1 == pytest.approx(max(sub_ys))

    def test_relative_sizes_are_preserved(self, hf_atom):
        """A 10x smaller splitting must be drawn 10x smaller -- the point of one scale."""
        scene = self._scene(hf_atom)
        def band(name):
            a = scene.anchors[hf_atom.levels[name]]
            return a.y1 - a.y0
        assert band('1s2.2s 2S1/2') / band('1s2.2p 2P*1/2') == pytest.approx(10.0, rel=1e-3)

    def test_ordering_is_by_energy_not_by_F(self):
        """A negative hyperfine A inverts the ladder (137Ba+ 5d 2D5/2 runs F=4,3,2,1)."""
        from atomtoolkit import GHz
        a = Atom(name='inverted', I=0.5)
        a.add_level(EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz, hfA=-1.0 * GHz))
        scene = Grotrian(a).add_level(a.levels['1s2.2s 2S1/2'],
                                      detail='hyperfine').build()
        level = a.levels['1s2.2s 2S1/2']
        for sublevel in level.sublevels():
            drawn_y = scene.anchors[sublevel].y0
            for other in level.sublevels():
                if other.shift_Hz > sublevel.shift_Hz:
                    assert scene.anchors[other].y0 < drawn_y   # higher energy is higher up

    def test_missing_data_falls_back_to_schematic(self, hf_atom):
        """Most manifolds have no hyperfine data, so this is the common path."""
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.3d 2D3/2'], detail='hyperfine')
        scene = g.build()
        assert len(scene.lines) == len(hf_atom.levels['1s2.3d 2D3/2'].sublevels())
        assert scene.brackets and scene.brackets[0].dashed   # dashed == not to scale

    def test_real_splittings_are_not_dashed(self, hf_atom):
        scene = self._scene(hf_atom)
        assert all(not b.dashed for b in scene.brackets)

    def test_scalebar_reports_the_reference_splitting(self, hf_atom):
        scene = self._scene(hf_atom)
        assert scene.scalebar is not None
        assert 'hyperfine scale' in scene.scalebar.label

    def test_explicit_hf_scale_is_respected(self, hf_atom):
        g = Grotrian(hf_atom, hf_scale=1e-8)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        scene = g.build()
        anchor = scene.anchors[hf_atom.levels['1s2.2s 2S1/2']]
        from atomtoolkit.render.grotrian.layout import hyperfine_span
        span = hyperfine_span(hf_atom.levels['1s2.2s 2S1/2'])
        assert anchor.y1 - anchor.y0 == pytest.approx(span * 1e-8)

    def test_hf_labels_do_not_move_the_lines(self, hf_atom):
        """Labels relax to stay readable; the geometry stays physical."""
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine')
        with_labels = g.build()
        g2 = Grotrian(hf_atom)
        g2.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine', hf_labels=False)
        without = g2.build()
        assert ([l.y for l in with_labels.lines] == [l.y for l in without.lines])


class TestZeemanLayout:

    def test_fan_is_horizontal(self, hf_atom):
        """Zeeman sublevels spread by m_F in x, unlike hyperfine which stacks in y."""
        g = Grotrian(hf_atom, B=0.0)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='zeeman')
        scene = g.build()
        xs = {round(line.x0, 3) for line in scene.lines}
        assert len(xs) > 1

    def test_one_line_per_zeeman_sublevel(self, hf_atom):
        g = Grotrian(hf_atom, B=0.0)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='zeeman')
        expected = sum(len(hf.sublevels())
                       for hf in hf_atom.levels['1s2.2s 2S1/2'].sublevels())
        assert len(g.build().lines) == expected

    def test_zeeman_sublevels_get_anchors(self, hf_atom):
        g = Grotrian(hf_atom, B=0.0)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='zeeman')
        scene = g.build()
        for hf in hf_atom.levels['1s2.2s 2S1/2'].sublevels():
            for z in hf.sublevels():
                assert z in scene.anchors

    def test_field_splits_the_fan_vertically(self, hf_atom):
        """At B=0 each F row is flat; at high field the m_F states separate in y."""
        level = hf_atom.levels['1s2.2s 2S1/2']
        flat = Grotrian(hf_atom, B=0.0).add_level(level, detail='zeeman').build()
        split = Grotrian(hf_atom, B=4000.0).add_level(level, detail='zeeman').build()
        assert len({round(l.y, 4) for l in split.lines}) > \
               len({round(l.y, 4) for l in flat.lines})


class TestScaleReferenceModes:
    """'auto' sizes to the excited states, not the ground state.

    Ground-state hyperfine splittings run 5-8x larger than anything else across the
    bundled species (5.5x in 43Ca+, 8.4x in 87Rb), so sizing to them squashes every other
    manifold to a sliver.
    """

    def test_auto_skips_the_lowest_manifold(self, hf_atom):
        from atomtoolkit.render.grotrian.layout import hyperfine_reference, hyperfine_span
        levels = [hf_atom.levels['1s2.2s 2S1/2'], hf_atom.levels['1s2.2p 2P*1/2']]
        excited_span = hyperfine_span(hf_atom.levels['1s2.2p 2P*1/2'])
        assert hyperfine_reference(levels, 'auto') == pytest.approx(excited_span)

    def test_widest_uses_the_ground_state(self, hf_atom):
        from atomtoolkit.render.grotrian.layout import hyperfine_reference, hyperfine_span
        levels = [hf_atom.levels['1s2.2s 2S1/2'], hf_atom.levels['1s2.2p 2P*1/2']]
        ground_span = hyperfine_span(hf_atom.levels['1s2.2s 2S1/2'])
        assert hyperfine_reference(levels, 'widest') == pytest.approx(ground_span)

    def test_auto_falls_back_when_only_one_manifold_has_data(self, hf_atom):
        from atomtoolkit.render.grotrian.layout import hyperfine_reference, hyperfine_span
        only = [hf_atom.levels['1s2.2s 2S1/2']]
        assert hyperfine_reference(only, 'auto') == pytest.approx(hyperfine_span(only[0]))

    def test_auto_makes_the_ground_state_overflow_proportionally(self, hf_atom):
        """The ground state is 10x the excited one here, so its band is 10x hf_band."""
        g = Grotrian(hf_atom, hf_scale='auto')
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine')
        scene = g.build()
        excited = scene.anchors[hf_atom.levels['1s2.2p 2P*1/2']]
        ground = scene.anchors[hf_atom.levels['1s2.2s 2S1/2']]
        assert excited.y1 - excited.y0 == pytest.approx(scene.theme.hf_band, rel=1e-3)
        assert ground.y1 - ground.y0 == pytest.approx(scene.theme.hf_band * 10, rel=1e-3)

    def test_quantity_scale(self, hf_atom):
        """hf_scale=Q_(1, 'GHz') means '1 GHz fills the band'."""
        from atomtoolkit import GHz
        g = Grotrian(hf_atom, hf_scale=1.0 * GHz)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        anchor = g.build().anchors[hf_atom.levels['1s2.2s 2S1/2']]
        # the ground manifold spans exactly 1 GHz here (hfA = 1 GHz, I=3/2, J=1/2)
        from atomtoolkit.render.grotrian.layout import hyperfine_span
        span = hyperfine_span(hf_atom.levels['1s2.2s 2S1/2'])
        expected = span / 1e9 * g.theme.hf_band
        assert anchor.y1 - anchor.y0 == pytest.approx(expected, rel=1e-6)


class TestZeemanExaggeration:

    def _bands(self, atom, **kwargs):
        level = atom.levels['1s2.2s 2S1/2']
        g = Grotrian(atom, B=2.0, **kwargs).add_level(level, detail='zeeman')
        scene = g.build()
        return scene, level

    def test_match_keeps_zeeman_on_the_hyperfine_scale(self, hf_atom):
        """Faithful, and therefore visually flat at ordinary fields.

        The offsets are non-zero -- the physics is still applied -- but at 2 G they come
        out sub-pixel, which is exactly why an exaggeration option is needed.
        """
        scene, level = self._bands(hf_atom, z_scale='match')
        ys = sorted(line.y for line in scene.lines)
        rows = []
        for y in ys:                       # cluster into F rows
            if rows and y - rows[-1][-1] < 1.0:
                rows[-1].append(y)
            else:
                rows.append([y])
        assert len(rows) == len(level.sublevels())
        assert all(row[-1] - row[0] < 1.0 for row in rows)   # each row is sub-pixel

    def test_auto_exaggerates_until_the_fan_splits(self, hf_atom):
        scene, level = self._bands(hf_atom, z_scale='auto')
        ys = {round(line.y, 6) for line in scene.lines}
        assert len(ys) > len(level.sublevels())

    def test_exaggeration_is_reported_on_the_scalebar(self, hf_atom):
        scene, _ = self._bands(hf_atom, z_scale='auto')
        assert scene.scalebar is not None
        assert 'exaggerated' in scene.scalebar.label

    def test_match_does_not_claim_exaggeration(self, hf_atom):
        scene, _ = self._bands(hf_atom, z_scale='match')
        assert scene.scalebar is None or 'exaggerated' not in scene.scalebar.label

    def test_hyperfine_spacing_is_unaffected_by_zeeman_exaggeration(self, hf_atom):
        """Exaggerating Zeeman must not silently rescale the hyperfine splitting."""
        level = hf_atom.levels['1s2.2s 2S1/2']
        flat = Grotrian(hf_atom, B=0.0, z_scale='match').add_level(
            level, detail='zeeman').build()
        loud = Grotrian(hf_atom, B=0.0, z_scale='auto').add_level(
            level, detail='zeeman').build()
        # at B=0 there is no Zeeman shift to exaggerate, so geometry must be identical
        assert sorted(round(l.y, 6) for l in flat.lines) == \
               sorted(round(l.y, 6) for l in loud.lines)


class TestPerLevelScales:
    """A per-manifold override, for when one manifold's internal structure must be
    resolved even though the shared scale collapses it (137Ba+ 5d 2D5/2 is ~1 px)."""

    def _band(self, scene, atom, name):
        anchor = scene.anchors[atom.levels[name]]
        return anchor.y1 - anchor.y0

    def test_default_inherits_the_global_scale(self, hf_atom):
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine')
        scene = g.build()
        ratio = (self._band(scene, hf_atom, '1s2.2s 2S1/2')
                 / self._band(scene, hf_atom, '1s2.2p 2P*1/2'))
        assert ratio == pytest.approx(10.0, rel=1e-3)      # true 10x size difference

    def test_fit_makes_one_manifold_fill_the_band(self, hf_atom):
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine', hf_scale='fit')
        scene = g.build()
        assert self._band(scene, hf_atom, '1s2.2p 2P*1/2') == \
            pytest.approx(scene.theme.hf_band, rel=1e-3)

    def test_exaggerate_multiplies_the_inherited_scale(self, hf_atom):
        plain = Grotrian(hf_atom).add_level(
            hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine').build()
        loud = Grotrian(hf_atom).add_level(
            hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine', hf_exaggerate=4.0).build()
        assert (self._band(loud, hf_atom, '1s2.2p 2P*1/2')
                == pytest.approx(4 * self._band(plain, hf_atom, '1s2.2p 2P*1/2'), rel=1e-6))

    def test_override_does_not_affect_other_manifolds(self, hf_atom):
        """The whole point of a shared scale is that one override stays local."""
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine', hf_exaggerate=6.0)
        scene = g.build()
        reference = Grotrian(hf_atom).add_level(
            hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine').add_level(
            hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine').build()
        assert (self._band(scene, hf_atom, '1s2.2s 2S1/2')
                == pytest.approx(self._band(reference, hf_atom, '1s2.2s 2S1/2')))

    def test_off_scale_manifold_is_annotated_with_its_true_span(self, hf_atom):
        """Drawing one manifold off the shared scale must not mislead about its size."""
        from atomtoolkit.render.grotrian.layout import format_splitting, hyperfine_span
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine', hf_exaggerate=5.0)
        scene = g.build()
        expected = format_splitting(hyperfine_span(hf_atom.levels['1s2.2p 2P*1/2']))
        texts = {run[0] for label in scene.labels for run in label.runs}
        assert expected in texts

    def test_on_scale_manifolds_are_not_annotated(self, hf_atom):
        from atomtoolkit.render.grotrian.layout import format_splitting, hyperfine_span
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine')
        scene = g.build()
        unwanted = format_splitting(hyperfine_span(hf_atom.levels['1s2.2p 2P*1/2']))
        texts = {run[0] for label in scene.labels for run in label.runs}
        assert unwanted not in texts

    def test_quantity_override(self, hf_atom):
        from atomtoolkit import GHz
        from atomtoolkit.render.grotrian.layout import hyperfine_span
        g = Grotrian(hf_atom).add_level(hf_atom.levels['1s2.2s 2S1/2'],
                                        detail='hyperfine', hf_scale=1.0 * GHz)
        scene = g.build()
        span = hyperfine_span(hf_atom.levels['1s2.2s 2S1/2'])
        assert self._band(scene, hf_atom, '1s2.2s 2S1/2') == \
            pytest.approx(span / 1e9 * scene.theme.hf_band, rel=1e-6)

    def test_per_level_zeeman_exaggeration(self, hf_atom):
        level = hf_atom.levels['1s2.2s 2S1/2']
        plain = Grotrian(hf_atom, B=2.0).add_level(level, detail='zeeman').build()
        loud = Grotrian(hf_atom, B=2.0).add_level(level, detail='zeeman',
                                                  z_exaggerate=500.0).build()
        spread = lambda sc: max(l.y for l in sc.lines) - min(l.y for l in sc.lines)
        assert spread(loud) > spread(plain)

    def test_overhang_accounts_for_the_override(self, hf_atom):
        """An exaggerated manifold must still fit on the canvas."""
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine', hf_exaggerate=4.0)
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine')
        scene = g.build()
        assert all(0 <= line.y <= scene.height for line in scene.lines)


# ---------------------------------------------------------------------------
# Phase 3: transitions
# ---------------------------------------------------------------------------

class TestWavelengthColour:

    def test_visible_band(self):
        from atomtoolkit.render.grotrian.color import is_visible
        assert is_visible(493) and is_visible(650)
        assert not is_visible(369) and not is_visible(935)

    def test_spectral_hues_are_right(self):
        """A green line must come out green, not merely 'some colour'."""
        from atomtoolkit.render.grotrian.color import wavelength_to_hex
        def rgb(nm):
            h = wavelength_to_hex(nm)
            return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        r, g, b = rgb(532)
        assert g > r and g > b                      # green
        r, g, b = rgb(650)
        assert r > g and r > b                      # red
        r, g, b = rgb(470)
        assert b > r                                # blue

    def test_far_infrared_desaturates(self):
        from atomtoolkit.render.grotrian.color import wavelength_to_hex
        h = wavelength_to_hex(3000)
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        assert max(r, g, b) - min(r, g, b) < 30      # essentially grey

    def test_degenerate_wavelength_is_handled(self):
        from atomtoolkit.render.grotrian.color import wavelength_to_hex
        assert wavelength_to_hex(0).startswith('#')


class TestTransitions:

    def _g(self, atom, **kwargs):
        g = Grotrian(atom, **kwargs)
        g.add_levels(atom.levels.values())
        return g

    def test_wavelength_from_level_energies(self, toy_atom):
        from atomtoolkit.render.grotrian.layout import transition_wavelength_nm
        lower = toy_atom.levels['1s2.2s 2S1/2']
        upper = toy_atom.levels['1s2.2p 2P*1/2']
        nm = transition_wavelength_nm(lower, upper)
        assert nm == pytest.approx(2.99792458e17 / 100e12, rel=1e-9)   # 100 THz

    def test_add_by_names_and_by_levels(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        g.add_transition((toy_atom.levels['1s2.2s 2S1/2'],
                          toy_atom.levels['1s2.3d 2D5/2']))
        assert len(g.build().arrows) == 2

    def test_order_is_normalised(self, toy_atom):
        """Given upper-first, the arrow must still point from lower to upper."""
        g = self._g(toy_atom)
        g.add_transition(('1s2.2p 2P*1/2', '1s2.2s 2S1/2'))
        arrow = g.build().arrows[0]
        assert arrow.y0 > arrow.y1                  # starts low on the page, ends high

    def test_accepts_a_transition_object(self, toy_atom):
        from atomtoolkit.atom import Transition
        t = Transition(toy_atom.levels['1s2.2s 2S1/2'], toy_atom.levels['1s2.2p 2P*1/2'])
        toy_atom.add_transition(t)
        g = self._g(toy_atom)
        g.add_transition(t)
        assert len(g.build().arrows) == 1

    def test_endpoints_use_drawn_anchors(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        scene = g.build()
        arrow = scene.arrows[0]
        lower = scene.anchors[toy_atom.levels['1s2.2s 2S1/2']]
        assert lower.x0 <= arrow.x0 <= lower.x1
        assert arrow.y0 == pytest.approx(lower.y0)

    def test_missing_level_warns_and_skips(self, toy_atom):
        g = Grotrian(toy_atom)
        g.add_level(toy_atom.levels['1s2.2s 2S1/2'])          # only one level drawn
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        with pytest.warns(UserWarning, match='not drawn'):
            scene = g.build()
        assert scene.arrows == []

    def test_sublevel_falls_back_to_the_drawn_manifold(self, hf_atom):
        """A transition to a sublevel of a manifold drawn collapsed still lands."""
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'])           # gross, no substructure
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'])
        sublevel = hf_atom.levels['1s2.2p 2P*1/2'].sublevels()[0]
        g.add_transition((hf_atom.levels['1s2.2s 2S1/2'], sublevel))
        scene = g.build()
        assert len(scene.arrows) == 1

    def test_transition_into_a_hyperfine_fan_lands_on_the_sublevel(self, hf_atom):
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'])
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine')
        sublevel = hf_atom.levels['1s2.2p 2P*1/2'].sublevels()[0]
        g.add_transition((hf_atom.levels['1s2.2s 2S1/2'], sublevel))
        scene = g.build()
        assert scene.arrows[0].y1 == pytest.approx(scene.anchors[sublevel].y0)

    def test_colour_comes_from_the_wavelength(self, toy_atom):
        from atomtoolkit.render.grotrian.color import wavelength_to_hex
        from atomtoolkit.render.grotrian.layout import transition_wavelength_nm
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        nm = transition_wavelength_nm(toy_atom.levels['1s2.2s 2S1/2'],
                                      toy_atom.levels['1s2.2p 2P*1/2'])
        assert g.build().arrows[0].color == wavelength_to_hex(nm)

    def test_explicit_colour_overrides(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'), color='#123456')
        assert g.build().arrows[0].color == '#123456'

    def test_non_visible_lines_are_dashed(self, toy_atom):
        """3 um here, well into the infrared."""
        g = self._g(toy_atom)
        g.add_transition(('1s2.2p 2P*3/2', '1s2.3d 2D3/2'))
        assert g.build().arrows[0].dashed

    def test_wavy_lines_are_not_also_dashed(self, toy_atom):
        """Wavy already signals 'different'; dashing a squiggle is just noise."""
        g = self._g(toy_atom)
        g.add_transition(('1s2.2p 2P*3/2', '1s2.3d 2D3/2'), wavy=True)
        arrow = g.build().arrows[0]
        assert arrow.wavy and not arrow.dashed

    def test_arc_control_is_perpendicular_to_the_chord(self, toy_atom):
        """Offsetting along +x barely bends a diagonal, so the bulge must be normal."""
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'), route='arc')
        arrow = g.build().arrows[0]
        assert arrow.control is not None
        mx, my = (arrow.x0 + arrow.x1) / 2, (arrow.y0 + arrow.y1) / 2
        chord = (arrow.x1 - arrow.x0, arrow.y1 - arrow.y0)
        offset = (arrow.control[0] - mx, arrow.control[1] - my)
        dot = chord[0] * offset[0] + chord[1] * offset[1]
        assert abs(dot) < 1e-6                      # perpendicular

    def test_straight_route_has_no_control(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'), route='straight')
        assert g.build().arrows[0].control is None

    def test_labels(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'), label='wavelength')
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'), label='frequency')
        g.add_transition(('1s2.2p 2P*1/2', '1s2.3d 2D3/2'), label='cooling')
        texts = {run[0] for label in g.build().labels for run in label.runs}
        assert any('nm' in t or 'µm' in t for t in texts)
        assert any('THz' in t or 'GHz' in t for t in texts)
        assert 'cooling' in texts

    def test_label_sits_off_the_line(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'), label='wavelength')
        scene = g.build()
        arrow = scene.arrows[0]
        label = [l for l in scene.labels if 'nm' in l.runs[0][0]][0]
        midpoint = ((arrow.x0 + arrow.x1) / 2, (arrow.y0 + arrow.y1) / 2)
        assert abs(label.x - midpoint[0]) > 1.0     # displaced, not sitting on the path

    def test_head_options(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'), head='none')
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'), head='both')
        heads = {arrow.head for arrow in g.build().arrows}
        assert heads == {'none', 'both'}

    def test_render_is_still_pure(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'), label='wavelength')
        assert g.render() == g.render()

    def test_svg_contains_arrow_paths_and_markers(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        root = ET.fromstring(g.render())
        assert root.findall(f'.//{SVG_NS}marker')
        assert root.findall(f'.//{SVG_NS}path')

    def test_wavy_path_has_many_points(self, toy_atom):
        g = self._g(toy_atom)
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'), wavy=True)
        root = ET.fromstring(g.render())
        paths = [p.get('d') for p in root.findall(f'.//{SVG_NS}path') if p.get('d')]
        assert any(path.count('L') > 10 for path in paths)


# ---------------------------------------------------------------------------
# Pluggable spectral colour models
# ---------------------------------------------------------------------------

class TestSpectralModels:

    def test_registry_models_all_work(self):
        from atomtoolkit.render.grotrian.color import SPECTRA
        for name, model in SPECTRA.items():
            assert model(500).startswith('#') and len(model(500)) == 7, name

    def test_get_spectrum_accepts_name_map_and_callable(self):
        from atomtoolkit.render.grotrian.color import (BrutonSpectrum, SPECTRA,
                                                       get_spectrum)
        assert get_spectrum('rainbow') is SPECTRA['rainbow']
        model = BrutonSpectrum()
        assert get_spectrum(model) is model
        assert get_spectrum(lambda nm: '#abcdef')(500) == '#abcdef'
        assert get_spectrum(None)(500).startswith('#')

    def test_unknown_name_lists_the_options(self):
        from atomtoolkit.render.grotrian.color import get_spectrum
        with pytest.raises(ValueError, match='rainbow'):
            get_spectrum('not-a-model')

    def test_bad_type_is_rejected(self):
        from atomtoolkit.render.grotrian.color import get_spectrum
        with pytest.raises(TypeError):
            get_spectrum(42)

    def test_interpolation_hits_its_control_points(self):
        from atomtoolkit.render.grotrian.color import InterpolatedSpectrum
        model = InterpolatedSpectrum([400, 600], [0, 255], [255, 0], [0, 0])
        assert model(400) == '#00ff00'
        assert model(600) == '#ff0000'
        assert model(500) == '#808000'          # halfway; blue is 0 at both ends

    def test_interpolation_clamps_outside_its_range(self):
        from atomtoolkit.render.grotrian.color import InterpolatedSpectrum
        model = InterpolatedSpectrum([400, 600], [0, 255], [255, 0], [0, 0])
        assert model(100) == model(400)
        assert model(9000) == model(600)

    def test_saturation_option(self):
        from atomtoolkit.render.grotrian.color import InterpolatedSpectrum
        muted = InterpolatedSpectrum([400, 600], [120, 120], [60, 60], [60, 60])
        vivid = InterpolatedSpectrum([400, 600], [120, 120], [60, 60], [60, 60],
                                     saturate=True)
        assert vivid(500) != muted(500)
        h = vivid(500)
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        assert min(r, g, b) == 0                # fully saturated

    def test_malformed_control_points_are_rejected(self):
        from atomtoolkit.render.grotrian.color import InterpolatedSpectrum
        with pytest.raises(ValueError, match='same length'):
            InterpolatedSpectrum([400, 600], [0], [0, 0], [0, 0])
        with pytest.raises(ValueError, match='two control points'):
            InterpolatedSpectrum([400], [0], [0], [0])
        with pytest.raises(ValueError, match='ascending'):
            InterpolatedSpectrum([600, 400], [0, 0], [0, 0], [0, 0])

    def test_models_disagree_as_expected(self):
        """The whole reason this is pluggable: the models are genuinely different."""
        from atomtoolkit.render.grotrian.color import SPECTRA
        assert SPECTRA['bruton'](493.5) != SPECTRA['rainbow'](493.5)

    def test_spectrum_changes_arrow_colour(self, toy_atom):
        g1 = Grotrian(toy_atom, spectrum='bruton')
        g2 = Grotrian(toy_atom, spectrum='rainbow')
        for g in (g1, g2):
            g.add_levels(toy_atom.levels.values())
            g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        assert g1.build().arrows[0].color != g2.build().arrows[0].color

    def test_bad_spectrum_fails_at_construction(self, toy_atom):
        with pytest.raises(ValueError):
            Grotrian(toy_atom, spectrum='nope')


# ---------------------------------------------------------------------------
# Block ("fancy") arrows
# ---------------------------------------------------------------------------

class TestBlockArrows:

    def _arrow(self, atom, **style):
        g = Grotrian(atom)
        g.add_levels(atom.levels.values())
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'), block=True, **style)
        return g.build()

    def test_block_produces_a_polygon(self, toy_atom):
        arrow = self._arrow(toy_atom).arrows[0]
        assert arrow.polygon is not None and len(arrow.polygon) == 7

    def test_tail_is_flat(self, toy_atom):
        """The defining feature: the tail is cut horizontally, flush with the level."""
        arrow = self._arrow(toy_atom).arrows[0]
        first, last = arrow.polygon[0], arrow.polygon[-1]
        assert first[1] == pytest.approx(last[1])
        assert first[1] == pytest.approx(arrow.y0)

    def test_shaft_is_as_wide_as_the_source_level(self, toy_atom):
        scene = self._arrow(toy_atom)
        arrow = scene.arrows[0]
        source = scene.anchors[toy_atom.levels['1s2.2s 2S1/2']]
        width = abs(arrow.polygon[-1][0] - arrow.polygon[0][0])
        assert width == pytest.approx(source.x1 - source.x0)

    def test_explicit_width_overrides(self, toy_atom):
        arrow = self._arrow(toy_atom, block_width=40.0).arrows[0]
        assert abs(arrow.polygon[-1][0] - arrow.polygon[0][0]) == pytest.approx(40.0)

    def test_tip_is_the_endpoint(self, toy_atom):
        arrow = self._arrow(toy_atom).arrows[0]
        assert arrow.polygon[3] == pytest.approx((arrow.x1, arrow.y1))

    def test_head_is_wider_than_the_shaft(self, toy_atom):
        arrow = self._arrow(toy_atom).arrows[0]
        shoulder_span = abs(arrow.polygon[1][0] - arrow.polygon[5][0])
        flare_span = abs(arrow.polygon[2][0] - arrow.polygon[4][0])
        assert flare_span > shoulder_span

    def test_head_scales_with_width_not_length(self, toy_atom):
        """A fraction-of-length head becomes an enormous wedge on a long diagonal."""
        narrow = self._arrow(toy_atom, block_width=20.0).arrows[0]
        wide = self._arrow(toy_atom, block_width=60.0).arrows[0]
        def head_len(a):
            import math
            return math.hypot(a.polygon[3][0] - a.polygon[1][0],
                              a.polygon[3][1] - a.polygon[1][1])
        assert head_len(wide) > head_len(narrow)

    def test_block_arrows_are_translucent_by_default(self, toy_atom):
        assert self._arrow(toy_atom).arrows[0].opacity < 1.0

    def test_svg_emits_a_polygon(self, toy_atom):
        g = Grotrian(toy_atom)
        g.add_levels(toy_atom.levels.values())
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'), block=True)
        root = ET.fromstring(g.render())
        assert root.findall(f'.//{SVG_NS}polygon')

    def test_non_block_arrows_have_no_polygon(self, toy_atom):
        g = Grotrian(toy_atom)
        g.add_levels(toy_atom.levels.values())
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'))
        assert g.build().arrows[0].polygon is None


# ---------------------------------------------------------------------------
# Phase 4: broken axes, level-of-detail labelling, auto() policy choice
# ---------------------------------------------------------------------------

class TestBrokenScale:

    ENERGIES = [0.0, 1e12, 2e12, 6e14, 6.1e14, 6.2e14]      # a big empty stretch

    def test_monotonic(self):
        from atomtoolkit.render.grotrian import BrokenScale
        scale = BrokenScale(breaks=[(3e12, 5.9e14)]).prepare(self.ENERGIES)
        values = [scale.normalize(e) for e in self.ENERGIES]
        assert values == sorted(values)

    def test_break_is_collapsed(self):
        from atomtoolkit.render.grotrian import BrokenScale, LinearScale
        broken = BrokenScale(breaks=[(3e12, 5.9e14)], gap=0.04).prepare(self.ENERGIES)
        linear = LinearScale().prepare(self.ENERGIES)
        # the collapsed range itself must occupy exactly `gap`
        assert broken.normalize(5.9e14) - broken.normalize(3e12) == pytest.approx(0.04,
                                                                                  abs=1e-6)
        assert linear.normalize(5.9e14) - linear.normalize(3e12) > 0.8

    def test_retained_regions_stay_linear(self):
        """The point of a break, versus a compressed scale: what is kept is exact."""
        from atomtoolkit.render.grotrian import BrokenScale
        scale = BrokenScale(breaks=[(3e12, 5.9e14)]).prepare(self.ENERGIES)
        a = scale.normalize(6.1e14) - scale.normalize(6.0e14)
        b = scale.normalize(6.2e14) - scale.normalize(6.1e14)
        assert a == pytest.approx(b, rel=1e-6)

    def test_spanning_finds_the_emptiest_stretch(self):
        from atomtoolkit.render.grotrian import BrokenScale
        scale = BrokenScale.spanning(self.ENERGIES, n_breaks=1)
        assert len(scale.breaks) == 1
        low, high = scale.breaks[0]
        assert low == pytest.approx(2e12) and high == pytest.approx(6e14)

    def test_spanning_declines_when_levels_are_even(self):
        """No arbitrary break on a diagram that does not need one."""
        from atomtoolkit.render.grotrian import BrokenScale
        even = [0.0, 1e14, 2e14, 3e14, 4e14]
        assert BrokenScale.spanning(even, n_breaks=1).breaks == []

    def test_break_positions_are_reported(self):
        from atomtoolkit.render.grotrian import BrokenScale
        scale = BrokenScale(breaks=[(3e12, 5.9e14)], gap=0.05).prepare(self.ENERGIES)
        positions = scale.break_positions
        assert len(positions) == 1
        lo, hi = positions[0]
        assert hi - lo == pytest.approx(0.05, abs=1e-6)

    def test_accepts_quantities(self):
        from atomtoolkit import THz
        from atomtoolkit.render.grotrian import BrokenScale
        scale = BrokenScale(breaks=[(3 * THz, 590 * THz)], gap=0.04).prepare(self.ENERGIES)
        assert scale.normalize(5.9e14) - scale.normalize(3e12) == pytest.approx(0.04,
                                                                                abs=1e-6)

    def test_scene_exposes_break_markers(self, toy_atom):
        from atomtoolkit.render.grotrian import BrokenScale
        g = Grotrian(toy_atom, scale=BrokenScale(breaks=[(5e12, 90e12)]))
        g.add_levels(toy_atom.levels.values())
        scene = g.build()
        assert scene.axis_breaks
        assert all(top < bottom for top, bottom in scene.axis_breaks)

    def test_toy_spread_is_not_broken(self, toy_atom):
        """Two comparably large gaps means no single break helps; leave it alone."""
        from atomtoolkit.render.grotrian import BrokenScale
        energies = [l.level_Hz for l in toy_atom.levels.values()]
        assert BrokenScale.spanning(energies).breaks == []


class TestZeemanLabels:

    def _labels(self, atom, level, theme=None, **style):
        """theme goes to the Grotrian; everything else is per-level style."""
        g = Grotrian(atom, B=2.0, z_scale='auto', **({'theme': theme} if theme else {}))
        g.add_level(level, detail='zeeman', **style)
        scene = g.build()
        return [''.join(run[0] for run in label.runs) for label in scene.labels]

    def test_zeeman_fans_are_labelled(self, hf_atom):
        """They were previously drawn with no labels at all."""
        texts = self._labels(hf_atom, hf_atom.levels['1s2.2s 2S1/2'])
        assert any(t.startswith('F') for t in texts)

    def test_mF_labels_are_off_by_default(self, hf_atom):
        """m_F sublevels are evenly spaced; most readers just count along."""
        texts = self._labels(hf_atom, hf_atom.levels['1s2.2s 2S1/2'],
                             theme=Theme(column_width=400.0))
        assert not any(t.lstrip('-').replace('/', '').isdigit() for t in texts)

    def test_wide_columns_label_each_slot_once(self, hf_atom):
        """One shared row of m_F labels serves the manifold: F rows share m_F columns."""
        texts = self._labels(hf_atom, hf_atom.levels['1s2.2s 2S1/2'], z_labels=True,
                             theme=Theme(column_width=400.0))
        assert '-1' in texts and '1' in texts
        assert texts.count('0') == 1        # not repeated per F row

    def test_narrow_columns_summarise_instead(self, hf_atom):
        """Never print illegible overlapping numbers; say the range once."""
        texts = self._labels(hf_atom, hf_atom.levels['1s2.2s 2S1/2'], z_labels=True,
                             theme=Theme(column_width=40.0))
        assert any('..' in t for t in texts)
        assert '-1' not in texts

    def test_row_labels_can_be_switched_off(self, hf_atom):
        g = Grotrian(hf_atom, B=2.0)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='zeeman',
                    hf_labels=False, z_labels=False)
        texts = [''.join(r[0] for r in l.runs) for l in g.build().labels]
        # the manifold's term symbol stays; the F and m_F labels go
        assert not any(t.startswith('F') or t.lstrip('-').isdigit() for t in texts)

    def test_labels_use_fractions_not_decimals(self):
        """Half-integer m_F must read 1/2, matching the rest of the toolkit."""
        from atomtoolkit import GHz
        a = Atom(name='half', I=0.0)
        a.add_level(EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz, hfA=1 * GHz))
        g = Grotrian(a, B=2.0, theme=Theme(column_width=400.0))
        g.add_level(a.levels['1s2.2s 2S1/2'], detail='zeeman', z_labels=True)
        texts = [''.join(r[0] for r in l.runs) for l in g.build().labels]
        assert '1/2' in texts and '-1/2' in texts
        assert '0.5' not in texts

    def test_no_negative_zero(self, hf_atom):
        texts = self._labels(hf_atom, hf_atom.levels['1s2.2s 2S1/2'], z_labels=True,
                             theme=Theme(column_width=400.0))
        assert '-0' not in texts


class TestAutoPolicyChoice:

    def test_auto_picks_a_policy(self, toy_atom):
        g = Grotrian.auto(toy_atom, n_levels=5)
        assert g.columns.by in ('J', 'L', 'parity')

    def test_explicit_columns_are_respected(self, toy_atom):
        g = Grotrian.auto(toy_atom, n_levels=5, columns='parity')
        assert g.columns.by == 'parity'

    def test_wide_layouts_are_avoided(self, toy_atom):
        """The chosen layout must be no worse than every alternative on aspect."""
        chosen = Grotrian.auto(toy_atom, n_levels=5).build()
        ratio = chosen.width / chosen.height
        assert 0.3 < ratio < 4.0


class TestOverlapWarning:
    """Overlapping fans make a diagram wrong, not merely ugly, so they are reported.

    Nothing is silently rescaled: the shared scale is what makes manifolds comparable, and
    shrinking it to fit would trade a visible problem for an invisible one.
    """

    def _grotrian(self, hf_atom, plot_height, hf_band=46.0):
        g = Grotrian(hf_atom, theme=Theme(plot_height=plot_height, hf_band=hf_band))
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        g.add_level(hf_atom.levels['1s2.2p 2P*1/2'], detail='hyperfine')
        return g

    def test_cramped_layout_warns(self, hf_atom):
        with pytest.warns(UserWarning, match='overlap'):
            self._grotrian(hf_atom, plot_height=40.0, hf_band=200.0).build()

    def test_roomy_layout_is_silent(self, hf_atom, recwarn):
        self._grotrian(hf_atom, plot_height=900.0, hf_band=20.0).build()
        assert not [w for w in recwarn if 'overlap' in str(w.message)]

    def test_the_warning_names_the_manifolds(self, hf_atom):
        with pytest.warns(UserWarning) as record:
            self._grotrian(hf_atom, plot_height=40.0, hf_band=200.0).build()
        message = ' '.join(str(w.message) for w in record)
        assert '2S1/2' in message and '2P*1/2' in message

    def test_nothing_is_silently_rescaled(self, hf_atom):
        """The band must stay the size the scale asked for, warning or not."""
        from atomtoolkit.render.grotrian.layout import hyperfine_span
        with pytest.warns(UserWarning):
            scene = self._grotrian(hf_atom, plot_height=40.0, hf_band=200.0).build()
        anchor = scene.anchors[hf_atom.levels['1s2.2p 2P*1/2']]
        assert anchor.y1 - anchor.y0 == pytest.approx(200.0, rel=1e-3)


class TestAxisAutoHide:
    """An energy axis invites reading values off the page. Inside a magnified manifold
    that reading is wrong, so 'auto' drops the axis once substructure is drawn."""

    def test_axis_shown_without_substructure(self, toy_atom):
        g = Grotrian(toy_atom).add_levels(toy_atom.levels.values())
        assert g.build().ticks

    def test_axis_hidden_with_hyperfine(self, hf_atom):
        g = Grotrian(hf_atom)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        assert g.build().ticks == []

    def test_axis_hidden_with_zeeman(self, hf_atom):
        g = Grotrian(hf_atom, B=1.0)
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='zeeman')
        assert g.build().ticks == []

    def test_axis_can_be_forced_on(self, hf_atom):
        g = Grotrian(hf_atom, theme=Theme(show_axis=True))
        g.add_level(hf_atom.levels['1s2.2s 2S1/2'], detail='hyperfine')
        assert g.build().ticks

    def test_axis_can_be_forced_off(self, toy_atom):
        g = Grotrian(toy_atom, theme=Theme(show_axis=False))
        g.add_levels(toy_atom.levels.values())
        assert g.build().ticks == []


# ---------------------------------------------------------------------------
# Inline notebook display
# ---------------------------------------------------------------------------

class TestNotebookDisplay:

    def _g(self, atom):
        g = Grotrian(atom)
        g.add_levels(atom.levels.values())
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
        return g

    def test_file_render_pins_a_pixel_size(self, toy_atom):
        """Standalone files keep an explicit size; that is predictable in a paper."""
        root = ET.fromstring(self._g(toy_atom).render())
        assert root.get('width', '').endswith('px')
        assert root.get('height', '').endswith('px')

    def test_responsive_render_scales_to_its_container(self, toy_atom):
        root = ET.fromstring(self._g(toy_atom).render(responsive=True))
        assert root.get('width') == '100%'
        assert root.get('height') is None          # height follows from the viewBox
        assert 'max-width' in root.get('style', '')
        assert root.get('viewBox')                 # aspect ratio preserved

    def test_repr_svg_is_responsive(self, toy_atom):
        """A 1600 px diagram must not overflow a notebook cell."""
        assert 'width="100%"' in self._g(toy_atom)._repr_svg_()

    def test_repr_svg_is_valid_xml(self, toy_atom):
        ET.fromstring(self._g(toy_atom)._repr_svg_())

    def test_tooltips_survive_into_the_inline_form(self, toy_atom):
        root = ET.fromstring(self._g(toy_atom)._repr_svg_())
        assert root.findall(f'.//{SVG_NS}title')

    def test_marker_ids_distinguish_different_geometry(self, toy_atom):
        """Notebooks put every diagram in one DOM, so url(#id) must not cross-resolve."""
        import re
        ids = []
        for head in (7.0, 20.0):
            g = Grotrian(toy_atom, theme=Theme(transition_head=head))
            g.add_levels(toy_atom.levels.values())
            g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'))
            ids.append(set(re.findall(r'id="([^"]+)"', g.render())))
        assert ids[0] and ids[1]
        assert not (ids[0] & ids[1])

    def test_identical_markers_still_share_an_id(self, toy_atom):
        """Deduplication must survive the uniqueness fix."""
        import re
        g = Grotrian(toy_atom)
        g.add_levels(toy_atom.levels.values())
        g.add_transition(('1s2.2s 2S1/2', '1s2.2p 2P*1/2'), color='#112233')
        g.add_transition(('1s2.2s 2S1/2', '1s2.3d 2D5/2'), color='#112233')
        svg = g.render()
        assert len(re.findall(r'<marker', svg)) == 1


class TestProgressBarsAreQuiet:
    """Building an atom wrote thousands of carriage-return updates. Anything recording
    streams rather than rendering them -- a notebook, a log, CI -- got them all."""

    def test_tqdm_defaults_to_tty_gated(self):
        from atomtoolkit.atom import tqdm
        bar = tqdm(range(3))
        assert bar.disable is True or not __import__('sys').stderr.isatty()
        bar.close()

    def test_explicit_disable_is_respected(self):
        from atomtoolkit.atom import tqdm
        bar = tqdm(range(3), disable=False)
        assert bar.disable is False
        bar.close()
