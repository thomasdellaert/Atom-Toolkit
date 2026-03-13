import json
import itertools

import pytest
from atomtoolkit.atom import Atom, Transition, EnergyLevel
from atomtoolkit.term import Term, MultiTerm
from atomtoolkit import *
import numpy as np


# class TestAtomCreation:
#     def test_creation(self):
#         import atomtoolkit.species.Yb_II_171
#         assert isinstance(atomtoolkit.species.Yb_II_171.Yb171, Atom)
#         import atomtoolkit.species.Yb_II_174
#         assert isinstance(atomtoolkit.species.Yb_II_174.Yb174, Atom)
#         import atomtoolkit.species.Yb_II_173
#         assert isinstance(atomtoolkit.species.Yb_II_173.Yb173, Atom)

@pytest.fixture
def dummy_atom():
    a = Atom(name="dummy")
    gs = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz, alias='gs')
    a.add_level(gs)
    e1 = EnergyLevel(Term('1s2.2p', '2P', 1.5), level=200 * THz)
    a.add_level(e1)
    e2 = EnergyLevel(Term('1s2.3d', '2D', 2.5), level=100 * THz)
    a.add_level(e2)
    tc = Transition(gs, e1, A=10 * MHz, alias='cooling')
    a.add_transition(tc)
    tw = Transition(gs, e2, A=100 * Hz)
    a.add_transition(tw)
    td = Transition(e1, e2, A=5 * MHz)
    a.add_transition(td)
    return a


@pytest.fixture(params=['1s2.2s 2S1/2', '1s2.2p 2P3/2', '1s2.3d 2D5/2'])
def lvl(request):
    a = Atom(name="dummy", I=1 / 2)
    gs = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz, hfA=1 * GHz)
    a.add_level(gs)
    e1 = EnergyLevel(Term('1s2.2p', '2P', 1.5), level=200 * THz, hfA=1 * GHz)
    a.add_level(e1)
    e2 = EnergyLevel(Term('1s2.3d', '2D', 2.5), level=100 * THz, lande=1.204, hfA=1 * GHz)
    a.add_level(e2)
    return a.levels[request.param]


class TestAtom:
    def test_init(self, dummy_atom):
        assert dummy_atom.name == 'dummy'
        assert dummy_atom.I == 0

    def test_str(self, dummy_atom):
        assert str(dummy_atom) == 'dummy'

    def test_repr(self, dummy_atom):
        assert 'Atom' in dummy_atom.__repr__()

    def test_add_level_and_transition(self, dummy_atom):
        d = EnergyLevel(Term('1s2.4s', '2S', 0.5), level=10.0 * THz)
        dummy_atom.add_level(d)
        assert d in dummy_atom.levels.values()

        td = Transition(d, dummy_atom.levels['1s2.2s 2S1/2'], A=50 * MHz)
        dummy_atom.add_transition(td)

    def test_save(self, dummy_atom):
        dummy_atom.save('dummy')
        dummy_atom.save('dummy.atom')

    def test_load(self):
        da = Atom.load('dummy.atom')
        assert da.name == 'dummy'

    def test_from_dataframe(self):
        pass  # TODO

    def test_populate_transitions(self):
        pass  # TODO

    def test_linked_levels(self, dummy_atom):
        assert len(dummy_atom.linked_levels('1s2.2s 2S1/2')) == 2
        assert len(dummy_atom.linked_levels('1s2.2p 2P3/2')) == 2
        assert len(dummy_atom.linked_levels('1s2.3d 2D5/2')) == 2
        assert len(dummy_atom.linked_levels('1s2.2s 2S1/2 F=1/2')) == 2

    def test_state_lifetime(self, dummy_atom):
        s = ureg.s
        assert dummy_atom.state_lifetime('1s2.2s 2S1/2').to(s).magnitude == np.inf
        assert abs(dummy_atom.state_lifetime('1s2.2p 2P3/2').to(s).magnitude - 4e-7) < 1e-7
        assert abs(dummy_atom.state_lifetime('1s2.3d 2D5/2').to(s).magnitude - 0.06) < 0.1

    def test_branching_ratios(self, dummy_atom):
        assert dummy_atom.compute_branching_ratios('1s2.2s 2S1/2') == {}
        assert dummy_atom.compute_branching_ratios('1s2.2p 2P3/2') == {'1s2.2s 2S1/2': 0.6666666666666666,
                                                                       '1s2.3d 2D5/2': 0.3333333333333333}
        assert dummy_atom.compute_branching_ratios('1s2.3d 2D5/2') == {'1s2.2s 2S1/2': 1.0}


class TestAtomAccess:
    def test_sublevels(self, dummy_atom):
        assert len(dummy_atom.levels['1s2.2s 2S1/2'].sublevels()) == 1
        assert len(dummy_atom.levels['1s2.2p 2P3/2'].sublevels()) == 1
        assert len(dummy_atom.levels['1s2.3d 2D5/2'].sublevels()) == 1

    def test_level_alias(self, dummy_atom):
        assert dummy_atom.levels['gs'] is dummy_atom.levels['1s2.2s 2S1/2']

    def test_transition_alias(self, dummy_atom):
        assert dummy_atom.transitions['cooling'] is dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')]

    def test_level_alias_assignment(self, dummy_atom):
        dummy_atom.levels['1s2.2p 2P3/2'].alias = 'es'
        assert dummy_atom.levels['es'] is dummy_atom.levels['1s2.2p 2P3/2']

    def test_transition_alias_assignment(self, dummy_atom):
        dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.3d 2D5/2')].alias = 'shelving'
        assert dummy_atom.transitions['shelving'] is dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.3d 2D5/2')]

    def test_sublevel_access(self, dummy_atom):
        ways_of_calling = [
            dummy_atom.levels['1s2.2s 2S1/2']['F=1/2']['mF=1/2'],
            dummy_atom.levels['1s2.2s 2S1/2 F=1/2']['mF=1/2'],
            dummy_atom.levels['1s2.2s 2S1/2 F=1/2 mF=1/2'],
            dummy_atom.levels['1s2.2s 2S1/2']['mJ=1/2'],
            dummy_atom.levels['1s2.2s 2S1/2 mJ=1/2'],
        ]
        for a, b in itertools.product(ways_of_calling, ways_of_calling):
            assert (a is b) and (a is not None)

    def test_sublevel_iteration(self, dummy_atom):
        assert len([l for l in dummy_atom.levels['1s2.2s 2S1/2']]) == 1
        assert len([l for l in dummy_atom.levels['1s2.2s 2S1/2']['F=1/2']]) == 2
        assert len([l for l in dummy_atom.levels['1s2.2p 2P3/2']]) == 1
        assert len([l for l in dummy_atom.levels['1s2.2p 2P3/2']['F=3/2']]) == 4

    def test_transition_access(self, dummy_atom):
        ways_of_calling = [
            dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')],
            dummy_atom.transitions[('1s2.2p 2P3/2', '1s2.2s 2S1/2')],
        ]
        for a, b in itertools.product(ways_of_calling, ways_of_calling):
            assert (a is b) and (a is not None)

    def test_hf_transition_access(self, dummy_atom):
        ways_of_calling = [
            dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')][('F=1/2', 'F=3/2')],
            dummy_atom.transitions[('1s2.2s 2S1/2 F=1/2', '1s2.2p 2P3/2 F=3/2')],
            dummy_atom.transitions[('1s2.2p 2P3/2 F=3/2', '1s2.2s 2S1/2 F=1/2')],
        ]
        for a, b in itertools.product(ways_of_calling, ways_of_calling):
            assert (a is b) and (a is not None)

    def test_zeeman_transition_access(self, dummy_atom):
        ways_of_calling = [
            dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')][('F=1/2', 'F=3/2')][('mF=1/2', 'mF=-1/2')],
            dummy_atom.transitions[('1s2.2s 2S1/2 F=1/2', '1s2.2p 2P3/2 F=3/2')][('mF=1/2', 'mF=-1/2')],
            dummy_atom.transitions[('1s2.2s 2S1/2 F=1/2 mF=1/2', '1s2.2p 2P3/2 F=3/2 mF=-1/2')],
            dummy_atom.transitions[('1s2.2p 2P3/2 F=3/2 mF=-1/2', '1s2.2s 2S1/2 F=1/2 mF=1/2')],
        ]
        for a, b in itertools.product(ways_of_calling, ways_of_calling):
            assert (a is b) and (a is not None)


class TestAtomEditing:
    def test_etransition_freq_assignment(self, dummy_atom):
        dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].set_frequency(110 * THz)
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 110 * THz

        dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq = 115 * THz
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 115 * THz

    def test_hftransition_freq_assignment(self, dummy_atom):
        dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')][('F=1/2', 'F=3/2')].set_frequency(110 * THz)
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 110 * THz

    def test_ztransition_freq_assignment(self, dummy_atom):
        dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')][('F=1/2', 'F=3/2')][('mF=1/2', 'mF=-1/2')].set_frequency(
            110 * THz)
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 110 * THz

    def test_elevel_level_assignment(self, dummy_atom):
        dummy_atom.levels['1s2.2p 2P3/2'].level = 110 * THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 110 * THz

    def test_hflevel_level_assignment(self, dummy_atom):
        dummy_atom.levels['1s2.2p 2P3/2']['F=3/2'].level = 110 * THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 110 * THz

        # dummy_atom.hflevels['1s2.2p 2P3/2 F=3/2'].level = 120 * THz
        # assert dummy_atom.levels['1s2.2p 2P3/2'].level == 120 * THz

    def test_zlevel_level_assignment(self, dummy_atom):
        dummy_atom.levels['1s2.2p 2P3/2']['F=3/2']['mF=1/2'].level = 110 * THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 110 * THz

        dummy_atom.levels['1s2.2p 2P3/2 F=3/2 mF=1/2'].level = 120 * THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 120 * THz


class TestEnergyLevel:
    def test_shift(self, lvl):
        assert lvl.shift == lvl.level

    def test_shift_Hz(self, lvl):
        assert lvl.shift_Hz == lvl.level_Hz

    def test_len(self, lvl):
        assert len(lvl) == 2

    def test_del(self, lvl):
        del lvl

    def test_values(self, lvl):
        assert len(lvl.values()) == 2

    def test_sublevels(self, lvl):
        assert list(lvl.sublevels())[0] is list(lvl.values())[0]

    def test_keys(self, lvl):
        assert len(lvl.keys()) == 2

    def test_items(self, lvl):
        assert len(lvl.items()) == 2

    def test_HFs(self, lvl):
        assert lvl.hfA == 1 * GHz
        assert lvl.hfB == 0 * Hz
        assert lvl.hfC == 0 * Hz

        lvl.hfA = 0.5 * GHz
        lvl.hfB = 0.2 * GHz
        lvl.hfC = 1 * MHz
        assert lvl.hfA == 0.5 * GHz
        assert lvl.hfB == 0.2 * GHz
        assert lvl.hfC == 0.001 * GHz


# ---------------------------------------------------------------------------
# TestZeemanShift
# ---------------------------------------------------------------------------

class TestZeemanShift:
    """Tests for ZLevel Zeeman shifts."""

    @pytest.fixture
    def zeeman_atom(self):
        """Minimal I=0 atom with a single 2S1/2 level and explicit lande=2."""
        a = Atom(name="z", I=0)
        gs = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz, lande=2.0)
        a.add_level(gs)
        return a

    def test_zlevel_shift_zero_at_zero_B(self, zeeman_atom):
        """At B=0, all ZLevel Zeeman shifts should be zero."""
        gs = zeeman_atom.levels['1s2.2s 2S1/2']
        for zlevel in gs['F=1/2'].sublevels():
            assert zlevel.shift_hz_at(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_zlevel_shift_linear_in_B(self, zeeman_atom):
        """Shift at B=20G should be 2× shift at B=10G for the same nonzero mF sublevel."""
        gs = zeeman_atom.levels['1s2.2s 2S1/2']
        zlevel = gs['F=1/2']['mF=1/2']

        shift_10 = zlevel.shift_hz_at(10.0)
        shift_20 = zlevel.shift_hz_at(20.0)

        assert shift_10 != pytest.approx(0.0, abs=1e-10)  # sanity: nonzero
        assert shift_20 == pytest.approx(2.0 * shift_10, rel=1e-5)

    def test_zlevel_mF_sign(self, zeeman_atom):
        """mF=+1/2 and mF=-1/2 shifts should be equal in magnitude and opposite in sign."""
        gs = zeeman_atom.levels['1s2.2s 2S1/2']
        z_pos = gs['F=1/2']['mF=1/2']
        z_neg = gs['F=1/2']['mF=-1/2']
        assert z_pos.shift_hz_at(10.0) == pytest.approx(-z_neg.shift_hz_at(10.0), rel=1e-5)


# ---------------------------------------------------------------------------
# TestHyperfineShift
# ---------------------------------------------------------------------------

class TestHyperfineShift:
    """Tests for HFLevel hyperfine shifts.

    Uses the existing lvl fixture (parametrized over 3 level names), but
    only runs the shift-sign test for '1s2.2s 2S1/2' (J=1/2).
    """

    def test_hf_shift_sign(self, lvl, request):
        """F=1 HFLevel shift_Hz > F=0 HFLevel shift_Hz (positive hfA=1 GHz)."""
        if request.node.callspec.params.get('lvl') != '1s2.2s 2S1/2':
            pytest.skip("HF shift sign test only applies to the J=1/2 level")
        hf_F1 = lvl['F=1']
        hf_F0 = lvl['F=0']
        assert hf_F1.shift_Hz > hf_F0.shift_Hz

    def test_hf_shift_zero_for_J_half_no_quadrupole(self, lvl, request):
        """For J=1/2, hfB contribution (FE2 term) must be zero."""
        if request.node.callspec.params.get('lvl') != '1s2.2s 2S1/2':
            pytest.skip("Quadrupole zero test only applies to the J=1/2 level")
        # For J=1/2: J<=0.5 so FE2=0 in compute_hf_shift
        # Verify: setting hfB nonzero doesn't change the shift
        hf_F1 = lvl['F=1']
        shift_before = hf_F1.shift_Hz
        lvl.hfB = Q_(1.0, 'GHz')
        shift_after = hf_F1.shift_Hz
        assert shift_before == pytest.approx(shift_after, rel=1e-10)


# ---------------------------------------------------------------------------
# TestTransitionLinewidth
# ---------------------------------------------------------------------------

class TestTransitionLinewidth:
    """Tests for ZTransition A coefficient properties."""

    def test_ztransition_A_not_none(self, dummy_atom):
        """After accessing subtransitions, ZTransition.A is not None when parent has A."""
        tc = dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')]
        # Access subtransitions to force lazy population
        hft = tc[('F=1/2', 'F=3/2')]
        zt_list = list(hft.subtransitions())
        assert len(zt_list) > 0
        for zt in zt_list:
            assert zt.A is not None

    def test_ztransition_A_leq_parent(self, dummy_atom):
        """Each ZTransition.A should be ≤ the parent Transition.A."""
        tc = dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')]
        parent_A_Hz = tc.A.to(Hz).magnitude
        hft = tc[('F=1/2', 'F=3/2')]
        for zt in hft.subtransitions():
            zt_A_Hz = zt.A.to(Hz).magnitude
            assert zt_A_Hz <= parent_A_Hz + 1e-6  # allow float rounding

    def test_subtransitions_lazy(self, dummy_atom):
        """_subtransitions dict is empty before access and non-empty after t[...]."""
        tc = dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')]
        # Before access: internal dict should be empty
        assert len(tc._subtransitions) == 0
        # Trigger lazy computation
        _ = tc[('F=1/2', 'F=3/2')]
        # After access: internal dict should be populated
        assert len(tc._subtransitions) > 0


# ---------------------------------------------------------------------------
# TestSubtransitions
# ---------------------------------------------------------------------------

class TestSubtransitions:
    """Tests for subtransition counts."""

    def test_subtransitions_count(self, dummy_atom):
        """For I=0, J=1/2→J=3/2 E1 transition, only F=1/2→F=3/2 is allowed."""
        # dummy_atom has I=0
        # tc: gs (2S1/2, J=1/2) → e1 (2P3/2, J=3/2)
        tc = dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')]
        # For I=0: only F=J, so F_gs=1/2, F_e1=3/2 → only one HFTransition
        subtrans = tc.subtransitions()
        assert len(subtrans) == 1


# ---------------------------------------------------------------------------
# TestIO
# ---------------------------------------------------------------------------

class TestIO:
    """Tests for atomtoolkit.IO hyperfine CSV roundtrip."""

    def test_apply_hf_csv_roundtrip(self, tmp_path):
        """set hfA → generate_hf_csv → apply_hf_csv to fresh atom → hfA matches."""
        from atomtoolkit import IO, Q_

        # Build first atom and set a custom hfA
        a1 = Atom(name="io_test", I=0)
        gs1 = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz)
        a1.add_level(gs1)
        a1.levels['1s2.2s 2S1/2'].hfA = Q_(0.5, 'GHz')

        csv_path = tmp_path / "test_hf.csv"
        IO.generate_hf_csv(a1, str(csv_path))

        # Build a second identical atom with default hfA (0)
        a2 = Atom(name="io_test", I=0)
        gs2 = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz)
        a2.add_level(gs2)
        # Confirm it starts at 0
        assert a2.levels['1s2.2s 2S1/2'].hfA.to(Hz).magnitude == pytest.approx(0.0, abs=1e-10)

        # Apply the CSV
        IO.apply_hf_csv(a2, str(csv_path))

        # After applying, hfA should match what we set on a1
        applied_hz = a2.levels['1s2.2s 2S1/2'].hfA.to(Hz).magnitude
        expected_hz = Q_(0.5, 'GHz').to(Hz).magnitude
        assert applied_hz == pytest.approx(expected_hz, rel=1e-5)


# ---------------------------------------------------------------------------
# TestJSONSerialization
# ---------------------------------------------------------------------------

class TestJSONSerialization:
    """Tests for Atom.to_dict / from_dict / save_json / load_json."""

    @pytest.fixture
    def rich_atom(self):
        """
        Atom with I=1/2, multi-term level, hyperfine coefficients, lande,
        two transitions (one with alias), and quadratic Zeeman overrides.
        """
        a = Atom(name="test_json", I=0.5)

        gs = EnergyLevel(Term('1s2.2s', '2S', 0.5),
                         level=0.0 * Hz, hfA=1 * GHz, lande=2.002,
                         alias='ground')
        ex = EnergyLevel(Term('1s2.2p', '2P*', 1.5),
                         level=100 * THz, hfA=200 * MHz)

        # A multi-term level with two mixed configurations
        ex_mt = EnergyLevel(
            MultiTerm(
                Term('1s2.3d', '2D', 2.5, percentage=80.0),
                Term('1s2.4d', '2D', 2.5, percentage=20.0),
            ),
            level=200 * THz,
            hfA=50 * MHz,
        )

        a.add_level(gs)
        a.add_level(ex)
        a.add_level(ex_mt)

        t1 = a.add_transition(Transition(gs, ex, A=10 * MHz, alias='cooling'))
        t2 = a.add_transition(Transition(gs, ex_mt, A=5 * MHz))

        # Force-compute ZLevels so quadratic_zeeman survives to_dict
        a.levels['1s2.2s 2S1/2']['F=0']['mF=0'].quadratic_zeeman = -155.305
        a.levels['1s2.2s 2S1/2']['F=1']['mF=0'].quadratic_zeeman = 155.305

        return a

    # ------------------------------------------------------------------
    # to_dict structure
    # ------------------------------------------------------------------

    def test_schema_version_present(self, rich_atom):
        d = rich_atom.to_dict()
        assert '_schema' in d
        assert isinstance(d['_schema'], int)

    def test_metadata_round_trip(self, rich_atom):
        d = rich_atom.to_dict()
        assert d['name'] == 'test_json'
        assert d['I'] == pytest.approx(0.5)

    def test_level_count(self, rich_atom):
        d = rich_atom.to_dict()
        assert len(d['levels']) == 3

    def test_level_energy_stored(self, rich_atom):
        d = rich_atom.to_dict()
        energies = sorted(lvl['level_Hz'] for lvl in d['levels'])
        assert energies[0] == pytest.approx(0.0, abs=1.0)
        assert energies[1] == pytest.approx(100e12, rel=1e-6)

    def test_hyperfine_stored(self, rich_atom):
        d = rich_atom.to_dict()
        gs_data = next(l for l in d['levels'] if l['level_Hz'] == pytest.approx(0.0, abs=1.0))
        assert gs_data['hfA_Hz'] == pytest.approx(1e9, rel=1e-6)

    def test_lande_stored(self, rich_atom):
        d = rich_atom.to_dict()
        gs_data = next(l for l in d['levels'] if l['level_Hz'] == pytest.approx(0.0, abs=1.0))
        assert gs_data['lande'] == pytest.approx(2.002, rel=1e-4)

    def test_alias_stored(self, rich_atom):
        d = rich_atom.to_dict()
        gs_data = next(l for l in d['levels'] if l['level_Hz'] == pytest.approx(0.0, abs=1.0))
        assert gs_data['alias'] == 'ground'

    def test_multiterm_stored(self, rich_atom):
        d = rich_atom.to_dict()
        ex_alt = next(l for l in d['levels'] if l['level_Hz'] == pytest.approx(200e12, rel=1e-6))
        assert len(ex_alt['terms']) == 2
        pcts = sorted(t['percentage'] for t in ex_alt['terms'])
        assert pcts == pytest.approx([20.0, 80.0], rel=1e-6)

    def test_transition_count(self, rich_atom):
        d = rich_atom.to_dict()
        assert len(d['transitions']) == 2

    def test_transition_A_stored(self, rich_atom):
        d = rich_atom.to_dict()
        A_vals = sorted(t['A_Hz'] for t in d['transitions'])
        assert A_vals[0] == pytest.approx(5e6, rel=1e-6)
        assert A_vals[1] == pytest.approx(10e6, rel=1e-6)

    def test_transition_alias_stored(self, rich_atom):
        d = rich_atom.to_dict()
        cooling = next((t for t in d['transitions'] if t.get('alias') == 'cooling'), None)
        assert cooling is not None

    def test_quadratic_zeeman_stored(self, rich_atom):
        d = rich_atom.to_dict()
        gs_data = next(l for l in d['levels'] if l['level_Hz'] == pytest.approx(0.0, abs=1.0))
        qz = gs_data['quadratic_zeeman']
        assert 'F=0 mF=0' in qz
        assert qz['F=0 mF=0'] == pytest.approx(-155.305, rel=1e-5)
        assert 'F=1 mF=0' in qz
        assert qz['F=1 mF=0'] == pytest.approx(155.305, rel=1e-5)

    def test_self_transitions_excluded(self, rich_atom):
        d = rich_atom.to_dict()
        for t in d['transitions']:
            assert t['lower'] != t['upper']

    # ------------------------------------------------------------------
    # from_dict reconstruction
    # ------------------------------------------------------------------

    def test_from_dict_name(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        assert a2.name == rich_atom.name

    def test_from_dict_I(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        assert a2.I == pytest.approx(rich_atom.I)

    def test_from_dict_level_count(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        assert len(a2.levels) == len(rich_atom.levels)

    def test_from_dict_level_energies(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        orig = sorted(l.level_Hz for l in rich_atom.levels.values())
        loaded = sorted(l.level_Hz for l in a2.levels.values())
        for o, l in zip(orig, loaded):
            assert o == pytest.approx(l, rel=1e-9)

    def test_from_dict_hfA(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        gs2 = a2.levels['1s2.2s 2S1/2']
        assert gs2.hfA.to('GHz').magnitude == pytest.approx(1.0, rel=1e-6)

    def test_from_dict_lande(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        assert a2.levels['1s2.2s 2S1/2'].lande == pytest.approx(2.002, rel=1e-4)

    def test_from_dict_alias(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        # alias should resolve to the same level
        assert a2.levels['ground'].name == '1s2.2s 2S1/2'

    def test_from_dict_transition_A(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        t = a2.transitions[('1s2.2s 2S1/2', '1s2.2p 2P*3/2')]
        assert t.A.to('MHz').magnitude == pytest.approx(10.0, rel=1e-6)  # cooling transition

    def test_from_dict_transition_alias(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        # alias should resolve to the cooling transition
        t = a2.transitions.aliases.get('cooling')
        assert t is not None
        assert t.A.to('MHz').magnitude == pytest.approx(10.0, rel=1e-6)

    def test_from_dict_quadratic_zeeman(self, rich_atom):
        a2 = Atom.from_dict(rich_atom.to_dict())
        qz = a2.levels['1s2.2s 2S1/2']['F=0']['mF=0'].quadratic_zeeman
        assert qz == pytest.approx(-155.305, rel=1e-5)

    def test_from_dict_hf_sublevels_still_lazy(self, rich_atom):
        """Levels not in quadratic_zeeman should not be eagerly computed."""
        a2 = Atom.from_dict(rich_atom.to_dict())
        ex = a2.levels['1s2.2p 2P*3/2']   # cooling level — no quadratic_zeeman entries
        assert len(ex._sublevels) == 0   # not yet populated

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def test_save_load_json_roundtrip(self, rich_atom, tmp_path):
        path = str(tmp_path / "atom.json")
        rich_atom.save_json(path)
        a2 = Atom.load_json(path)
        assert a2.name == rich_atom.name
        assert len(a2.levels) == len(rich_atom.levels)

    def test_load_dispatches_to_json(self, rich_atom, tmp_path):
        """Atom.load() should auto-detect .json and use the JSON loader."""
        path = str(tmp_path / "atom.json")
        rich_atom.save_json(path)
        a2 = Atom.load(path)
        assert isinstance(a2, Atom)
        assert a2.name == rich_atom.name

    def test_json_extension_appended(self, rich_atom, tmp_path):
        path = str(tmp_path / "atom")   # no extension
        rich_atom.save_json(path)
        assert (tmp_path / "atom.json").exists()

    def test_json_is_valid_utf8_text(self, rich_atom, tmp_path):
        path = str(tmp_path / "atom.json")
        rich_atom.save_json(path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_no_A_transition_roundtrip(self, tmp_path):
        """Transitions with A=None should survive the roundtrip."""
        a = Atom(name="no_A", I=0)
        gs = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz)
        ex = EnergyLevel(Term('1s2.2p', '2P*', 0.5), level=50 * THz)
        a.add_level(gs)
        a.add_level(ex)
        a.add_transition(Transition(gs, ex))  # A=None

        path = str(tmp_path / "no_a.json")
        a.save_json(path)
        a2 = Atom.load_json(path)
        t = a2.transitions[('1s2.2s 2S1/2', '1s2.2p 2P*1/2')]  # E1 allowed, parity changes
        assert t.A is None
