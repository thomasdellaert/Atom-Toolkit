import itertools

import pytest
from atomtoolkit.atom import Atom, Transition, EnergyLevel
from atomtoolkit.term import Term
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
    gs = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz)
    a.add_level(gs)
    e1 = EnergyLevel(Term('1s2.2p', '2P', 1.5), level=200 * THz)
    a.add_level(e1)
    e2 = EnergyLevel(Term('1s2.3d', '2D', 2.5), level=100 * THz)
    a.add_level(e2)
    tc = Transition(gs, e1, A=10 * MHz)
    a.add_transition(tc)
    tw = Transition(gs, e2, A=100 * Hz)
    a.add_transition(tw)
    td = Transition(e1, e2, A=5 * MHz)
    a.add_transition(td)
    return a

@pytest.fixture(params=['1s2.2s 2S1/2', '1s2.2p 2P3/2', '1s2.3d 2D5/2'])
def lvl(request):
    a = Atom(name="dummy", I=1/2)
    gs = EnergyLevel(Term('1s2.2s', '2S', 0.5), level=0.0 * Hz, hfA = 1*GHz)
    a.add_level(gs)
    e1 = EnergyLevel(Term('1s2.2p', '2P', 1.5), level=200 * THz, hfA = 1*GHz)
    a.add_level(e1)
    e2 = EnergyLevel(Term('1s2.3d', '2D', 2.5), level=100 * THz, lande=1.204, hfA = 1*GHz)
    a.add_level(e2)
    return a.levels[request.param]

class TestAtom:
    def test_init(self, dummy_atom):
        assert dummy_atom.name == 'dummy'
        assert dummy_atom.I == 0
        assert dummy_atom.B_gauss == 0

    def test_str(self, dummy_atom):
        assert str(dummy_atom) == 'dummy'

    def test_repr(self, dummy_atom):
        assert dummy_atom.__repr__().startswith('Atom')

    def test_add_level_and_transition(self, dummy_atom):
        d = EnergyLevel(Term('1s2.4s', '2S', 0.5), level=10.0 * THz)
        dummy_atom.add_level(d)
        assert d in dummy_atom.levels.values()

        td = Transition(d, dummy_atom.levels['1s2.2s 2S1/2'], A=50 * MHz)
        dummy_atom.add_transition(td)

    def test_B(self, dummy_atom):
        assert dummy_atom.B == Q_(0.0, 'G')
        dummy_atom.B = Q_(10.0, 'G')
        assert dummy_atom.B_gauss == 10.0
        assert dummy_atom.B == Q_(10.0, 'G')

    def test_save(self, dummy_atom):
        dummy_atom.save('dummy')
        dummy_atom.save('dummy.atom')

    def test_load(self):
        da = Atom.load('dummy.atom')
        assert da.name == 'dummy'

    def test_from_dataframe(self):
        pass #TODO

    def test_populate_transitions(self):
        pass #TODO

    def test_linked_levels(self, dummy_atom):
        assert len(dummy_atom.linked_levels('1s2.2s 2S1/2')) == 2
        assert len(dummy_atom.linked_levels('1s2.2p 2P3/2')) == 2
        assert len(dummy_atom.linked_levels('1s2.3d 2D5/2')) == 2

    def test_state_lifetime(self, dummy_atom):
        s = ureg.s
        assert dummy_atom.state_lifetime('1s2.2s 2S1/2').to(s).magnitude == np.inf
        assert abs(dummy_atom.state_lifetime('1s2.2p 2P3/2').to(s).magnitude - 4e-7) < 1e-7
        assert abs(dummy_atom.state_lifetime('1s2.3d 2D5/2').to(s).magnitude - 0.06) < 0.1

    def test_branching_ratios(self, dummy_atom):
        assert dummy_atom.compute_branching_ratios('1s2.2s 2S1/2') == {}
        assert dummy_atom.compute_branching_ratios('1s2.2p 2P3/2') == {'1s2.2s 2S1/2': 0.6666666666666666, '1s2.3d 2D5/2': 0.3333333333333333}
        assert dummy_atom.compute_branching_ratios('1s2.3d 2D5/2') == {'1s2.2s 2S1/2': 1.0}

class TestAtomAccess:
    def test_sublevels(self, dummy_atom):
        assert len(dummy_atom.levels['1s2.2s 2S1/2'].sublevels()) == 1
        assert len(dummy_atom.levels['1s2.2p 2P3/2'].sublevels()) == 1
        assert len(dummy_atom.levels['1s2.3d 2D5/2'].sublevels()) == 1

    def test_sublevel_access(self, dummy_atom):
        ways_of_calling = [
        dummy_atom.levels['1s2.2s 2S1/2']['F=1/2']['mF=1/2'],
        dummy_atom.levels['1s2.2s 2S1/2']['mJ=1/2'],
        dummy_atom.hflevels['1s2.2s 2S1/2 F=1/2']['mF=1/2'],
        dummy_atom.zlevels['1s2.2s 2S1/2 F=1/2 mF=1/2'],
        ]
        for a, b in itertools.product(ways_of_calling, ways_of_calling):
            assert a is b

    def test_sublevel_iteration(self, dummy_atom):
        assert len([l for l in dummy_atom.levels['1s2.2s 2S1/2']]) == 1
        assert len([l for l in dummy_atom.levels['1s2.2s 2S1/2']['F=1/2']]) == 2
        assert len([l for l in dummy_atom.levels['1s2.2p 2P3/2']]) == 1
        assert len([l for l in dummy_atom.levels['1s2.2p 2P3/2']['F=3/2']]) == 4

    def test_transition_access(self, dummy_atom):
        # CONSIDER: think about how transitions and things are indexed and named such that
        #  access is intuitive and makes sense
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')] is not None
        assert dummy_atom.transitions[('1s2.2p 2P3/2', '1s2.2s 2S1/2')] is not None
        assert dummy_atom.transitions[
                   ('1s2.2s 2S1/2', '1s2.2p 2P3/2')][
                   ('1s2.2s 2S1/2 F=1/2', '1s2.2p 2P3/2 F=3/2')] is not None
        assert dummy_atom.transitions[
                   ('1s2.2s 2S1/2', '1s2.2p 2P3/2')][
                   ('1s2.2s 2S1/2 F=1/2', '1s2.2p 2P3/2 F=3/2')][
                   ('1s2.2s 2S1/2 F=1/2 mF=1/2', '1s2.2p 2P3/2 F=3/2 mF=-1/2')
               ] is not None
        assert dummy_atom.hftransitions[('1s2.2s 2S1/2 F=1/2', '1s2.2p 2P3/2 F=3/2')] is not None
        assert dummy_atom.hftransitions[('1s2.2s 2S1/2 F=1/2', '1s2.2p 2P3/2 F=3/2')][
                   ('1s2.2s 2S1/2 F=1/2 mF=1/2', '1s2.2p 2P3/2 F=3/2 mF=-1/2')
               ] is not None
        assert dummy_atom.ztransitions[
                   ('1s2.2s 2S1/2 F=1/2 mF=1/2', '1s2.2p 2P3/2 F=3/2 mF=-1/2')] is not None

class TestAtomEditing:
    def test_etransition_freq_assignment(self, dummy_atom):
        dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].set_frequency(110*THz)
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 110*THz

        dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq = 115*THz
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 115*THz

    def test_hftransition_freq_assignment(self, dummy_atom):
        dummy_atom.hftransitions[('1s2.2s 2S1/2 F=1/2', '1s2.2p 2P3/2 F=3/2')].set_frequency(110 * THz)
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 110 * THz

    def test_ztransition_freq_assignment(self, dummy_atom):
        dummy_atom.ztransitions[('1s2.2s 2S1/2 F=1/2 mF=1/2', '1s2.2p 2P3/2 F=3/2 mF=-1/2')].set_frequency(110 * THz)
        assert dummy_atom.transitions[('1s2.2s 2S1/2', '1s2.2p 2P3/2')].freq == 110 * THz

    def test_elevel_level_assignment(self, dummy_atom):
        dummy_atom.levels['1s2.2p 2P3/2'].level = 110*THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 110*THz

    def test_hflevel_level_assignment(self, dummy_atom):
        dummy_atom.levels['1s2.2p 2P3/2']['F=3/2'].level = 110*THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 110*THz

        dummy_atom.hflevels['1s2.2p 2P3/2 F=3/2'].level = 120 * THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 120 * THz

    def test_zlevel_level_assignment(self, dummy_atom):
        dummy_atom.levels['1s2.2p 2P3/2']['F=3/2']['mF=1/2'].level = 110*THz
        assert dummy_atom.levels['1s2.2p 2P3/2'].level == 110*THz

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