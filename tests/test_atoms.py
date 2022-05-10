import pytest
from atomtoolkit.atom import Atom, Transition, EnergyLevel, Term
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

# TODO: test energy level methods, transition methods

#     def test_energy_level_access(self):
#         assert len(self.yb171.levels['4f13.(2F*).6s2 2F*7/2'].sublevels()) == 2
#         assert self.yb171.levels['4f14.6s 2S1/2']['F=1']['mF=1'] is not None
#         assert self.yb174.levels['4f14.6s 2S1/2']['F=1/2']['mF=1/2'] is not None
#         assert self.yb174.levels['4f14.6s 2S1/2']['mJ=1/2'] is not None
#
#         assert self.yb171.transitions[('4f14.6s 2S1/2', '4f13.(2F*<7/2>).5d2.(3F) 3[3/2]*1/2')].A != 0
#
#         assert self.yb171.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*3/2'].lande != 0
