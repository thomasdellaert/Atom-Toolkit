import pytest
from atomtoolkit.atom import Atom, Transition
from atomtoolkit import Hz, Q_

# class TestAtomCreation:
#     def test_creation(self):
#         import atomtoolkit.species.Yb_II_171
#         assert isinstance(atomtoolkit.species.Yb_II_171.Yb171, Atom)
#         import atomtoolkit.species.Yb_II_174
#         assert isinstance(atomtoolkit.species.Yb_II_174.Yb174, Atom)
#         import atomtoolkit.species.Yb_II_173
#         assert isinstance(atomtoolkit.species.Yb_II_173.Yb173, Atom)


class TestAtomComponents:
    yb174 = Atom.load('../species/Yb_II_174.atom')
    yb171 = Atom.load('../species/Yb_II_171.atom')
    yb173 = Atom.load('../species/Yb_II_173.atom')

    def test_instantiation(self):
        assert isinstance(self.yb171, Atom)
        assert isinstance(self.yb174, Atom)
        assert isinstance(self.yb173, Atom)

    def test_save(self):
        pass

    def test_add_level(self):
        pass

    def test_add_transition(self):
        t = Transition(self.yb171.levels['4f13.(2F*).6s2 2F*7/2'],
                       self.yb171.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*3/2'],
                       1e5*Hz)
        self.yb171.add_transition(t)
        assert self.yb171.transitions[('4f13.(2F*).6s2 2F*7/2', '4f13.(2F*).6s2 2F*7/2')] is not None
        t = Transition(self.yb173.levels['4f13.(2F*).6s2 2F*7/2'],
                       self.yb173.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*3/2'],
                       1e5 * Hz)
        self.yb173.add_transition(t)
        assert self.yb173.transitions[('4f13.(2F*).6s2 2F*7/2', '4f13.(2F*).6s2 2F*7/2')] is not None
        t = Transition(self.yb174.levels['4f13.(2F*).6s2 2F*7/2'],
                       self.yb174.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*3/2'],
                       1e5 * Hz)
        self.yb174.add_transition(t)
        assert self.yb174.transitions[('4f13.(2F*).6s2 2F*7/2', '4f13.(2F*).6s2 2F*7/2')] is not None

    def test_B(self):
        self.yb171.B = Q_(0.0, 'G')
        zero_field = self.yb171.levels['4f14.6s 2S1/2']['F=1']['mF=1'].level
        assert self.yb171.B == Q_(0.0, 'G')
        assert self.yb171.B_gauss == 0.0
        self.yb171.B = Q_(5.0, 'G')
        small_field = self.yb171.levels['4f14.6s 2S1/2']['F=1']['mF=1'].level
        assert self.yb171.B == Q_(5.0, 'G')
        assert self.yb171.B_gauss == 5.0
        assert small_field > zero_field

        self.yb174.B = Q_(0.0, 'G')
        zero_field = self.yb174.levels['4f14.6s 2S1/2']['F=1/2']['mF=1/2'].level
        assert self.yb174.B == Q_(0.0, 'G')
        assert self.yb174.B_gauss == 0.0
        self.yb174.B = Q_(5.0, 'G')
        small_field = self.yb174.levels['4f14.6s 2S1/2']['F=1/2']['mF=1/2'].level
        assert self.yb174.B == Q_(5.0, 'G')
        assert self.yb174.B_gauss == 5.0
        assert small_field > zero_field

    def test_linked_levels(self):
        assert len(self.yb171.linked_levels('4f14.6p 2P*3/2')) > 0
        assert len(self.yb173.linked_levels('4f14.6p 2P*3/2')) > 0
        assert len(self.yb174.linked_levels('4f14.6p 2P*3/2')) > 0

    def test_state_lifetime(self):
        assert self.yb171.state_lifetime('4f14.6p 2P*3/2') > 0
        assert self.yb173.state_lifetime('4f14.6p 2P*3/2') > 0
        assert self.yb174.state_lifetime('4f14.6p 2P*3/2') > 0

    def test_branching_ratios(self):
        assert len(self.yb171.compute_branching_ratios('4f14.6p 2P*3/2')) == 3
        assert len(self.yb173.compute_branching_ratios('4f14.6p 2P*3/2')) == 3
        assert len(self.yb174.compute_branching_ratios('4f14.6p 2P*3/2')) == 3

    def test_energy_level_access(self):
        assert len(self.yb171.levels['4f13.(2F*).6s2 2F*7/2'].sublevels()) == 2
        assert self.yb171.levels['4f14.6s 2S1/2']['F=1']['mF=1'] is not None
        assert self.yb174.levels['4f14.6s 2S1/2']['F=1/2']['mF=1/2'] is not None
        assert self.yb174.levels['4f14.6s 2S1/2']['mJ=1/2'] is not None

        assert self.yb171.transitions[('4f14.6s 2S1/2', '4f13.(2F*<7/2>).5d2.(3F) 3[3/2]*1/2')].A != 0

        assert self.yb171.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*3/2'].lande != 0
