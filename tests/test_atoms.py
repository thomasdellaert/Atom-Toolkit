import pytest
from atomtoolkit.atom import Atom

# class TestAtomCreation:
#     def test_creation(self):
#         import atomtoolkit.species.Yb_II_171
#         assert isinstance(atomtoolkit.species.Yb_II_171.Yb171, Atom)
#         import atomtoolkit.species.Yb_II_174
#         assert isinstance(atomtoolkit.species.Yb_II_174.Yb174, Atom)
#         import atomtoolkit.species.Yb_II_173
#         assert isinstance(atomtoolkit.species.Yb_II_173.Yb173, Atom)


class TestAtoms:
    yb174 = Atom.load('../species/Yb_II_174.atom')
    yb171 = Atom.load('../species/Yb_II_171.atom')
    yb173 = Atom.load('../species/Yb_II_173.atom')

    def test_instantiation(self):
        assert isinstance(self.yb171, Atom)
        assert isinstance(self.yb174, Atom)
        assert isinstance(self.yb173, Atom)

    def test_save(self):
        pass

    def test_atom(self):

        assert len(self.yb171.compute_branching_ratios('4f14.6p 2P*3/2')) == 3

        assert len(self.yb171.levels['4f13.(2F*).6s2 2F*7/2'].sublevels()) == 2

        assert self.yb171.levels['4f14.6s 2S1/2']['F=1']['mF=1'].level > 0

        assert self.yb171.transitions[('4f14.6s 2S1/2', '4f13.(2F*<7/2>).5d2.(3F) 3[3/2]*1/2')].A != 0

        assert self.yb171.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*3/2'].lande != 0