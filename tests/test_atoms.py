import pytest
from atomtoolkit.atom import Atom
# from atomtoolkit.species import Yb_II_171, Yb_II_173, Yb_II_174

class TestAtoms:
    def test_atom(self):
        yb171 = Atom.load('../atomtoolkit/species/171Yb_II.atom')

        # assert len(yb171.compute_branching_ratios('4f14.6p 2P*3/2')) != 0
        #
        # assert len(yb171.levels['4f13.(2F*).6s2 2F*7/2'].sublevels()) != 0
        #
        # assert yb171.levels['4f14.6s 2S1/2']['F=1']['mF=1'].level > 0
        #
        # assert yb171.transitions[('4f14.6s 2S1/2', '4f13.(2F*<7/2>).5d2.(3F) 3[3/2]*1/2')].A != 0
        #
        # assert yb171.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*3/2'].lande != 0