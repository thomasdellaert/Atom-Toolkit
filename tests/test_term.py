import itertools
import pytest
from atomtoolkit.atom import Term, MultiTerm


class TestTerm:
    term_args = {
        ('3s2.3p.(2P*).4f', 'G 2[7/2]', '3'),
        ('3d7.(4P).4s.4p.(3P*)', 'D* 3[5/2]*', '7/2'),
        ('4f14.6s', '2S', '1/2'),
        ('4f13.(2F*).6s2', '2F*', '5/2'),
        ('3d8', '3F', '4'),
        ('5d10.5g', '2G', '9/2'),
        ('5d9.6s.6p', '(2D<5/2>,3P<0>)*', '5/2'),
        ('4f13.(2F*<7/2>).6s.6p.(3P*<1>)', '(7/2,1)', '9/2'),
        ('5d9.6p2', '(2D<5/2>,1D<2>)', '7/2'),
        ('4f9.(6H*).5d.(7H*<8>).6s.6p.(3P*<0>)', '(8,0)', '8'),
        ('4f12.(3H<6>).5d.(2D).6s.6p.(3P*).(4F*<3/2>)', '(6, 3/2)*', '13/2'),
        ('3d9.(2D<5/2>).4p<3/2>', ' (5/2, 3/2)*', '3'),
        ('5f4<7/2>.5f5<5/2>.(8,5/2)*<21/2>.7p<3/2>', '(21/2, 3/2)', '10'),
        ('5f3<7/2>.5f3<5/2>.(9/2, 9/2)<9>.7s.7p.(3P*<2>)', '(9,2)*', '7'),
        ('3p5.(2P*<1/2>).5g', '2[9/2]*', '5'),
        ('4f2.(3H<4>).5g', '2[3]', '5/2'),
        ('4f13.(2F*<7/2>)5d2.(1D)', '1[7/2]*', '7/2'),
        ('4f13.(2F*<5/2>).5d.6s.(3D)', '3[9/2]*', '11/2'),
    }
    terms = []

    @classmethod
    def setUpClass(cls) -> None:
        cls.terms = [Term(*ins) for ins in cls.term_args]

    def test_percentage(self):
        for term in TestTerm.terms:
            assert term.percentage == 100.0

    def test_inequality(self):
        for pair in itertools.product(TestTerm.terms, TestTerm.terms):
            if not(pair[0] is pair[1]):
                assert pair[0] != pair[1]
            else:
                assert pair[0] == pair[1]

    def test_copy(self):
        for term in TestTerm.terms:
            term2 = term.make_term_copy()
            assert term == term2
            assert not(term is term2)

    def test_str(self):
        for term in TestTerm.terms:
            assert str(term) == term.name

    def test_repr(self):
        for term in TestTerm.terms:
            assert term.name in term.__repr__


class TestMultiTerm:
    t1_100 = Term('4f14.6s', '2S', '1/2', percentage=100.0)
    t1_60 = Term('4f14.6s', '2S', '1/2', percentage=60.0)
    t2_40 = Term('4f14.6d', '2D', '5/2', percentage=40.0)

    m100 = MultiTerm(t1_100)
    m6040 = MultiTerm(t1_60, t2_40)

    def test_getattribute(self):
        assert self.t1_100.quantum_nums == self.m100.quantum_nums
        assert self.t1_100.name == self.m100.name
        assert self.m100 == self.t1_100
        assert self.m100.terms == [self.t1_100]
        assert self.m100.terms_dict == {100.0: self.t1_100}

        assert self.t1_60.quantum_nums == self.m6040.quantum_nums
        assert self.t1_60.name == self.m6040.name
        assert self.m6040 == self.t1_60
        assert self.m6040.terms == [self.t1_60, self.t2_40]
        assert self.m6040.terms_dict == {60.0: self.t1_60, 40.0: self.t2_40}

    def test_len(self):
        assert len(self.m6040) == 2

    def test_getitem(self):
        assert self.m6040[0] == self.t1_60
        assert self.m6040[1] == self.t2_40
        assert self.m6040 == self.t1_60

    def test_copy(self):
        m1 = self.m6040.make_term_copy()
        assert self.m6040 == m1
        assert self.m6040 is not m1
        for i in range(len(self.m6040)):
            assert self.m6040[i] == m1[i]
            assert self.m6040[i] is not m1[i]

        m2 = self.m6040.make_term_copy(F=1)
        assert m2[0].F == 1
        assert m2[1].F == 1

    def test_full_name(self):
        assert self.m100.full_name == '4f14.6s 2S1/2 (100.0%)'
        assert self.m6040.full_name == '4f14.6s 2S1/2 (60.0%)'
