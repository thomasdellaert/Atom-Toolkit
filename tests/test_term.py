import itertools
import pytest
from atomtoolkit.term import Term, MultiTerm

term_args = [
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
]

@pytest.fixture(params=term_args)
def term(request):
    return Term(*request.param)

@pytest.fixture(params=term_args)
def term_2(request):
    return Term(*request.param)

@pytest.mark.usefixtures("term")
class TestTerm:
    def test_percentage(self, term):
        assert term.percentage == 100.0

    def test_inequality(self, term, term_2):
        if term.name != term_2.name:
            assert term != term_2
        else:
            assert term == term_2

    def test_copy(self, term):
        term_copy = term.make_term_copy()
        assert term == term_copy
        assert not(term is term_copy)

    def test_str(self, term):
        assert str(term) == term.term_name

    def test_repr(self, term):
        assert term.name in term.__repr__()


class TestMultiTerm:
    t1_100 = Term('4f14.6s', '2S', '1/2', percentage=100.0)
    t1_60 = Term('4f14.6s', '2S', '1/2', percentage=60.0)
    t2_40 = Term('4f14.6d', '2D', '5/2', percentage=40.0)

    m100 = MultiTerm(t1_100)
    m6040 = MultiTerm(t1_60, t2_40)

    def test_getattribute(self):
        assert self.t1_100.quantum_nums == self.m100.quantum_nums
        assert self.t1_100.name == self.m100.name
        assert self.m100 != self.t1_100              # __eq__ never crosses types
        assert self.m100.equal_to(self.t1_100, level='J')  # equal_to can
        assert self.m100.terms == [self.t1_100]
        assert self.m100.terms_dict == {100.0: self.t1_100}

        assert self.t1_60.quantum_nums == self.m6040.quantum_nums
        assert self.t1_60.name == self.m6040.name
        assert self.m6040 != self.t1_60              # __eq__ never crosses types
        assert self.m6040.equal_to(self.t1_60, level='J')  # equal_to can
        assert self.m6040.terms == [self.t1_60, self.t2_40]
        assert self.m6040.terms_dict == {60.0: self.t1_60, 40.0: self.t2_40}

    def test_len(self):
        assert len(self.m6040) == 2

    def test_getitem(self):
        assert self.m6040[0] == self.t1_60
        assert self.m6040[1] == self.t2_40
        assert self.m6040 != self.t1_60                     # __eq__ never crosses types
        assert self.m6040.equal_to(self.t1_60, level='J')  # equal_to can

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


class TestTermEqualitySemantics:
    """Targeted tests for Term/__eq__, __hash__, and equal_to semantics."""

    conf = '4f14.6s'
    term_str = '2S'

    # Base terms at J level
    t_J = Term(conf, term_str, '1/2')

    # F subterms
    t_F0 = Term(conf, term_str, '1/2', F=0)
    t_F1 = Term(conf, term_str, '1/2', F=1)

    # mF subterms (mF=0 is the important edge case)
    t_F1_mF0 = Term(conf, term_str, '1/2', F=1, mF=0)
    t_F1_mFm1 = Term(conf, term_str, '1/2', F=1, mF=-1)

    # Different J
    t_other_conf = Term('4f14.6d', '2D', '5/2')

    # MultiTerms wrapping the same leading term
    m_J = MultiTerm(t_J)
    m_F0 = MultiTerm(t_F0)

    # --- Term.__eq__ strict same-type ---

    def test_eq_same_term(self):
        assert self.t_J == Term(self.conf, self.term_str, '1/2')

    def test_eq_different_term(self):
        assert self.t_J != self.t_other_conf

    def test_eq_never_matches_multiterm(self):
        """Term.__eq__ must return NotImplemented (-> !=) for MultiTerm."""
        assert self.t_J != self.m_J

    def test_eq_multiterm_never_matches_term(self):
        """MultiTerm.__eq__ must return NotImplemented (-> !=) for Term."""
        assert self.m_J != self.t_J

    def test_eq_F_distinguishes(self):
        """Two Terms with same J but different F are not equal."""
        assert self.t_F0 != self.t_F1

    def test_eq_mF_zero_distinguishes(self):
        """mF=0 must not be treated as falsy — it still distinguishes terms."""
        assert self.t_F1_mF0 != self.t_F1_mFm1

    def test_eq_mF_zero_matches_self(self):
        t_copy = Term(self.conf, self.term_str, '1/2', F=1, mF=0)
        assert self.t_F1_mF0 == t_copy

    # --- __hash__ consistent with __eq__ ---

    def test_hash_equal_terms_equal_hash(self):
        t_copy = Term(self.conf, self.term_str, '1/2')
        assert hash(self.t_J) == hash(t_copy)

    def test_hash_usable_in_set(self):
        t_copy = Term(self.conf, self.term_str, '1/2')
        s = {self.t_J, t_copy}
        assert len(s) == 1

    def test_hash_F_subterms_differ(self):
        assert hash(self.t_F0) != hash(self.t_F1)

    def test_multiterm_hash_usable_in_set(self):
        m_copy = MultiTerm(Term(self.conf, self.term_str, '1/2'))
        s = {self.m_J, m_copy}
        assert len(s) == 1

    # --- Term.equal_to cross-type and granularity ---

    def test_equal_to_none_level_strict(self):
        """equal_to(level=None) is strict __eq__: rejects MultiTerm."""
        assert not self.t_J.equal_to(self.m_J)

    def test_equal_to_J_accepts_multiterm(self):
        assert self.t_J.equal_to(self.m_J, level='J')

    def test_equal_to_J_ignores_F(self):
        assert self.t_J.equal_to(self.t_F0, level='J')
        assert self.t_F0.equal_to(self.t_F1, level='J')

    def test_equal_to_F_respects_F(self):
        assert not self.t_F0.equal_to(self.t_F1, level='F')
        assert self.t_F0.equal_to(Term(self.conf, self.term_str, '1/2', F=0), level='F')

    def test_equal_to_F_ignores_mF(self):
        assert self.t_F1_mF0.equal_to(self.t_F1_mFm1, level='F')

    def test_equal_to_unknown_level_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown level"):
            self.t_J.equal_to(self.t_J, level='X')

    # --- MultiTerm.equal_to delegates to leading term ---

    def test_multiterm_equal_to_term_at_J(self):
        assert self.m_J.equal_to(self.t_J, level='J')

    def test_multiterm_equal_to_multiterm_at_J(self):
        m2 = MultiTerm(Term(self.conf, self.term_str, '1/2'))
        assert self.m_J.equal_to(m2, level='J')
