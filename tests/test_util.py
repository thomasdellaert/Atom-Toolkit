import pytest
from atomtoolkit.atom import Term
from atomtoolkit import util

class TestTermParsing:
    def test_get_coupling_LS(self):
        term = Term('4f14.6s', '2S', '1/2')  # From Yb II
        assert util.get_term_coupling(term.term) == 'LS'
        term = Term('4f13.(2F*).6s2', '2F*', '5/2')  # From Yb II
        assert util.get_term_coupling(term.term) == 'LS'
        term = Term('3d8', '3F', '4')  # From Co II
        assert util.get_term_coupling(term.term) == 'LS'
        term = Term('5f3.6d.7s2', '2L', '6')  # From U I
        assert util.get_term_coupling(term.term) == 'LS'
        term = Term('5d10.5g', '2G', '9/2')  # From Hg II
        assert util.get_term_coupling(term.term) == 'LS'

    def test_get_coupling_JJ(self):
        term = Term('5d9.6s.6p', '(2D<5/2>,3P<0>)*', '5/2')  # from Hg II
        assert util.get_term_coupling(term.term) == 'JJ'
        term = Term('4f13.(2F*<7/2>).6s.6p.(3P*<1>)', '(7/2,1)', '9/2')  # from Yb II
        assert util.get_term_coupling(term.term) == 'JJ'
        term = Term('5d9.6p2', '(2D<5/2>,1D<2>)', '7/2')  # from Hg II
        assert util.get_term_coupling(term.term) == 'JJ'
        term = Term('4f9.(6H*).5d.(7H*<8>).6s.6p.(3P*<0>)', '(8,0)', '8')  # from NIST examples
        assert util.get_term_coupling(term.term) == 'JJ'
        term = Term('4f12.(3H<6>).5d.(2D).6s.6p.(3P*).(4F*<3/2>)', '(6, 3/2)*', '13/2')
        assert util.get_term_coupling(term.term) == 'JJ'
        term = Term('3d9.(2D<5/2>).4p<3/2>', ' (5/2, 3/2)*', '3')
        assert util.get_term_coupling(term.term) == 'JJ'
        term = Term('5f4<7/2>.5f5<5/2>.(8,5/2)*<21/2>.7p<3/2>', '(21/2, 3/2)', '10')
        assert util.get_term_coupling(term.term) == 'JJ'
        term = Term('5f3<7/2>.5f3<5/2>.(9/2, 9/2)<9>.7s.7p.(3P*<2>)', '(9,2)*', '7')
        assert util.get_term_coupling(term.term) == 'JJ'

    def test_get_coupling_LK(self):
        term = Term('3s2.3p.(2P*).4f', 'G 2[7/2]', '3')  # from NIST examples
        assert util.get_term_coupling(term.term) == 'LK'
        term = Term('3d7.(4P).4s.4p.(3P*)', 'D* 3[5/2]*', '7/2')
        assert util.get_term_coupling(term.term) == 'LK'
        # term = Term('3s<2>.3p.(2P*).4f', '5 2[7/2]', '3')
        # self.assertEqual(term.get_coupling(), 'LK')

    def test_get_coupling_JK(self):
        term = Term('3p5.(2P*<1/2>).5g', '2[9/2]*', '5')  # from NIST examples
        assert util.get_term_coupling(term.term) == 'JK'
        term = Term('4f2.(3H<4>).5g', '2[3]', '5/2')  # from NIST examples
        assert util.get_term_coupling(term.term) == 'JK'
        term = Term('4f13.(2F*<7/2>)5d2.(1D)', '1[7/2]*', '7/2')  # from NIST examples
        assert util.get_term_coupling(term.term) == 'JK'
        term = Term('4f13.(2F*<5/2>).5d.6s.(3D)', '3[9/2]*', '11/2')  # from NIST examples
        assert util.get_term_coupling(term.term) == 'JK'

    def test_get_quantum_nums_LS(self):
        conf, term = ('4f14.6s', '2S')  # From Yb II
        assert util.get_quantum_nums(conf, term) == (None, None, None, None, None, None, 0, 0.5, None)
        conf, term = ('4f13.(2F*).6s2', '2F*')  # From Yb II
        assert util.get_quantum_nums(conf, term) == (None, None, None, None, None, None, 3, 0.5, None)
        conf, term = ('3d8', '3F')  # From Co II
        assert util.get_quantum_nums(conf, term) == (None, None, None, None, None, None, 3, 1.0, None)
        conf, term = ('5f3.6d.7s2', '2L')  # From U I
        assert util.get_quantum_nums(conf, term) == (None, None, None, None, None, None, 8, 0.5, None)
        conf, term = ('5d10.5g', '2G')  # From Hg II
        assert util.get_quantum_nums(conf, term) == (None, None, None, None, None, None, 4, 0.5, None)

    def test_get_quantum_nums_JJ(self):
        conf, term = ('5d9.6s.6p', '(2D<5/2>,3P<0>)*')  # from Hg II
        assert util.get_quantum_nums(conf, term) == (2, 0.5, 1, 1.0, 2.5, 0.0, None, None, None)
        conf, term = ('4f13.(2F*<7/2>).6s.6p.(3P*<1>)', '(7/2,1)')  # from Yb II
        assert util.get_quantum_nums(conf, term) == (3, 0.5, 1, 1.0, 3.5, 1.0, None, None, None)
        conf, term = ('5d9.6p2', '(2D<5/2>,1D<2>)')  # from Hg II
        assert util.get_quantum_nums(conf, term) == (2, 0.5, 2, 0.0, 2.5, 2.0, None, None, None)
        conf, term = ('4f9.(6H*).5d.(7H*<8>).6s.6p.(3P*<0>)', '(8,0)')  # from NIST examples
        assert util.get_quantum_nums(conf, term) == (5, 3.0, 1, 1.0, 8.0, 0.0, None, None, None)
        conf, term = ('4f12.(3H<6>).5d.(2D).6s.6p.(3P*).(4F*<3/2>)', '(6, 3/2)*')
        assert util.get_quantum_nums(conf, term) == (5, 1.0, 3, 1.5, 6.0, 1.5, None, None, None)
        conf, term = ('3d9.(2D<5/2>).4p<3/2>', ' (5/2, 3/2)*')
        assert util.get_quantum_nums(conf, term) == (2, 0.5, 1, 0.0, 2.5, 1.5, None, None, None)
        conf, term = ('5f4<7/2>.5f5<5/2>.(8,5/2)*<21/2>.7p<3/2>', '(21/2, 3/2)')
        assert util.get_quantum_nums(conf, term) == (None, None, 1, 0.0, 10.5, 1.5, None, None, None)
        conf, term = ('5f3<7/2>.5f3<5/2>.(9/2, 9/2)<9>.7s.7p.(3P*<2>)', '(9,2)*')
        assert util.get_quantum_nums(conf, term) == (None, None, 1, 1.0, 9.0, 2.0, None, None, None)

    def test_get_quantum_nums_JK(self):
        conf, term = ('3p5.(2P*<1/2>).5g', '2[9/2]*')  # from NIST examples
        assert util.get_quantum_nums(conf, term) == (1, 0.5, 4, 0.5, 0.5, None, None, None, 4.5)
        conf, term = ('4f2.(3H<4>).5g', '2[3]')  # from NIST examples
        assert util.get_quantum_nums(conf, term) == (5, 1.0, 4, 0.5, 4.0, None, None, None, 3.0)
        conf, term = ('4f13.(2F*<7/2>).5d2.(1D)', '1[7/2]*')  # from NIST examples
        assert util.get_quantum_nums(conf, term) == (3, 0.5, 2, 0.0, 3.5, None, None, None, 3.5)
        conf, term = ('4f13.(2F*<5/2>).5d.6s.(3D)', '3[9/2]*')  # from NIST examples
        assert util.get_quantum_nums(conf, term) == (3, 0.5, 2, 1.0, 2.5, None, None, None, 4.5)

    def test_get_quantum_nums_LK(self):
        conf, term = ('3s2.3p.(2P*).4f', 'G 2[7/2]')  # from NIST examples
        assert util.get_quantum_nums(conf, term) == (1, 0.5, 3, 0.5, None, None, 4, None, 3.5)
        conf, term = ('3d7.(4P).4s.4p.(3P*)', 'D* 3[5/2]*')
        assert util.get_quantum_nums(conf, term) == (1, 1.5, 1, 1.0, None, None, 2, None, 2.5)

class TestReadableCouplings:
    pass  # TODO

class TestStringMethods:
    def test_l_methods(self):
        assert util.let_to_l('D') == 2
        assert util.let_to_l('') is None
        with pytest.raises(ValueError):
            util.let_to_l('SS')
        with pytest.raises(ValueError):
            util.let_to_l('J')
        with pytest.raises(ValueError):
            util.let_to_l('A')
        with pytest.raises(TypeError):
            util.let_to_l(2)

        assert util.l_to_let(2) == 'D'
        with pytest.raises(IndexError):
            util.l_to_let(58)
        with pytest.raises(TypeError):
            util.l_to_let('S')

    def test_frac_methods(self):
        assert util.frac_to_float('1/2') == 0.5
        assert util.frac_to_float('8') == 8.0
        assert util.frac_to_float('0') == 0.0
        assert util.frac_to_float('19/2') == 9.5
        assert util.frac_to_float('123') == 123.0
        assert util.frac_to_float('123/2') == 61.5
        assert util.frac_to_float('') is None
        assert util.frac_to_float('-1/2') == -0.5
        assert util.frac_to_float('1/3') == 1.0/3.0
        with pytest.raises(ValueError):
            util.frac_to_float('Blueberry')

        assert util.float_to_frac(1.5) == '3/2'
        assert util.float_to_frac(2) == '2'
        assert util.float_to_frac(13.5) == '27/2'
        assert util.float_to_frac(13.0) == '13'
        assert util.float_to_frac('13') == '13'
        assert util.float_to_frac('27/2') == '27/2'
        with pytest.raises(ValueError):
            util.float_to_frac('Blueberry')