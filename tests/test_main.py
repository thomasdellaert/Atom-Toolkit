from unittest import TestCase
from main import Term


class TestTerm(TestCase):
    def test_get_coupling_LS(self):
        term = Term('4f14.6s', '2S', '1/2')  # From Yb II
        self.assertEqual(term.get_coupling(), 'LS')
        term = Term('4f13.(2F*).6s2', '2F*', '5/2')  # From Yb II
        self.assertEqual(term.get_coupling(), 'LS')
        term = Term('3d8', '3F', '4')  # From Co II
        self.assertEqual(term.get_coupling(), 'LS')
        term = Term('5f3.6d.7s2', '2L', '6')  # From U I
        self.assertEqual(term.get_coupling(), 'LS')
        term = Term('5d10.5g', '2G', '9/2')  # From Hg II
        self.assertEqual(term.get_coupling(), 'LS')

    def test_get_coupling_JJ(self):
        term = Term('5d9.6s.6p', '(2D<5/2>,3P<0>)*', '5/2')  # from Hg II
        self.assertEqual(term.get_coupling(), 'JJ')
        term = Term('4f13.(2F*<7/2>).6s.6p.(3P*<1>)', '(7/2,1)', '9/2')  # from Yb II
        self.assertEqual(term.get_coupling(), 'JJ')
        term = Term('5d9.6p2', '(2D<5/2>,1D<2>)', '7/2')  # from Hg II
        self.assertEqual(term.get_coupling(), 'JJ')
        term = Term('4f9.(6H*).5d.(7H*<8>).6s.6p.(3P*<0>)', '(8,0)', '8')  # from NIST examples
        self.assertEqual(term.get_coupling(), 'JJ')
        term = Term('4f12.(3H<6>).5d.(2D).6s.6p.(3P*).(4F*<3/2>)', '(6, 3/2)*', '13/2')
        self.assertEqual(term.get_coupling(), 'JJ')
        term = Term('3d9.(2D<5/2>).4p<3/2>', ' (5/2, 3/2)*', '3')
        self.assertEqual(term.get_coupling(), 'JJ')
        term = Term('5f4<7/2>.5f5<5/2>.(8,5/2)*<21/2>.7p<3/2>', '(21/2, 3/2)', '10')
        self.assertEqual(term.get_coupling(), 'JJ')
        term = Term('5f3<7/2>.5f3<5/2>.(9/2, 9/2)<9>.7s.7p.(3P*<2>)', '(9,2)*', '7')
        self.assertEqual(term.get_coupling(), 'JJ')

    def test_get_coupling_LK(self):
        term = Term('3s2.3p.(2P*).4f', 'G 2[7/2]', '3')  # from NIST examples
        self.assertEqual(term.get_coupling(), 'LK')
        term = Term('3d7.(4P).4s.4p.(3P*)', 'D* 3[5/2]*', '7/2')
        self.assertEqual(term.get_coupling(), 'LK')
        # term = Term('3s<2>.3p.(2P*).4f', '5 2[7/2]', '3')
        # self.assertEqual(term.get_coupling(), 'LK')

    def test_get_coupling_JK(self):
        term = Term('3p5.(2P*<1/2>).5g', '2[9/2]*', '5')  # from NIST examples
        self.assertEqual(term.get_coupling(), 'JK')
        term = Term('4f2.(3H<4>).5g', '2[3]', '5/2')  # from NIST examples
        self.assertEqual(term.get_coupling(), 'JK')
        term = Term('4f13.(2F*<7/2>)5d2.(1D)', '1[7/2]*', '7/2')  # from NIST examples
        self.assertEqual(term.get_coupling(), 'JK')
        term = Term('4f13.(2F*<5/2>).5d.6s.(3D)', '3[9/2]*', '11/2')  # from NIST examples
        self.assertEqual(term.get_coupling(), 'JK')

    def test_get_quantum_nums(self):
        # LS terms
        term = Term('4f14.6s', '2S', '1/2')  # From Yb II
        self.assertEqual(term.get_quantum_nums(), (None, None, None, None, None, None, 0, 0.5, None))
        term = Term('4f13.(2F*).6s2', '2F*', '5/2')  # From Yb II
        self.assertEqual(term.get_quantum_nums(), (None, None, None, None, None, None, 3, 0.5, None))
        term = Term('3d8', '3F', '4')  # From Co II
        self.assertEqual(term.get_quantum_nums(), (None, None, None, None, None, None, 3, 1.0, None))
        term = Term('5f3.6d.7s2', '2L', '6')  # From U I
        self.assertEqual(term.get_quantum_nums(), (None, None, None, None, None, None, 8, 0.5, None))
        term = Term('5d10.5g', '2G', '9/2')  # From Hg II
        self.assertEqual(term.get_quantum_nums(), (None, None, None, None, None, None, 4, 0.5, None))
        # JJ terms
        term = Term('5d9.6s.6p', '(2D<5/2>,3P<0>)*', '5/2')  # from Hg II
        self.assertEqual(term.get_quantum_nums(), (2, 0.5, 1, 1.0, 2.5, 0.0, None, None, None))
        term = Term('4f13.(2F*<7/2>).6s.6p.(3P*<1>)', '(7/2,1)', '9/2')  # from Yb II
        self.assertEqual(term.get_quantum_nums(), (3, 0.5, 1, 1.0, 3.5, 1.0, None, None, None))
        term = Term('5d9.6p2', '(2D<5/2>,1D<2>)', '7/2')  # from Hg II
        self.assertEqual(term.get_quantum_nums(), (2, 0.5, 2, 0.0, 2.5, 2.0, None, None, None))
        term = Term('4f9.(6H*).5d.(7H*<8>).6s.6p.(3P*<0>)', '(8,0)', '8')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (5, 3.0, 1, 1.0, 8.0, 0.0, None, None, None))
        term = Term('4f12.(3H<6>).5d.(2D).6s.6p.(3P*).(4F*<3/2>)', '(6, 3/2)*', '13/2')
        self.assertEqual(term.get_quantum_nums(), (5, 1.0, 3, 1.5, 6.0, 1.5, None, None, None))
        term = Term('3d9.(2D<5/2>).4p<3/2>', ' (5/2, 3/2)*', '3')
        self.assertEqual(term.get_quantum_nums(), (2, 0.5, 1, 0.0, 2.5, 1.5, None, None, None))
        term = Term('5f4<7/2>.5f5<5/2>.(8,5/2)*<21/2>.7p<3/2>', '(21/2, 3/2)', '10')
        self.assertEqual(term.get_quantum_nums(), (None, None, 1, 0.0, 10.5, 1.5, None, None, None))
        term = Term('5f3<7/2>.5f3<5/2>.(9/2, 9/2)<9>.7s.7p.(3P*<2>)', '(9,2)*', '7')
        self.assertEqual(term.get_quantum_nums(), (None, None, 1, 1.0, 9.0, 2.0, None, None, None))
        # JK terms
        term = Term('3p5.(2P*<1/2>).5g', '2[9/2]*', '5')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (1, 0.5, 4, 0.5, 0.5, None, None, None, 4.5))
        term = Term('4f2.(3H<4>).5g', '2[3]', '5/2')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (5, 1.0, 4, 0.5, 4.0, None, None, None, 3.0))
        term = Term('4f13.(2F*<7/2>).5d2.(1D)', '1[7/2]*', '7/2')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (3, 0.5, 2, 0.0, 3.5, None, None, None, 3.5))
        term = Term('4f13.(2F*<5/2>).5d.6s.(3D)', '3[9/2]*', '11/2')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (3, 0.5, 2, 1.0, 2.5, None, None, None, 4.5))
        # LK terms
        term = Term('3s2.3p.(2P*).4f', 'G 2[7/2]', '3')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (1, 0.5, 3, 0.5, None, None, 4, None, 3.5))
        term = Term('3d7.(4P).4s.4p.(3P*)', 'D* 3[5/2]*', '7/2')
        self.assertEqual(term.get_quantum_nums(), (1, 1.5, 1, 1.0, None, None, 2, None, 2.5))
