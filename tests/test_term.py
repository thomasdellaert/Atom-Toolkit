import itertools
from unittest import TestCase
from atomtoolkit.atom import Term, MultiTerm


class TestTermParsing(TestCase):
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

    def test_get_quantum_nums_LS(self):
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

    def test_get_quantum_nums_JJ(self):
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

    def test_get_quantum_nums_JK(self):
        term = Term('3p5.(2P*<1/2>).5g', '2[9/2]*', '5')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (1, 0.5, 4, 0.5, 0.5, None, None, None, 4.5))
        term = Term('4f2.(3H<4>).5g', '2[3]', '5/2')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (5, 1.0, 4, 0.5, 4.0, None, None, None, 3.0))
        term = Term('4f13.(2F*<7/2>).5d2.(1D)', '1[7/2]*', '7/2')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (3, 0.5, 2, 0.0, 3.5, None, None, None, 3.5))
        term = Term('4f13.(2F*<5/2>).5d.6s.(3D)', '3[9/2]*', '11/2')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (3, 0.5, 2, 1.0, 2.5, None, None, None, 4.5))

    def test_get_quantum_nums_LK(self):
        term = Term('3s2.3p.(2P*).4f', 'G 2[7/2]', '3')  # from NIST examples
        self.assertEqual(term.get_quantum_nums(), (1, 0.5, 3, 0.5, None, None, 4, None, 3.5))
        term = Term('3d7.(4P).4s.4p.(3P*)', 'D* 3[5/2]*', '7/2')
        self.assertEqual(term.get_quantum_nums(), (1, 1.5, 1, 1.0, None, None, 2, None, 2.5))

class TestTerm(TestCase):
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
            self.assertEqual(term.percentage, 100.0)

    def test_inequality(self):
        for pair in itertools.product(TestTerm.terms, TestTerm.terms):
            if not(pair[0] is pair[1]):
                self.assertNotEqual(*pair)
            else:
                self.assertEqual(*pair)

    def test_copy(self):
        for term in TestTerm.terms:
            term2 = term.make_term_copy()
            self.assertTrue(term == term2)
            self.assertFalse(term is term2)

    def test_l_methods(self):
        self.assertEqual(Term.let_to_l('D'), 2)
        self.assertIsNone(Term.let_to_l(''))
        with self.assertRaises(ValueError):
            Term.let_to_l('SS')
        with self.assertRaises(ValueError):
            Term.let_to_l('J')
        with self.assertRaises(ValueError):
            Term.let_to_l('A')
        with self.assertRaises(TypeError):
            Term.let_to_l(2)

        self.assertEqual(Term.l_to_let(2), 'D')
        with self.assertRaises(IndexError):
            Term.l_to_let(58)
        with self.assertRaises(TypeError):
            Term.l_to_let('S')

    def test_frac_methods(self):
        self.assertEqual(Term.frac_to_float('1/2'), 0.5)
        self.assertEqual(Term.frac_to_float('8'), 8.0)
        self.assertEqual(Term.frac_to_float('0'), 0.0)
        self.assertEqual(Term.frac_to_float('19/2'), 9.5)
        self.assertEqual(Term.frac_to_float('123'), 123.0)
        self.assertEqual(Term.frac_to_float('123/2'), 61.5)
        self.assertIsNone(Term.frac_to_float(''))
        self.assertEqual(Term.frac_to_float('-1/2'), -0.5)
        self.assertEqual(Term.frac_to_float('1/3'), 1.0/3.0)
        with self.assertRaises(ValueError):
            Term.frac_to_float('Blueberry')

        self.assertEqual(Term.float_to_frac(1.5), '3/2')
        self.assertEqual(Term.float_to_frac(2), '2')
        self.assertEqual(Term.float_to_frac(13.5), '27/2')
        self.assertEqual(Term.float_to_frac(13.0), '13')
        self.assertEqual(Term.float_to_frac('13'), '13')
        self.assertEqual(Term.float_to_frac('27/2'), '27/2')
        with self.assertRaises(ValueError):
            Term.float_to_frac('Blueberry')

class TestMultiTerm(TestCase):
    def setUp(self) -> None:
        self.t1_100 = Term('4f14.6s', '2S', '1/2', percentage=100.0)
        self.t1_60 = Term('4f14.6s', '2S', '1/2', percentage=60.0)
        self.t2_40 = Term('4f14.6d', '2D', '5/2', percentage=40.0)

    def test_getattribute(self):
        m = MultiTerm(self.t1_100)
        self.assertEqual(self.t1_100.quantum_nums, m.quantum_nums)
        self.assertEqual(self.t1_100.name, m.name)
        self.assertEqual(m, self.t1_100)
        self.assertEqual(m.terms, [self.t1_100])
        self.assertEqual(m.terms_dict, {100.0: self.t1_100})

        m = MultiTerm(self.t1_60, self.t2_40)
        self.assertEqual(self.t1_60.quantum_nums, m.quantum_nums)
        self.assertEqual(self.t1_60.name, m.name)
        self.assertEqual(m, self.t1_60)
        self.assertEqual(m.terms, [self.t1_60, self.t2_40])
        self.assertEqual(m.terms_dict, {60.0: self.t1_60, 40.0: self.t2_40})

    def test_len(self):
        m = MultiTerm(self.t1_60, self.t2_40)
        self.assertEqual(len(m), 2)

    def test_getitem(self):
        m = MultiTerm(self.t1_60, self.t2_40)
        self.assertEqual(m[0], self.t1_60)
        self.assertEqual(m[1], self.t2_40)
        self.assertEqual(m, self.t1_60)

    def test_copy(self):
        m = MultiTerm(self.t1_60, self.t2_40)
        m1 = m.make_term_copy()
        self.assertEqual(m, m1)
        self.assertIsNot(m, m1)
        for i in range(len(m)):
            self.assertEqual(m[i], m1[i])
            self.assertIsNot(m[i], m1[i])

        m2 = m.make_term_copy(F=1)
        self.assertEqual(m2[0].F, 1)
        self.assertEqual(m2[1].F, 1)