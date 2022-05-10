import pytest
from atomtoolkit.atom import Term
from atomtoolkit import util

exps = (
    {'conf':'4f14.6s'                                       , 'term':'2S'              , 'j':'1/2' , 'coupling':'LS', 'qnums':(None, None, None, None, None, None, 0, 0.5, None)},
    {'conf':'4f13.(2F*).6s2'                                , 'term':'2F*'             , 'j':'5/2' , 'coupling':'LS', 'qnums':(None, None, None, None, None, None, 3, 0.5, None)},
    {'conf':'3d8'                                           , 'term':'3F'              , 'j':'4'   , 'coupling':'LS', 'qnums':(None, None, None, None, None, None, 3, 1.0, None)},
    {'conf':'5f3.6d.7s2'                                    , 'term':'2L'              , 'j':'6'   , 'coupling':'LS', 'qnums':(None, None, None, None, None, None, 8, 0.5, None)},
    {'conf':'5d10.5g'                                       , 'term':'2G'              , 'j':'9/2' , 'coupling':'LS', 'qnums':(None, None, None, None, None, None, 4, 0.5, None)},
    {'conf':'5d9.6s.6p'                                     , 'term':'(2D<5/2>,3P<0>)*', 'j':'5/2' , 'coupling':'JJ', 'qnums':(2, 0.5, 1, 1.0, 2.5, 0.0, None, None, None)},
    {'conf':'4f13.(2F*<7/2>).6s.6p.(3P*<1>)'                , 'term':'(7/2,1)'         , 'j':'9/2' , 'coupling':'JJ', 'qnums':(3, 0.5, 1, 1.0, 3.5, 1.0, None, None, None)},
    {'conf':'5d9.6p2'                                       , 'term':'(2D<5/2>,1D<2>)' , 'j':'7/2' , 'coupling':'JJ', 'qnums':(2, 0.5, 2, 0.0, 2.5, 2.0, None, None, None)},
    {'conf':'4f9.(6H*).5d.(7H*<8>).6s.6p.(3P*<0>)'          , 'term':'(8,0)'           , 'j':'8'   , 'coupling':'JJ', 'qnums':(5, 3.0, 1, 1.0, 8.0, 0.0, None, None, None)},
    {'conf':'4f12.(3H<6>).5d.(2D).6s.6p.(3P*).(4F*<3/2>)'   , 'term':'(6, 3/2)*'       , 'j':'13/2', 'coupling':'JJ', 'qnums':(5, 1.0, 3, 1.5, 6.0, 1.5, None, None, None)},
    {'conf':'3d9.(2D<5/2>).4p<3/2>'                         , 'term':' (5/2, 3/2)*'    , 'j':'3'   , 'coupling':'JJ', 'qnums':(2, 0.5, 1, 0.0, 2.5, 1.5, None, None, None)},
    {'conf':'5f4<7/2>.5f5<5/2>.(8,5/2)*<21/2>.7p<3/2>'      , 'term':'(21/2, 3/2)'     , 'j':'10'  , 'coupling':'JJ', 'qnums':(None, None, 1, 0.0, 10.5, 1.5, None, None, None)},
    {'conf':'5f3<7/2>.5f3<5/2>.(9/2, 9/2)<9>.7s.7p.(3P*<2>)', 'term':'(9,2)*'          , 'j':'7'   , 'coupling':'JJ', 'qnums':(None, None, 1, 1.0, 9.0, 2.0, None, None, None)},
    {'conf':'3s2.3p.(2P*).4f'                               , 'term':'G 2[7/2]'        , 'j':'3'   , 'coupling':'LK', 'qnums':(1, 0.5, 3, 0.5, None, None, 4, None, 3.5)},
    {'conf':'3d7.(4P).4s.4p.(3P*)'                          , 'term':'D* 3[5/2]*'      , 'j':'7/2' , 'coupling':'LK', 'qnums':(1, 1.5, 1, 1.0, None, None, 2, None, 2.5)},
    {'conf':'3p5.(2P*<1/2>).5g'                             , 'term':'2[9/2]*'         , 'j':'5'   , 'coupling':'JK', 'qnums':(1, 0.5, 4, 0.5, 0.5, None, None, None, 4.5)},
    {'conf':'4f2.(3H<4>).5g'                                , 'term':'2[3]'            , 'j':'5/2' , 'coupling':'JK', 'qnums':(5, 1.0, 4, 0.5, 4.0, None, None, None, 3.0)},
    {'conf':'4f13.(2F*<7/2>)5d2.(1D)'                       , 'term':'1[7/2]*'         , 'j':'7/2' , 'coupling':'JK', 'qnums':(3, 0.5, 2, 0.0, 3.5, None, None, None, 3.5)},
    {'conf':'4f13.(2F*<5/2>).5d.6s.(3D)'                    , 'term':'3[9/2]*'         , 'j':'11/2', 'coupling':'JK' , 'qnums':(3, 0.5, 2, 1.0, 2.5, None, None, None, 4.5)},)

class TestTermParsing:
    @pytest.mark.parametrize(
        "test_input,expected",
        [(Term(d['conf'], d['term'], d['j']), d['coupling']) for d in exps])
    def test_get_coupling_LS(self, test_input, expected):
        assert util.get_term_coupling(test_input.term) == expected

    @pytest.mark.parametrize(
        "test_input,expected",
        [((d['conf'], d['term']), d['qnums']) for d in exps])
    def test_get_quantum_nums(self, test_input, expected):
        assert util.get_quantum_nums(*test_input) == expected

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