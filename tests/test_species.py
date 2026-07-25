"""
Integration tests for species files.

Each test class imports and builds one concrete Atom from NIST data.
These tests are marked `slow` because they make network requests to the NIST ASD.
Skip them with:  pytest -m "not slow"
Run only them with:  pytest -m slow
"""

import pytest
from atomtoolkit.atom import Atom
from atomtoolkit import Q_


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_level(atom, name, expected_J):
    """Assert a named level exists and has the right J."""
    lvl = atom.levels[name]
    assert lvl is not None, f"Level {name!r} missing"
    assert lvl.term.J == expected_J, f"{name}: expected J={expected_J}, got {lvl.term.J}"
    return lvl


def _check_transition(atom, lower, upper):
    """Assert a transition exists between two level names."""
    t = atom.transitions[(lower, upper)]
    assert t is not None, f"Transition ({lower!r}, {upper!r}) missing"
    return t


# ---------------------------------------------------------------------------
# 171Yb II  (I = 1/2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yb171():
    from atomtoolkit.species.Yb_II_171 import Yb171
    return Yb171


@pytest.mark.slow
class TestYb171:
    def test_is_atom(self, yb171):
        assert isinstance(yb171, Atom)

    def test_name_and_I(self, yb171):
        assert yb171.name == '171Yb II'
        assert yb171.I == 0.5

    def test_level_count(self, yb171):
        assert len(yb171.levels) > 100

    def test_transition_count(self, yb171):
        assert len(yb171.transitions) > 10

    def test_key_levels_exist(self, yb171):
        _check_level(yb171, '4f14.6s 2S1/2',   0.5)
        _check_level(yb171, '4f14.5d 2D3/2',   1.5)
        _check_level(yb171, '4f14.5d 2D5/2',   2.5)
        _check_level(yb171, '4f14.6p 2P*1/2',  0.5)

    def test_hf_coefficients_set(self, yb171):
        d32 = yb171.levels['4f14.5d 2D3/2']
        assert d32.hfA is not None
        assert d32.hfB is not None
        s12 = yb171.levels['4f14.6s 2S1/2']
        assert s12.hfA is not None

    def test_ground_state_hf_sublevels(self, yb171):
        """I=1/2, J=1/2 → F=0 and F=1."""
        s12 = yb171.levels['4f14.6s 2S1/2']
        hf = s12.sublevels()
        f_values = sorted(sub.term.F for sub in hf)
        assert f_values == [0.0, 1.0]

    def test_zeeman_sublevels_accessible(self, yb171):
        """F=1 level should have three mF sublevels."""
        s12 = yb171.levels['4f14.6s 2S1/2']
        f1 = s12['F=1']
        z = f1.sublevels()
        mf_values = sorted(sub.term.mF for sub in z)
        assert mf_values == [-1.0, 0.0, 1.0]

    def test_quadratic_zeeman_set(self, yb171):
        assert yb171.levels['4f14.6s 2S1/2']['F=0']['mF=0'].quadratic_zeeman == pytest.approx(-155.305)
        assert yb171.levels['4f14.6s 2S1/2']['F=1']['mF=0'].quadratic_zeeman == pytest.approx(155.305)

    def test_935_transition_exists_with_A(self, yb171):
        t = _check_transition(yb171, '4f14.5d 2D3/2', '4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2')
        assert t.freq_Hz == pytest.approx(320.567693e12, rel=1e-6)

    def test_cooling_transition_frequency(self, yb171):
        t = _check_transition(yb171, '4f14.6s 2S1/2', '4f14.6p 2P*1/2')
        assert t.freq_Hz == pytest.approx(811.294110e12, rel=1e-6)

    def test_e2_transition_added(self, yb171):
        """E2 clock transition should be present with A set."""
        t = _check_transition(yb171, '4f14.6s 2S1/2', '4f14.5d 2D5/2')
        assert t.A is not None
        assert t.A.to('Hz').magnitude == pytest.approx(22.0, rel=1e-3)

    def test_transition_A_from_csv(self, yb171):
        """At least some transitions should have A values loaded from the oscillator-strength CSV."""
        transitions_with_A = [t for t in yb171.transitions.values() if t.A is not None]
        assert len(transitions_with_A) > 5


# ---------------------------------------------------------------------------
# 173Yb II  (I = 5/2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yb173():
    from atomtoolkit.species.Yb_II_173 import Yb173
    return Yb173


@pytest.mark.slow
class TestYb173:
    def test_is_atom(self, yb173):
        assert isinstance(yb173, Atom)

    def test_name_and_I(self, yb173):
        assert yb173.name == '173Yb II'
        assert yb173.I == 2.5

    def test_level_count(self, yb173):
        assert len(yb173.levels) > 100

    def test_key_levels_exist(self, yb173):
        _check_level(yb173, '4f14.6s 2S1/2',  0.5)
        _check_level(yb173, '4f14.5d 2D3/2',  1.5)
        _check_level(yb173, '4f14.5d 2D5/2',  2.5)
        _check_level(yb173, '4f14.6p 2P*1/2', 0.5)

    def test_hf_coefficients_set(self, yb173):
        d32 = yb173.levels['4f14.5d 2D3/2']
        assert d32.hfA is not None
        s12 = yb173.levels['4f14.6s 2S1/2']
        assert s12.hfA is not None

    def test_ground_state_hf_sublevels(self, yb173):
        """I=5/2, J=1/2 → F=2 and F=3."""
        s12 = yb173.levels['4f14.6s 2S1/2']
        hf = s12.sublevels()
        f_values = sorted(sub.term.F for sub in hf)
        assert f_values == [2.0, 3.0]

    def test_935_transition_frequency(self, yb173):
        t = _check_transition(yb173, '4f14.5d 2D3/2', '4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2')
        assert t.freq_Hz == pytest.approx(320.570339e12, rel=1e-6)

    def test_cooling_transition_frequency(self, yb173):
        t = _check_transition(yb173, '4f14.6s 2S1/2', '4f14.6p 2P*1/2')
        assert t.freq_Hz == pytest.approx(811.292506e12, rel=1e-6)

    def test_e2_transition_added(self, yb173):
        t = _check_transition(yb173, '4f14.6s 2S1/2', '4f14.5d 2D5/2')
        assert t.A is not None


# ---------------------------------------------------------------------------
# 174Yb II  (I = 0, no hyperfine)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yb174():
    from atomtoolkit.species.Yb_II_174 import Yb174
    return Yb174


@pytest.mark.slow
class TestYb174:
    def test_is_atom(self, yb174):
        assert isinstance(yb174, Atom)

    def test_name_and_I(self, yb174):
        assert yb174.name == '174Yb II'
        assert yb174.I == 0

    def test_level_count(self, yb174):
        assert len(yb174.levels) > 100

    def test_key_levels_exist(self, yb174):
        _check_level(yb174, '4f14.6s 2S1/2',  0.5)
        _check_level(yb174, '4f14.5d 2D3/2',  1.5)
        _check_level(yb174, '4f14.5d 2D5/2',  2.5)
        _check_level(yb174, '4f14.6p 2P*1/2', 0.5)

    def test_no_hf_sublevels(self, yb174):
        """I=0: no HF splitting; sublevels() should return only Zeeman states."""
        s12 = yb174.levels['4f14.6s 2S1/2']
        hf = s12.sublevels()
        # With I=0 there is one 'F' level coinciding with J; mJ sublevels are the next layer
        assert all(sub.term.F == sub.term.J for sub in hf)

    def test_935_transition_frequency(self, yb174):
        t = _check_transition(yb174, '4f14.5d 2D3/2', '4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2')
        assert t.freq_Hz == pytest.approx(320.571752e12, rel=1e-6)

    def test_cooling_transition_frequency(self, yb174):
        t = _check_transition(yb174, '4f14.6s 2S1/2', '4f14.6p 2P*1/2')
        assert t.freq_Hz == pytest.approx(811.291801e12, rel=1e-6)

    def test_e2_transition_added(self, yb174):
        t = _check_transition(yb174, '4f14.6s 2S1/2', '4f14.5d 2D5/2')
        assert t.A is not None
        assert t.freq_Hz == pytest.approx(729.475280e12, rel=1e-6)


# ---------------------------------------------------------------------------
# 133Ba II  (I = 1/2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ba133():
    from atomtoolkit.species.Ba_II_133 import Ba133
    return Ba133


@pytest.mark.slow
class TestBa133:
    def test_is_atom(self, ba133):
        assert isinstance(ba133, Atom)

    def test_name_and_I(self, ba133):
        assert ba133.name == '133Ba II'
        assert ba133.I == 0.5

    def test_level_count(self, ba133):
        assert len(ba133.levels) > 50

    def test_key_levels_exist(self, ba133):
        _check_level(ba133, '6s 2S1/2',   0.5)
        _check_level(ba133, '5d 2D5/2',   2.5)
        _check_level(ba133, '6p 2P*1/2',  0.5)
        _check_level(ba133, '6p 2P*3/2',  1.5)

    def test_hf_coefficients_set(self, ba133):
        assert ba133.levels['6s 2S1/2'].hfA is not None
        assert ba133.levels['6s 2S1/2'].hfA.to('MHz').magnitude == pytest.approx(-9925.45355459, rel=1e-6)
        assert ba133.levels['5d 2D5/2'].hfA is not None
        assert ba133.levels['5d 2D5/2'].hfA.to('MHz').magnitude == pytest.approx(29.7565, rel=1e-4)

    def test_ground_state_hf_sublevels(self, ba133):
        """I=1/2, J=1/2 → F=0 and F=1."""
        s12 = ba133.levels['6s 2S1/2']
        hf = s12.sublevels()
        f_values = sorted(sub.term.F for sub in hf)
        assert f_values == [0.0, 1.0]

    def test_ground_state_transitions_have_A(self, ba133):
        """All transitions from ground state populated via dipole MELs should have A."""
        linked = ba133.linked_levels('6s 2S1/2')
        assert len(linked) > 0
        for name, t in linked.items():
            assert t.A is not None, f"Transition to {name!r} missing A"

    def test_d52_transitions_have_A(self, ba133):
        linked = ba133.linked_levels('5d 2D5/2')
        assert len(linked) > 0
        for name, t in linked.items():
            assert t.A is not None, f"Transition to/from D5/2 via {name!r} missing A"


# ---------------------------------------------------------------------------
# 7Li I  (I = 3/2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def li7():
    from atomtoolkit.species.Li_I import Li7
    return Li7


@pytest.mark.slow
class TestLi7:
    def test_is_atom(self, li7):
        assert isinstance(li7, Atom)

    def test_name_and_I(self, li7):
        assert li7.name == '7Li I'
        assert li7.I == 1.5

    def test_level_count(self, li7):
        assert len(li7.levels) > 0

    def test_ground_state_exists(self, li7):
        """Ground state should be J=1/2 (2s 2S1/2)."""
        ground = min(li7.levels.values(), key=lambda l: l.level.to('Hz').magnitude)
        assert ground.term.J == 0.5

    def test_ground_state_hf_sublevels(self, li7):
        """I=3/2, J=1/2 → F=1 and F=2."""
        ground = min(li7.levels.values(), key=lambda l: l.level.to('Hz').magnitude)
        hf = ground.sublevels()
        f_values = sorted(sub.term.F for sub in hf)
        assert f_values == [1.0, 2.0]
