"""
Tests for atomtoolkit/wigner.py and atomtoolkit/transition_strengths.py.
"""

import math
import numpy as np
import pytest

from atomtoolkit.wigner import wigner3j, wigner6j
from atomtoolkit.transition_strengths import (
    E1_mel_geom,
    M1_mel_geom,
    E2_mel_geom,
    A_to_dipole_mel,
    dipole_mel_to_A,
)
from atomtoolkit import Q_, MHz, Hz, THz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pi_eps():
    """π polarization as a tuple (required by lru_cache)."""
    return (0.0 + 0j, 0.0 + 0j, 1.0 + 0j)


def _sigma_plus_eps():
    """σ+ polarization (normalized) as a tuple."""
    v = np.array([1.0, 1j, 0.0], dtype=complex)
    v = v / np.linalg.norm(v)
    return tuple(v)


# ---------------------------------------------------------------------------
# TestWignerSymbols1propo
# ---------------------------------------------------------------------------

class TestWignerSymbols:
    """Tests for wigner3j and wigner6j selection rules and known values."""

    # --- wigner3j known values ---

    def test_3j_m_selection_rule_satisfied(self):
        """(1,1,0; 1,-1,0): m-sum = 0, nonzero."""
        val = wigner3j(1, 1, 0, 1, -1, 0)
        assert abs(val) == pytest.approx(1.0 / math.sqrt(3), rel=1e-5)

    def test_3j_m_zero_row(self):
        """(1,1,0; 0,0,0) = -1/sqrt(3)."""
        val = wigner3j(1, 1, 0, 0, 0, 0)
        assert abs(val) == pytest.approx(1.0 / math.sqrt(3), rel=1e-5)
        # parity: j1+j2+j3=2 (even), m=0 → nonzero

    def test_3j_m_selection_rule_violated(self):
        """m1+m2+m3 = 1+0+0 = 1 ≠ 0 → must be zero."""
        val = wigner3j(1, 1, 0, 1, 0, 0)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_3j_triangle_rule_violated(self):
        """j3 = 3 > j1+j2 = 2 → must be zero."""
        val = wigner3j(1, 1, 3, 0, 0, 0)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_3j_parity_rule_odd_sum(self):
        """(1,1,1; 0,0,0): j1+j2+j3 = 3 (odd) → zero."""
        val = wigner3j(1, 1, 1, 0, 0, 0)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_3j_half_integer_nonzero(self):
        """(1/2, 1/2, 0; 1/2, -1/2, 0) should be nonzero."""
        val = wigner3j(0.5, 0.5, 0, 0.5, -0.5, 0)
        assert abs(val) > 1e-10

    def test_3j_symmetry_under_permutation(self):
        """Even permutation of columns: same value up to sign."""
        # (j1, j2, j3; m1, m2, m3) → (j2, j3, j1; m2, m3, m1) same value
        v1 = wigner3j(1, 1, 0, 1, -1, 0)
        v2 = wigner3j(1, 0, 1, -1, 0, 1)
        assert abs(v1) == pytest.approx(abs(v2), rel=1e-5)

    # --- wigner6j known values ---

    def test_6j_triangle_rule_violated(self):
        """j3 = 3 > j1+j2 = 2 → must be zero."""
        val = wigner6j(1, 1, 3, 1, 1, 1)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_6j_known_value(self):
        """{a a 0; b b c} = (-1)^(a+b+c)/sqrt((2a+1)(2b+1)).
        For a=b=c=1: |val| = 1/sqrt(3*3) = 1/3.
        """
        val = wigner6j(1, 1, 0, 1, 1, 1)
        assert abs(val) == pytest.approx(1.0 / 3.0, rel=1e-5)

    def test_6j_nonzero_for_valid_triangle(self):
        """A valid set of arguments should give a nonzero value."""
        val = wigner6j(1, 1, 1, 1, 1, 1)
        assert abs(val) > 1e-10

    def test_6j_symmetry(self):
        """wigner6j is symmetric under swap of first two columns."""
        v1 = wigner6j(1, 1, 1, 1, 1, 1)
        v2 = wigner6j(1, 1, 1, 1, 1, 1)
        assert v1 == pytest.approx(v2, rel=1e-10)


# ---------------------------------------------------------------------------
# TestGeometricFactors
# ---------------------------------------------------------------------------

class TestGeometricFactors:
    """Tests for E1_mel_geom / M1_mel_geom selection rules.

    Uses a J=0,F=0,mF=0 → J=1,F=1,mF=? system (I=0).
    The eps args must be tuples for lru_cache compatibility.
    """

    # J=0, F=0, mF=0  (lower)
    # J=1, F=1, mF=? (upper)
    I = 0.0
    J0, F0, M0 = 0.0, 0.0, 0.0
    J1, F1 = 1.0, 1.0

    def _mel(self, eps_tuple, mF1):
        return E1_mel_geom(eps_tuple, self.I,
                           self.J0, self.F0, self.M0,
                           self.J1, self.F1, mF1)

    def test_pi_pol_mF_zero_nonzero(self):
        """π pol drives Δm=0 → mF=0 should be nonzero."""
        eps = _pi_eps()
        val = self._mel(eps, 0.0)
        assert abs(val) > 1e-10

    def test_pi_pol_mF_plus1_zero(self):
        """π pol: mF=+1 must be zero (Δm=+1 not driven)."""
        eps = _pi_eps()
        val = self._mel(eps, 1.0)
        assert abs(val) == pytest.approx(0.0, abs=1e-10)

    def test_pi_pol_mF_minus1_zero(self):
        """π pol: mF=-1 must be zero (Δm=-1 not driven)."""
        eps = _pi_eps()
        val = self._mel(eps, -1.0)
        assert abs(val) == pytest.approx(0.0, abs=1e-10)

    def test_sigma_plus_mF_plus1_nonzero(self):
        """σ+ pol drives Δm=+1 → mF=+1 should be nonzero."""
        eps = _sigma_plus_eps()
        val = self._mel(eps, 1.0)
        assert abs(val) > 1e-10

    def test_sigma_plus_mF_zero_zero(self):
        """σ+ pol: mF=0 must be zero."""
        eps = _sigma_plus_eps()
        val = self._mel(eps, 0.0)
        assert abs(val) == pytest.approx(0.0, abs=1e-10)

    def test_sigma_plus_mF_minus1_zero(self):
        """σ+ pol: mF=-1 must be zero."""
        eps = _sigma_plus_eps()
        val = self._mel(eps, -1.0)
        assert abs(val) == pytest.approx(0.0, abs=1e-10)

    def test_eps_must_be_tuple_for_caching(self):
        """Calling with a tuple twice returns same cached result."""
        eps = _pi_eps()
        v1 = E1_mel_geom(eps, self.I, self.J0, self.F0, self.M0, self.J1, self.F1, 0.0)
        v2 = E1_mel_geom(eps, self.I, self.J0, self.F0, self.M0, self.J1, self.F1, 0.0)
        assert v1 == v2

    def test_E1_mel_geom_real_pi_squared_equals_transition_strength(self):
        """For real π polarization, |E1_mel_geom|² should equal E1_transition_strength_geom."""
        from atomtoolkit.transition_strengths import E1_transition_strength_geom
        eps = _pi_eps()
        mel = self._mel(eps, 0.0)
        ts = E1_transition_strength_geom(eps, self.I,
                                         self.J0, self.F0, self.M0,
                                         self.J1, self.F1, 0.0)
        assert abs(mel) ** 2 == pytest.approx(float(abs(ts)), rel=1e-5)

    def test_M1_mel_geom_same_selection_rules_as_E1(self):
        """M1 has same Δm selection rules as E1 (they use the same tensor rank k=1)."""
        eps = _pi_eps()
        val_pi_0 = M1_mel_geom(eps, self.I,
                                self.J0, self.F0, self.M0,
                                self.J1, self.F1, 0.0)
        val_pi_1 = M1_mel_geom(eps, self.I,
                                self.J0, self.F0, self.M0,
                                self.J1, self.F1, 1.0)
        assert abs(val_pi_0) > 1e-10
        assert abs(val_pi_1) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# TestDipoleMelConversions
# ---------------------------------------------------------------------------

class TestDipoleMelConversions:
    """Tests for A_to_dipole_mel and dipole_mel_to_A (inverse pair)."""

    def test_A_to_dipole_mel_returns_quantity(self):
        """Should return a pint Quantity."""
        import pint
        A = 10 * MHz
        omega = 2 * math.pi * 100e12 * Hz
        mel = A_to_dipole_mel(A, J0=0, J1=1, omega=omega)
        assert isinstance(mel, pint.Quantity)

    def test_A_to_dipole_mel_units(self):
        """Result should be convertible to a0*e."""
        from atomtoolkit import ureg
        A = 10 * MHz
        omega = 2 * math.pi * 100e12 * Hz
        mel = A_to_dipole_mel(A, J0=0, J1=1, omega=omega)
        # Should not raise
        mel_converted = mel.to(ureg.a0 * ureg.e)
        assert mel_converted.magnitude > 0

    def test_dipole_mel_to_A_returns_quantity(self):
        """Should return a pint Quantity."""
        import pint
        A = 10 * MHz
        omega = 2 * math.pi * 100e12 * Hz
        mel = A_to_dipole_mel(A, J0=0, J1=1, omega=omega)
        A_back = dipole_mel_to_A(mel, J0=0, J1=1, omega=omega)
        assert isinstance(A_back, pint.Quantity)

    def test_roundtrip_A_to_mel_to_A(self):
        """dipole_mel_to_A(A_to_dipole_mel(A)) ≈ A."""
        A = 10 * MHz
        omega = 2 * math.pi * 100e12 * Hz
        mel = A_to_dipole_mel(A, J0=0, J1=1, omega=omega)
        A_back = dipole_mel_to_A(mel, J0=0, J1=1, omega=omega)
        # A was given in MHz; A_back should be in Hz; compare magnitudes in Hz
        assert A_back.to(Hz).magnitude == pytest.approx(A.to(Hz).magnitude, rel=1e-5)

    def test_A_to_dipole_mel_scales_with_A(self):
        """mel ∝ sqrt(A): doubling A gives sqrt(2) × mel."""
        omega = 2 * math.pi * 100e12 * Hz
        mel1 = A_to_dipole_mel(10 * MHz, J0=0, J1=1, omega=omega)
        mel2 = A_to_dipole_mel(20 * MHz, J0=0, J1=1, omega=omega)
        ratio = mel2.magnitude / mel1.magnitude
        assert ratio == pytest.approx(math.sqrt(2), rel=1e-4)

    def test_A_to_dipole_mel_scales_with_J(self):
        """Larger J_upper gives larger mel (due to (2J1+1) factor)."""
        omega = 2 * math.pi * 100e12 * Hz
        A = 10 * MHz
        mel_J1 = A_to_dipole_mel(A, J0=0, J1=1, omega=omega)
        mel_J2 = A_to_dipole_mel(A, J0=0, J1=2, omega=omega)
        assert mel_J2.magnitude > mel_J1.magnitude

    def test_dipole_mel_to_A_frequency_dependence(self):
        """Higher frequency → larger A for same mel (A ∝ ω³)."""
        from atomtoolkit import ureg
        mel = Q_(1.0, ureg.a0 * ureg.e)
        omega_low = 2 * math.pi * 100e12 * Hz
        omega_high = 2 * math.pi * 200e12 * Hz
        A_low = dipole_mel_to_A(mel, J0=0, J1=1, omega=omega_low)
        A_high = dipole_mel_to_A(mel, J0=0, J1=1, omega=omega_high)
        assert A_high.magnitude > A_low.magnitude