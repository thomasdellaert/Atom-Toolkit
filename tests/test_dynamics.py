"""
Tests for atomtoolkit/dynamics.py: generate_hamiltonian.

Requires qutip; the entire module is skipped if it is not installed.
"""

import math
import numpy as np
import pytest

qutip = pytest.importorskip('qutip')

from atomtoolkit.atom import Atom, EnergyLevel, Transition
from atomtoolkit.term import Term
from atomtoolkit.environment import Environment, LightField, Polarization
from atomtoolkit.dynamics import generate_hamiltonian
from atomtoolkit import Q_, MHz, Hz, THz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_e1_atom(with_A=True):
    """
    Minimal E1-allowed atom:
      gs: 1s2 1S0   (J=0, parity +1, level = 0)
      ex: 1s.2p 1P1* (J=1, parity -1, level = 100 THz)
    With I=0:
      - gs has 1 ZLevel  (mJ=0)
      - ex has 3 ZLevels (mJ=-1, 0, +1)
      Sorted by energy: idx 0=gs_mJ=0, 1=ex_mJ=-1, 2=ex_mJ=0, 3=ex_mJ=+1
    """
    a = Atom(name="test_e1", I=0)
    gs = EnergyLevel(Term('1s2', '1S', 0.0), level=0.0 * Hz)
    ex = EnergyLevel(Term('1s.2p', '1P*', 1.0), level=100 * THz)
    a.add_level(gs)
    a.add_level(ex)
    A_val = 10 * MHz if with_A else None
    t = a.add_transition(Transition(gs, ex, A=A_val))
    return a, gs, ex, t


def _pi_field():
    k = np.array([1.0, 0.0, 0.0])
    intensity = Q_(1e4, 'W/m^2')
    pol = Polarization([0, 0, 1])
    return LightField(k=k, i=intensity, pol=pol)


def _sigma_plus_field():
    k = np.array([0.0, 0.0, 1.0])
    intensity = Q_(1e4, 'W/m^2')
    pol = Polarization([1, 1j, 0])  # normalized internally
    return LightField(k=k, i=intensity, pol=pol)


@pytest.fixture
def e1_atom():
    a, gs, ex, t = _make_e1_atom(with_A=True)
    return a, gs, ex, t


@pytest.fixture
def e1_atom_no_A():
    a, gs, ex, t = _make_e1_atom(with_A=False)
    return a, gs, ex, t


# ---------------------------------------------------------------------------
# TestGenerateHamiltonian
# ---------------------------------------------------------------------------

class TestGenerateHamiltonian:

    # ------------------------------------------------------------------
    # H0 structure
    # ------------------------------------------------------------------

    def test_H0_shape(self, e1_atom):
        """H0 should be (n, n) for n = total z_states."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())  # both EnergyLevels
        H0, _ = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        # Count ZLevels: for I=0, each HFLevel expands to (2J+1) ZLevels
        n = sum(len(list(hfl.sublevels())) for hfl in gs.sublevels()) + \
            sum(len(list(hfl.sublevels())) for hfl in ex.sublevels())  # 1 + 3 = 4
        assert H0.shape == (n, n)

    def test_H0_diagonal(self, e1_atom):
        """H0 should be diagonal: all off-diagonal elements zero."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        H0, _ = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H0_arr = H0.full()
        n = H0_arr.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert H0_arr[i, j] == pytest.approx(0.0, abs=1e-6)

    def test_H0_ground_is_zero(self, e1_atom):
        """H0[0,0] should be zero (zeroed to lowest state)."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        H0, _ = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        assert H0.full()[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_H0_energies(self, e1_atom):
        """H0[3,3] - H0[0,0] should ≈ 2π * 100e12 rad/s."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        H0, _ = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H0_arr = H0.full()
        expected = 2 * math.pi * 100e12
        # Indices 1,2,3 are ex sublevels; all degenerate at B=0
        assert H0_arr[3, 3].real == pytest.approx(expected, rel=1e-5)

    def test_H0_hermitian(self, e1_atom):
        """H0 should be Hermitian (H0 == H0†)."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        H0, _ = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H0_arr = H0.full()
        np.testing.assert_allclose(H0_arr, H0_arr.conj().T, atol=1e-10)

    # ------------------------------------------------------------------
    # Coupling Hamiltonian structure
    # ------------------------------------------------------------------

    def test_coupling_shape_matches_H0(self, e1_atom):
        """Each coupling matrix should have the same shape as H0."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        H0, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        assert len(coupling_Hs) == 1
        assert coupling_Hs[0].shape == H0.shape

    def test_coupling_hermitian(self, e1_atom):
        """The coupling matrix should be Hermitian."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H_c = coupling_Hs[0].full()
        np.testing.assert_allclose(H_c, H_c.conj().T, atol=1e-10)

    def test_coupling_nonzero_for_E1(self, e1_atom):
        """At least one element should be nonzero when the transition has A."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H_c = coupling_Hs[0].full()
        assert np.any(np.abs(H_c) > 1e-6)

    def test_coupling_zero_without_A(self, e1_atom_no_A):
        """Without A coefficient, coupling matrix should be all zeros."""
        a, gs, ex, t = e1_atom_no_A
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H_c = coupling_Hs[0].full()
        np.testing.assert_allclose(np.abs(H_c), 0.0, atol=1e-10)

    # ------------------------------------------------------------------
    # Selection rule tests (π polarization)
    # ------------------------------------------------------------------

    def test_pi_pol_zero_for_delta_m_nonzero(self, e1_atom):
        """π pol: H_coupling[0,1] and H_coupling[0,3] must be zero (Δm≠0)."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H_c = coupling_Hs[0].full()
        # idx 0: gs mJ=0, idx 1: ex mJ=-1, idx 3: ex mJ=+1
        assert H_c[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert H_c[0, 3] == pytest.approx(0.0, abs=1e-10)

    def test_pi_pol_nonzero_for_delta_m_zero(self, e1_atom):
        """π pol: H_coupling[0,2] must be nonzero (Δm=0)."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        H_c = coupling_Hs[0].full()
        # idx 0: gs mJ=0, idx 2: ex mJ=0
        assert abs(H_c[0, 2]) > 1e-6

    # ------------------------------------------------------------------
    # Selection rule tests (σ+ polarization)
    # ------------------------------------------------------------------

    def test_sigma_plus_zero_for_delta_m_nonzero(self, e1_atom):
        """σ+ pol: H_coupling[0,1] (Δm=-1) and H_coupling[0,2] (Δm=0) must be zero."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_sigma_plus_field()]))
        H_c = coupling_Hs[0].full()
        assert H_c[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert H_c[0, 2] == pytest.approx(0.0, abs=1e-10)

    def test_sigma_plus_nonzero_for_delta_m_plus1(self, e1_atom):
        """σ+ pol: H_coupling[0,3] (Δm=+1) must be nonzero."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states, Environment(fields=[_sigma_plus_field()]))
        H_c = coupling_Hs[0].full()
        # idx 3: ex mJ=+1
        assert abs(H_c[0, 3]) > 1e-6

    # ------------------------------------------------------------------
    # State expansion
    # ------------------------------------------------------------------

    def test_expand_energy_level_to_zlevels(self, e1_atom):
        """Passing EnergyLevel objects should expand to all ZLevels."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())  # [gs, ex]
        H0, _ = generate_hamiltonian(a, all_states, Environment(fields=[_pi_field()]))
        # 1 gs ZLevel + 3 ex ZLevels = 4
        assert H0.shape == (4, 4)

    def test_empty_states_raises(self, e1_atom):
        """Passing an empty states list should raise ValueError."""
        a, gs, ex, t = e1_atom
        with pytest.raises(ValueError):
            generate_hamiltonian(a, [], Environment(fields=[_pi_field()]))

    # ------------------------------------------------------------------
    # Multiple fields
    # ------------------------------------------------------------------

    def test_multiple_fields(self, e1_atom):
        """Two fields should produce two independent coupling matrices."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        H0, coupling_Hs = generate_hamiltonian(a, all_states,
                                               Environment(fields=[_pi_field(), _sigma_plus_field()]))
        assert len(coupling_Hs) == 2
        assert coupling_Hs[0].shape == H0.shape
        assert coupling_Hs[1].shape == H0.shape

    def test_two_fields_are_independent(self, e1_atom):
        """The two coupling matrices from different polarizations should differ."""
        a, gs, ex, t = e1_atom
        all_states = list(a.levels.values())
        _, coupling_Hs = generate_hamiltonian(a, all_states,
                                              Environment(fields=[_pi_field(), _sigma_plus_field()]))
        H_pi = coupling_Hs[0].full()
        H_sp = coupling_Hs[1].full()
        # They should NOT be identical
        assert not np.allclose(H_pi, H_sp, atol=1e-10)