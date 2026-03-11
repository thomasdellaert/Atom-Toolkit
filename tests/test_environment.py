"""
Tests for atomtoolkit/environment.py: Polarization and LightField.
"""

import numpy as np
import pytest

from atomtoolkit.environment import Polarization, LightField
from atomtoolkit import Q_, ureg


# ---------------------------------------------------------------------------
# TestPolarization
# ---------------------------------------------------------------------------

class TestPolarization:
    """Tests for the Polarization class."""

    def test_norm_is_one_after_init_real(self):
        """Passing a real list: xyz should be normalized to unit norm."""
        pol = Polarization([0, 0, 1])
        assert np.linalg.norm(pol.xyz) == pytest.approx(1.0, rel=1e-5)

    def test_norm_is_one_after_init_unnormalized(self):
        """Passing an unnormalized vector: should normalize."""
        pol = Polarization([3, 4, 0])
        assert np.linalg.norm(pol.xyz) == pytest.approx(1.0, rel=1e-5)

    def test_norm_is_one_after_init_complex(self):
        """Complex input (σ+): should normalize."""
        pol = Polarization([1, 1j, 0])
        assert np.linalg.norm(pol.xyz) == pytest.approx(1.0, rel=1e-5)

    def test_z_direction_stays_z(self):
        """[0,0,1] should remain [0,0,1] after normalization."""
        pol = Polarization([0, 0, 1])
        expected = np.array([0, 0, 1], dtype=complex)
        np.testing.assert_allclose(pol.xyz, expected, atol=1e-6)

    def test_xyz_is_complex_array(self):
        """The internal array should have complex dtype."""
        pol = Polarization([1, 0, 0])
        assert np.iscomplexobj(pol.xyz)

    def test_spherical_property_returns_array(self):
        """pol.spherical should return a numpy array."""
        pol = Polarization([0, 0, 1])
        assert isinstance(pol.spherical, np.ndarray)

    def test_pi_polarization_spherical_component(self):
        """[0,0,1] is π polarization: the 'pi' component (index 1) should be 1."""
        pol = Polarization([0, 0, 1])
        sph = pol.spherical
        # sigma- is index 0, pi is index 1, sigma+ is index 2
        assert abs(sph[1]) == pytest.approx(1.0, rel=1e-5)
        assert abs(sph[0]) == pytest.approx(0.0, abs=1e-6)
        assert abs(sph[2]) == pytest.approx(0.0, abs=1e-6)

    def test_pi_has_no_transverse_component(self):
        """π polarization [0,0,1] has zero q=±1 spherical components."""
        pol = Polarization([0, 0, 1])
        sph = pol.spherical
        assert abs(sph[0]) == pytest.approx(0.0, abs=1e-6)  # q=-1
        assert abs(sph[2]) == pytest.approx(0.0, abs=1e-6)  # q=+1

    def test_transverse_pol_has_no_pi_component(self):
        """Purely transverse polarization [1,0,0] has zero π (q=0) spherical component."""
        pol = Polarization([1, 0, 0])
        sph = pol.spherical
        assert abs(sph[1]) == pytest.approx(0.0, abs=1e-6)  # q=0 (π)

    def test_spherical_setter(self):
        """Setting spherical should update xyz consistently."""
        pol = Polarization([0, 0, 1])
        original_sph = pol.spherical.copy()
        # Re-assign via setter: should be idempotent
        pol.spherical = original_sph
        np.testing.assert_allclose(pol.xyz, Polarization.sigma_to_xyz(original_sph), atol=1e-6)

    def test_xyz_setter(self):
        """Setting xyz directly should update the internal state."""
        pol = Polarization([0, 0, 1])
        new_xyz = np.array([1.0, 0.0, 0.0], dtype=complex)
        pol.xyz = new_xyz
        np.testing.assert_allclose(pol.xyz, new_xyz, atol=1e-10)

    def test_from_numpy_array(self):
        """Should accept a numpy array as input."""
        pol = Polarization(np.array([0, 0, 1]))
        assert np.linalg.norm(pol.xyz) == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# TestLightField
# ---------------------------------------------------------------------------

class TestLightField:
    """Tests for the LightField class."""

    @pytest.fixture
    def pi_field(self):
        """A π-polarized field with a known intensity."""
        k = np.array([1.0, 0.0, 0.0])   # propagating along x
        intensity = Q_(1e4, 'W/m^2')
        pol = Polarization([0, 0, 1])    # π polarization (along z)
        return LightField(k=k, i=intensity, pol=pol)

    @pytest.fixture
    def pi_field_double(self):
        """Same field but 2× intensity."""
        k = np.array([1.0, 0.0, 0.0])
        intensity = Q_(2e4, 'W/m^2')
        pol = Polarization([0, 0, 1])
        return LightField(k=k, i=intensity, pol=pol)

    def test_E_returns_array_like(self, pi_field):
        """E() should return an array-like object (polarization × amplitude)."""
        E = pi_field.E()
        assert hasattr(E, '__len__') or hasattr(E, 'magnitude')

    def test_E_direction_parallel_to_polarization(self, pi_field):
        """For real π polarization, E should be parallel to [0,0,1]."""
        E = pi_field.E()
        # The magnitude of E; for pint quantities strip units by .magnitude
        try:
            E_mag = E.magnitude
        except AttributeError:
            E_mag = np.array(E)
        # E should point in z only
        E_arr = np.array(E_mag, dtype=complex)
        assert abs(E_arr[0]) == pytest.approx(0.0, abs=1e-20)
        assert abs(E_arr[1]) == pytest.approx(0.0, abs=1e-20)
        assert abs(E_arr[2]) > 0

    def test_E_squared_proportional_to_intensity(self, pi_field, pi_field_double):
        """|E|² should scale linearly with intensity: doubling I doubles |E|²."""
        E1 = pi_field.E()
        E2 = pi_field_double.E()
        try:
            E1_sq = float(np.linalg.norm(E1.magnitude) ** 2)
            E2_sq = float(np.linalg.norm(E2.magnitude) ** 2)
        except AttributeError:
            E1_sq = float(np.linalg.norm(E1) ** 2)
            E2_sq = float(np.linalg.norm(E2) ** 2)
        assert E2_sq == pytest.approx(2.0 * E1_sq, rel=1e-5)

    def test_E_magnitude_positive(self, pi_field):
        """The magnitude of E should be positive."""
        E = pi_field.E()
        try:
            norm = np.linalg.norm(E.magnitude)
        except AttributeError:
            norm = np.linalg.norm(E)
        assert float(norm.real) > 0

    def test_B_field_perpendicular_to_k(self, pi_field):
        """B = (1/c) k × E, so B ⊥ k."""
        B = pi_field.B()
        k = pi_field.k
        try:
            B_arr = np.array(B.magnitude, dtype=complex)
        except AttributeError:
            B_arr = np.array(B, dtype=complex)
        dot = abs(np.dot(B_arr, k))
        assert dot == pytest.approx(0.0, abs=1e-20)

    def test_B_field_perpendicular_to_E(self, pi_field):
        """B ⊥ E (EM wave transversality)."""
        E = pi_field.E()
        B = pi_field.B()
        try:
            E_arr = np.array(E.magnitude, dtype=complex)
            B_arr = np.array(B.magnitude, dtype=complex)
        except AttributeError:
            E_arr = np.array(E, dtype=complex)
            B_arr = np.array(B, dtype=complex)
        dot = abs(np.dot(E_arr, np.conj(B_arr)))
        assert dot == pytest.approx(0.0, abs=1e-25)

    def test_default_polarization_is_pi(self):
        """Default polarization is [0,0,1] (π)."""
        k = np.array([1.0, 0.0, 0.0])
        intensity = Q_(1e4, 'W/m^2')
        field = LightField(k=k, i=intensity)
        np.testing.assert_allclose(field.pol.xyz.real, np.array([0, 0, 1]), atol=1e-6)

    def test_intensity_stored(self, pi_field):
        """Intensity should be stored correctly."""
        assert pi_field.intensity.magnitude == pytest.approx(1e4, rel=1e-10)

    def test_k_stored(self, pi_field):
        """k vector should be stored correctly."""
        np.testing.assert_allclose(pi_field.k, np.array([1.0, 0.0, 0.0]))