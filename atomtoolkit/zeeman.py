"""
Nonlinear Zeeman effect via exact diagonalization of H_HF + H_Z.

Public API
----------
diagonalize_hfz(level, B_gauss)
    Returns {mF: (eigenvalues_Hz, eigenvectors)} for each mF block.

diagonalize_hfz_batch(level, B_array)
    Vectorized version returning {mF: eigenvalues_array (shape n_B x n_states)}.

zeeman_sweep(level, B_array)
    Returns {(F, mF): ndarray} of absolute energies in Hz over a B array.
    This is the recommended high-level API for plotting Zeeman diagrams.

breit_rabi(level, F, mF, B_gauss)
    Exact analytic Breit-Rabi energy for J=1/2 levels (Hz).

level_hz_nonlinear(zlevel, B_gauss)
    Nonlinear Zeeman energy for a single ZLevel (Hz).
    Uses spline LUT if precomputed, otherwise falls back to cached diagonalization.

Physics
-------
The full Hamiltonian in the |F, mF> coupled basis is:
    H = H_HF + H_Z
    H_HF is diagonal: E_F = hfA * FM1 + hfB * FE2 + hfC * FM3
    H_Z = mu_B * B * (gJ * J_z - gI * (mu_N/mu_B) * I_z)

Because [H, F_z] = 0, mF is a conserved quantum number and the matrix
block-diagonalises into blocks of fixed mF. Each block is diagonalised
with numpy.linalg.eigh. Adiabatic correspondence with the zero-field
|F, mF> states is established by sorting eigenvalues at B=0 by ascending
HF energy.
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING

import numpy as np

from . import mu_B, mu_N, Hz, G

if TYPE_CHECKING:
    from .atom import EnergyLevel, ZLevel

_MU_B_HZ_G: float = float(mu_B.to(Hz / G).magnitude)
_MU_N_HZ_G: float = float(mu_N.to(Hz / G).magnitude)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hf_shift_hz(J: float, I: float, F: float,
                 hfA_Hz: float, hfB_Hz: float, hfC_Hz: float) -> float:
    """Hyperfine energy shift in Hz for the state with total angular momentum F."""
    IdotJ = 0.5 * (F * (F + 1) - J * (J + 1) - I * (I + 1))
    FM1 = IdotJ
    FE2 = 0.0
    FM3 = 0.0
    if J > 0.5 and I > 0.5:
        FE2 = (3 * IdotJ ** 2 + 1.5 * IdotJ - I * (I + 1) * J * (J + 1)) / \
              (2.0 * I * (2.0 * I - 1.0) * J * (2.0 * J - 1.0))
    if J > 1 and I > 1:
        FM3 = (10 * IdotJ ** 3 + 20 * IdotJ ** 2 + 2 * IdotJ * (
                -3 * I * (I + 1) * J * (J + 1) + I * (I + 1) + J * (J + 1) + 3)
               - 5 * I * (I + 1) * J * (J + 1)) / (I * (I - 1) * J * (J - 1) * (2 * J - 1))
    return hfA_Hz * FM1 + hfB_Hz * FE2 + hfC_Hz * FM3


def _jz_mel(J: float, I: float, F: float, mF: float, Fp: float) -> float:
    """<F, mF | J_z | Fp, mF> matrix element in the |F, mF> coupled basis.

    Derived via the Wigner-Eckart theorem:
      <F,mF|J_z|F',mF> = (-1)^(F-mF) * W3j(F,1,F';-mF,0,mF)
                         * (-1)^(J+I+F'+1) * sqrt[(2F+1)(2F'+1)]
                         * W6j(J,F,I;F',J,1) * sqrt[J(J+1)(2J+1)]
    """
    from .wigner import wigner3j, wigner6j
    if I == 0.0:
        # No HF structure: F = J, matrix is diagonal with eigenvalue mF
        return float(mF) if abs(F - Fp) < 1e-9 else 0.0
    phase = (-1.0) ** (F - mF) * (-1.0) ** (J + I + Fp + 1)
    w3 = float(wigner3j(F, 1, Fp, -mF, 0, mF))
    w6 = float(wigner6j(J, F, I, Fp, J, 1))
    return phase * w3 * w6 * np.sqrt((2 * F + 1) * (2 * Fp + 1)) * np.sqrt(J * (J + 1) * (2 * J + 1))


def _iz_mel(J: float, I: float, F: float, mF: float, Fp: float) -> float:
    """<F, mF | I_z | Fp, mF> matrix element in the |F, mF> coupled basis.

    Same as _jz_mel but with J <-> I swapped in the reduced matrix element.
    """
    from .wigner import wigner3j, wigner6j
    if I == 0.0:
        return 0.0
    # Phase uses F (the bra), not Fp (the ket): this tensor acts on the
    # second (I) subspace, so the recoupling phase picks up F not F'.
    phase = (-1.0) ** (F - mF) * (-1.0) ** (J + I + F + 1)
    w3 = float(wigner3j(F, 1, Fp, -mF, 0, mF))
    w6 = float(wigner6j(I, F, J, Fp, I, 1))
    return phase * w3 * w6 * np.sqrt((2 * F + 1) * (2 * Fp + 1)) * np.sqrt(I * (I + 1) * (2 * I + 1))


def _mF_blocks(J: float, I: float) -> dict[float, list[float]]:
    """Return {mF: [F values containing that mF]}, with F values ascending."""
    blocks: dict[float, list[float]] = {}
    for F in np.arange(abs(J - I), J + I + 1):
        for mF in np.arange(-F, F + 1):
            key = round(float(mF) * 2) / 2
            blocks.setdefault(key, [])
            if not any(abs(F - f) < 1e-9 for f in blocks[key]):
                blocks[key].append(float(F))
    for key in blocks:
        blocks[key].sort()
    return blocks


# ---------------------------------------------------------------------------
# Core cached diagonalization
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def _diagonalize_hfz_cached(
        B_gauss: float,
        J: float, I: float,
        gJ: float, gI: float,
        hfA_Hz: float, hfB_Hz: float, hfC_Hz: float,
        level_Hz: float,
) -> dict:
    """Diagonalize H_HF + H_Z at a single B_gauss value.

    All arguments are plain Python scalars so they are hashable and
    can be cached with functools.lru_cache.

    Returns
    -------
    dict mapping mF (float) -> (eigenvalues_Hz, eigenvectors):
        eigenvalues_Hz : 1-D ndarray, absolute energies in Hz, ascending order.
        eigenvectors   : 2-D ndarray (n x n), columns are eigenstates in the
                         |F, mF> basis with F values ascending.
    """
    blocks = _mF_blocks(J, I)
    result = {}
    for mF, F_list in blocks.items():
        n = len(F_list)
        H = np.zeros((n, n), dtype=float)
        for i, Fi in enumerate(F_list):
            H[i, i] += _hf_shift_hz(J, I, Fi, hfA_Hz, hfB_Hz, hfC_Hz)
            for j, Fj in enumerate(F_list):
                jz = _jz_mel(J, I, Fi, mF, Fj)
                iz = _iz_mel(J, I, Fi, mF, Fj)
                H[i, j] += B_gauss * (gJ * _MU_B_HZ_G * jz - gI * _MU_N_HZ_G * iz)
        eigenvalues, eigvecs = np.linalg.eigh(H)
        result[mF] = (eigenvalues + level_Hz, eigvecs)
    return result


# ---------------------------------------------------------------------------
# Public diagonalization API
# ---------------------------------------------------------------------------

def diagonalize_hfz(level: EnergyLevel, B_gauss: float) -> dict:
    """Diagonalize H_HF + H_Z for *level* at *B_gauss* gauss.

    Returns
    -------
    dict mapping mF (float) -> (eigenvalues_Hz, eigenvectors)
        eigenvalues_Hz : absolute energies in Hz (ascending)
        eigenvectors   : columns are eigenstates in |F, mF> basis (F ascending)
    """
    atom = level.atom
    gI = getattr(atom, 'gI', 0.0)
    return _diagonalize_hfz_cached(
        float(B_gauss),
        float(level.term.J), float(atom.I),
        float(level.lande), float(gI),
        float(level.hfA_Hz), float(level.hfB_Hz), float(level.hfC_Hz),
        float(level.level_Hz),
    )


def diagonalize_hfz_batch(level: EnergyLevel, B_array) -> dict:
    """Vectorized diagonalization over an array of B values.

    Uses the lru_cache-backed :func:`diagonalize_hfz` for each element so
    repeated B values are free.

    Returns
    -------
    dict mapping mF (float) -> ndarray of shape (len(B_array), n_states_in_block)
        Each row is the ascending eigenvalues (absolute Hz) at that B.
    """
    B_array = np.asarray(B_array, dtype=float)
    first = diagonalize_hfz(level, float(B_array[0]))
    batch = {mF: np.empty((len(B_array), vals.shape[0]))
             for mF, (vals, _) in first.items()}
    for mF in first:
        batch[mF][0] = first[mF][0]
    for k in range(1, len(B_array)):
        res = diagonalize_hfz(level, float(B_array[k]))
        for mF in first:
            batch[mF][k] = res[mF][0]
    return batch


# ---------------------------------------------------------------------------
# Breit-Rabi analytic formula (J=1/2 only)
# ---------------------------------------------------------------------------

def breit_rabi(level: EnergyLevel, F: float, mF: float, B_gauss: float) -> float:
    """Exact Breit-Rabi energy in Hz for a J=1/2 level.

    Parameters
    ----------
    level : EnergyLevel  (must have J=1/2)
    F     : total angular momentum quantum number (I-1/2 or I+1/2)
    mF    : magnetic quantum number
    B_gauss : magnetic field in gauss

    Returns
    -------
    float : absolute energy in Hz

    Raises
    ------
    ValueError if J != 1/2.

    References
    ----------
    Arimondo, Inguscio, Violino, Rev. Mod. Phys. 49, 31 (1977), Appendix.
    Formula:
        E(F=I±1/2, mF) = -hfA*I/(2*(2I+1)) - gI*μN*mF*B
                         ± (hfA*(2I+1)/4) * sqrt(1 + 4mF*x/(2I+1) + x²)
        x = (gJ*μB - gI*μN) * B / (hfA * (I+1/2))
    The ± is + for F = I+1/2 and - for F = I-1/2.
    """
    J = float(level.term.J)
    if abs(J - 0.5) > 1e-9:
        raise ValueError(f"Breit-Rabi formula requires J=1/2, got J={J}")

    atom = level.atom
    I = float(atom.I)
    gJ = float(level.lande)
    gI = float(getattr(atom, 'gI', 0.0))
    hfA_Hz = float(level.hfA_Hz)

    sign = +1.0 if abs(F - (I + 0.5)) < 1e-9 else -1.0

    delta_hf = hfA_Hz * (I + 0.5)          # full HF interval (Hz)
    # gI is signed (negative for Yb-171). The Breit-Rabi parameter uses
    # (gJ·μB + gI·μN) — equivalent to Arimondo's (gJ·μB - |gI|·μN) when gI < 0.
    x = (gJ * _MU_B_HZ_G + gI * _MU_N_HZ_G) * B_gauss / delta_hf

    center = -hfA_Hz / 4.0  # geometric midpoint of F=I±1/2 at B=0
    nuclear = -gI * _MU_N_HZ_G * mF * B_gauss
    radical = np.sqrt(1.0 + 4.0 * mF * x / (2 * I + 1) + x ** 2)

    hf_z_shift = center + nuclear + sign * (hfA_Hz * (2 * I + 1) / 4) * radical
    return float(level.level_Hz) + hf_z_shift


# ---------------------------------------------------------------------------
# High-level sweep API
# ---------------------------------------------------------------------------

def zeeman_sweep(level: EnergyLevel, B_array) -> dict:
    """Compute nonlinear Zeeman energies for all sublevels over a B array.

    This is the recommended entry point for plotting Zeeman diagrams or
    driving B-sweep simulations.  Internally it calls
    :func:`diagonalize_hfz_batch` once and labels the resulting eigenvalues
    by (F, mF) using adiabatic correspondence with the zero-field HF states.

    If a spline LUT has been precomputed via
    :meth:`~atomtoolkit.atom.EnergyLevel.precompute_zeeman`, individual
    :meth:`~atomtoolkit.atom.ZLevel.level_hz_at` calls will be even faster;
    but for a fresh sweep this function is the most efficient option.

    Parameters
    ----------
    level : EnergyLevel
        A fine-structure manifold with hyperfine structure set.
    B_array : array-like
        Magnetic field values in gauss.

    Returns
    -------
    dict mapping (F, mF) -> ndarray of shape (len(B_array),)
        Absolute energies in Hz for each sublevel at each field value.
        F and mF are plain Python floats.

    Examples
    --------
    >>> from atomtoolkit.zeeman import zeeman_sweep
    >>> import numpy as np, matplotlib.pyplot as plt
    >>> B = np.linspace(0, 500, 500)
    >>> curves = zeeman_sweep(level, B)
    >>> for (F, mF), energies in curves.items():
    ...     plt.plot(B, (energies - level.level_Hz) / 1e6,
    ...              label=f'F={F}, mF={mF}')
    """
    B_array = np.asarray(B_array, dtype=float)
    batch = diagonalize_hfz_batch(level, B_array)

    J = float(level.term.J)
    I = float(level.atom.I)

    result = {}
    for mF, eigenvalues_array in batch.items():
        F_sorted = sorted(
            [float(f) for f in np.arange(abs(J - I), J + I + 1) if f >= abs(mF) - 1e-9],
            key=lambda f: _hf_shift_hz(J, I, f, level.hfA_Hz, level.hfB_Hz, level.hfC_Hz),
        )
        for idx, F in enumerate(F_sorted):
            result[(F, mF)] = eigenvalues_array[:, idx]
    return result


# ---------------------------------------------------------------------------
# Per-ZLevel energy with adiabatic state tracking
# ---------------------------------------------------------------------------

def _adiabatic_index(J: float, I: float, F: float, mF: float,
                     hfA_Hz: float, hfB_Hz: float, hfC_Hz: float) -> int:
    """Index of F within the mF block, ordered by ascending B=0 HF energy.

    At B=0, numpy.linalg.eigh returns eigenvalues in ascending order matching
    the HF energy order. This gives the adiabatic correspondence between
    eigenvalue index and (F, mF) label at all fields (assuming no level
    crossings within the mF block).
    """
    F_list = sorted(
        [float(f) for f in np.arange(abs(J - I), J + I + 1) if f >= abs(mF) - 1e-9],
        key=lambda f: _hf_shift_hz(J, I, f, hfA_Hz, hfB_Hz, hfC_Hz),
    )
    for idx, f in enumerate(F_list):
        if abs(f - F) < 1e-9:
            return idx
    raise ValueError(f"F={F} not found in mF={mF} block for J={J}, I={I}")


def level_hz_nonlinear(zlevel: ZLevel, B_gauss: float) -> float:
    """Exact nonlinear Zeeman energy for *zlevel* at *B_gauss* gauss (Hz).

    Checks for a precomputed spline LUT on the manifold first.  Falls back to
    the lru_cache-backed :func:`diagonalize_hfz` otherwise.
    """
    manifold = zlevel.manifold
    F = float(zlevel.term.F)
    mF = float(zlevel.term.mF)

    # Fast path: spline LUT
    splines = getattr(manifold, '_zeeman_splines', None)
    if splines is not None:
        entry = splines.get((F, mF))
        if entry is not None:
            spline, (B_min, B_max) = entry
            if B_min <= B_gauss <= B_max:
                return float(spline(B_gauss))
            warnings.warn(
                f"B={B_gauss} G is outside the precomputed spline range "
                f"[{B_min}, {B_max}] G for {manifold.name}; falling back to diagonalization.",
                stacklevel=2,
            )

    # Slow path: diagonalization (cached per exact B value)
    mF_block = diagonalize_hfz(manifold, B_gauss)
    eigenvalues = mF_block[mF][0]

    J = float(manifold.term.J)
    I = float(manifold.atom.I)
    idx = _adiabatic_index(J, I, F, mF,
                           manifold.hfA_Hz, manifold.hfB_Hz, manifold.hfC_Hz)
    return float(eigenvalues[idx])


# ---------------------------------------------------------------------------
# TODO: Breit-Rabi / Zeeman diagram plot helper
# ---------------------------------------------------------------------------
# A plot_breit_rabi(level, B_array, ax=None, **kwargs) convenience function
# should live in atomtoolkit/render/ (e.g. render/zeeman.py or render/spectra.py).
# It should call zeeman_sweep(), subtract the zero-field centroid energy, and
# colour curves by (F, mF) with a sensible legend.  Also useful:
#   - Differential-frequency plot: (E_a(B) - E_b(B)) for a chosen qubit pair
#   - Sensitivity plot: d(E_a - E_b)/dB via ZLevel.zeeman_sensitivity_hz_per_gauss()
