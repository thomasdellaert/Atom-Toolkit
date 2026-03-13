"""
Tools for calculating the relative transition strengths and allowed-ness of transitions using angular momentum math.
Currently also contains some other angular-momentum-heavy functions like converting between coupling schemes, but
this may eventually get moved either into the Term class or into a module of its own.
"""

import warnings
import functools
import numpy as np
from . import ureg, c, hb, e, a0

from .wigner import wigner3j, wigner6j, wigner9j


def A_to_oscillator_strength(A, J_lower, J_upper, omega):
    """
    Convert an Einstein A coefficient to the absorption oscillator strength f_lu.

    Uses the standard relation (Foot, "Atomic Physics", eq. 2.54):
        A_ul = (e² ω²) / (2π ε₀ mₑ c³)  ·  g_l/g_u  ·  f_lu

    :param A: Einstein A coefficient as a pint Quantity (units of Hz or s⁻¹)
    :param J_lower: total angular momentum of the lower level
    :param J_upper: total angular momentum of the upper level
    :param omega: angular transition frequency as a pint Quantity (rad/s)
    :return: dimensionless absorption oscillator strength f_lu (float)
    """
    m_e = ureg.electron_mass
    g_u = 2 * J_upper + 1
    g_l = 2 * J_lower + 1
    f = (2 * np.pi * ureg.epsilon_0 * m_e * c ** 3 / (e ** 2 * omega ** 2)) * (g_u / g_l) * A
    return float(f.to(''))


def oscillator_strength_to_A(f, J_lower, J_upper, omega):
    """
    Convert an absorption oscillator strength f_lu to an Einstein A coefficient.

    Inverse of A_to_oscillator_strength.

    :param f: dimensionless absorption oscillator strength f_lu
    :param J_lower: total angular momentum of the lower level
    :param J_upper: total angular momentum of the upper level
    :param omega: angular transition frequency as a pint Quantity (rad/s)
    :return: Einstein A coefficient as a pint Quantity in Hz
    """
    m_e = ureg.electron_mass
    g_u = 2 * J_upper + 1
    g_l = 2 * J_lower + 1
    A = (e ** 2 * omega ** 2 / (2 * np.pi * ureg.epsilon_0 * m_e * c ** 3)) * (g_l / g_u) * f
    return A.to('Hz')


def A_to_dipole_mel(A, J0, J1, omega):
    return ((A * (2 * J1 + 1) * 3 * np.pi * ureg.epsilon_0 * hb * c ** 3 / omega ** 3) ** 0.5).to(ureg.a0 * ureg.e)


def dipole_mel_to_A(mel, J0, J1, omega):
    return ((omega ** 3 / (3 * np.pi * hb * (c ** 3) * ureg.epsilon_0)) * 1 / (2 * J1 + 1) * mel ** 2).to(ureg.Hz)


@functools.lru_cache(maxsize=None)
def tkq_transition_strength(I: float, k: int, q: int,
                            J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    return tkq_mel(I, k, q, J0, F0, M0, J1, F1, M1)**2


@functools.lru_cache(maxsize=None)
def tkq_mel(I: float, k: int, q: int,
            J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    prod = 1
    prod *= (2 * F0 + 1) ** 0.5 * (2 * F1 + 2) ** 0.5 * wigner6j(J0, J1, k,
                                                                 F1, F0, I)
    prod *= wigner3j(F1, k, F0,
                     -M1, q, M0)
    return prod


# The tensor math for E2 (and somewhat E1) transitions is adapted from Tony's thesis (Ransford 2020)
@functools.lru_cache(maxsize=None)
def E1_mel_geom(eps: np.array, I: float,
                J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    eps = eps / np.linalg.norm(eps)
    tot = 0
    tot += tkq_mel(I, 1, -1, J0, F0, M0, J1, F1, M1) * 0.5 ** 0.5 * (eps[0] + 1j * eps[1])
    tot += tkq_mel(I, 1, 0, J0, F0, M0, J1, F1, M1) * eps[2]
    tot += tkq_mel(I, 1, 1, J0, F0, M0, J1, F1, M1) * 0.5 ** 0.5 * (eps[0] - 1j * eps[1])
    return tot


@functools.lru_cache(maxsize=None)
def E1_transition_strength_geom(eps: np.array, I: float,
                                J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    eps = eps / np.linalg.norm(eps)
    tot = 0
    tot += tkq_transition_strength(I, 1, -1, J0, F0, M0, J1, F1, M1) * 0.5 * (eps[0] ** 2 + eps[1] ** 2)
    tot += tkq_transition_strength(I, 1, 0, J0, F0, M0, J1, F1, M1) * eps[2] ** 2
    tot += tkq_transition_strength(I, 1, 1, J0, F0, M0, J1, F1, M1) * 0.5 * (eps[0] ** 2 + eps[1] ** 2)
    return tot


@functools.lru_cache(maxsize=None)
def E1_transition_strength_avg(I: float, J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    return sum([tkq_transition_strength(I, 1, q, J0, F0, M0, J1, F1, M1) for q in [-1, 0, 1]]) / 3.0


@functools.lru_cache(maxsize=None)
def M1_mel_geom(eps: np.array, I: float,
                J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    eps = eps / np.linalg.norm(eps)
    tot = 0
    tot += tkq_mel(I, 1, -1, J0, F0, M0, J1, F1, M1) * 0.5 ** 0.5 * (eps[0] + 1j * eps[1])
    tot += tkq_mel(I, 1, 0, J0, F0, M0, J1, F1, M1) * eps[2]
    tot += tkq_mel(I, 1, 1, J0, F0, M0, J1, F1, M1) * 0.5 ** 0.5 * (eps[0] - 1j * eps[1])
    return tot


@functools.lru_cache(maxsize=None)
def M1_transition_strength_geom(eps: np.array, I: float,
                                J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    eps = eps / np.linalg.norm(eps)
    tot = 0
    tot += (tkq_transition_strength(I, 1, -1, J0, F0, M0, J1, F1, M1) *
            0.5 * (eps[0] ** 2 + eps[1] ** 2))
    tot += tkq_transition_strength(I, 1, 0, J0, F0, M0, J1, F1, M1) * eps[2] ** 2
    tot += (tkq_transition_strength(I, 1, 1, J0, F0, M0, J1, F1, M1) *
            0.5 * (eps[0] ** 2 + eps[1] ** 2))
    return tot


@functools.lru_cache(maxsize=None)
def M1_transition_strength_avg(I: float, J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    return sum([tkq_transition_strength(I, 1, q, J0, F0, M0, J1, F1, M1) for q in [-1, 0, 1]]) / 3.0


@functools.lru_cache(maxsize=None)
def E2_mel_geom(eps: np.array, k: np.array, I: float,
                J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    eps = eps / np.linalg.norm(eps)
    k = k / np.linalg.norm(k)
    if np.dot(eps, k) != 0:
        warnings.warn("k-vector and polarization are not orthogonal")
    tot = 0
    tot += (tkq_mel(I, 2, -2, J0, F0, M0, J1, F1, M1) *
            (eps[0] * k[0] - 1j * (k[0] * eps[1] + k[1] + eps[0]) - k[1] * eps[1]))
    tot += (tkq_mel(I, 2, -1, J0, F0, M0, J1, F1, M1) *
            (-(k[0] * eps[2] + k[2] * eps[0]) + 1j * (k[1] * eps[2] + k[2] * eps[1])))
    tot += (tkq_mel(I, 2, 0, J0, F0, M0, J1, F1, M1) *
            (-6 ** 0.5 * k[0] * eps[0] + 6 ** 0.5 * k[1] * eps[1] + (2. / 3.) * 6 ** 0.5 * k[2] * eps[2]))
    tot += (tkq_mel(I, 2, 1, J0, F0, M0, J1, F1, M1) *
            ((k[0] * eps[2] + k[2] * eps[0]) + 1j * (k[1] * eps[2] + k[2] * eps[1])))
    tot += (tkq_mel(I, 2, 2, J0, F0, M0, J1, F1, M1) *
            (eps[0] * k[0] + 1j * (k[0] * eps[1] + k[1] + eps[0]) - k[1] * eps[1]))
    return tot


@functools.lru_cache(maxsize=None)
def E2_transition_strength_geom(eps: np.array, k: np.array, I: float,
                                J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    eps = eps / np.linalg.norm(eps)
    k = k / np.linalg.norm(k)
    if np.dot(eps, k) != 0:
        warnings.warn("k-vector and polarization are not orthogonal")
    tot = 0
    tot += (tkq_transition_strength(I, 2, -2, J0, F0, M0, J1, F1, M1) *
            (eps[0] ** 2 + eps[1] ** 2) * (k[0] ** 2 + k[1] ** 2))
    tot += (tkq_transition_strength(I, 2, -1, J0, F0, M0, J1, F1, M1) *
            (eps[2] * k[0] + eps[0] * k[2]) ** 2 + (eps[1] * k[0] + eps[0] * k[1]) ** 2)
    tot += (tkq_transition_strength(I, 2, 0, J0, F0, M0, J1, F1, M1) *
            (2. / 3.) * (3 * k[0] * eps[0] + 3 * k[1] * eps[1] + 2 * k[2] * eps[2]) ** 2)
    tot += (tkq_transition_strength(I, 2, 1, J0, F0, M0, J1, F1, M1) *
            (eps[2] * k[0] + eps[0] * k[2]) ** 2 + (eps[1] * k[0] + eps[0] * k[1]) ** 2)
    tot += (tkq_transition_strength(I, 2, 2, J0, F0, M0, J1, F1, M1) *
            (eps[0] ** 2 + eps[1] ** 2) * (k[0] ** 2 + k[1] ** 2))
    return tot


@functools.lru_cache(maxsize=None)
def E2_transition_strength_avg(I: float, J0: float, F0: float, M0: float, J1: float, F1: float, M1: float) -> float:
    tot = 0
    tot += tkq_transition_strength(I, 2, -2, J0, F0, M0, J1, F1, M1) * (3.0 / 29.0)
    tot += tkq_transition_strength(I, 2, -1, J0, F0, M0, J1, F1, M1) * (9.0 / 58.0)
    tot += tkq_transition_strength(I, 2, 0, J0, F0, M0, J1, F1, M1) * (14.0 / 29.0)
    tot += tkq_transition_strength(I, 2, 1, J0, F0, M0, J1, F1, M1) * (9.0 / 58.0)
    tot += tkq_transition_strength(I, 2, 2, J0, F0, M0, J1, F1, M1) * (3.0 / 29.0)
    return tot


@functools.lru_cache(maxsize=None)
def JJ_to_LS(J: float, Jc: float, Jo: float, lc: float, sc: float, lo: float, so: float) -> np.array:
    ls = np.arange(abs(lc - lo), abs(lc + lo) + 1)
    ss = np.arange(abs(sc - so), abs(sc + so) + 1)
    outls = np.repeat(ls, len(ss))
    outss = np.tile(ss, len(ls))
    outpercs = np.array([float(wigner9j(lc, sc, Jc,
                                        lo, so, Jo,
                                        outls[i], outss[i], J) *
                               np.sqrt((2 * Jc + 1) * (2 * Jo + 1) * (2 * outls[i] + 1) * (2 * outss[i] + 1)))
                         for i in range(len(outls))])
    return np.stack([outls, outss, outpercs])


@functools.lru_cache(maxsize=None)
def JK_to_LS(J: float, Jc: float, K: float, lc: float, sc: float, lo: float, so: float) -> np.array:
    ls = np.arange(abs(lc - lo), abs(lc + lo) + 1)
    ss = np.arange(abs(sc - so), abs(sc + so) + 1)
    outls = np.repeat(ls, len(ss))
    outss = np.tile(ss, len(ls))
    outpercs = np.array([float((-1) ** (-lc - lo - 2 * sc - so - K - outls[i] - J) *
                               np.sqrt((2 * Jc + 1) * (2 * outls[i] + 1) * (2 * K + 1) * (2 * outss[i] + 1)) *
                               wigner6j(sc, lc, Jc,
                                        lo, K, outls[i]) *
                               wigner6j(outls[i], sc, K,
                                        so, J, outss[i])) for i in range(len(outls))])
    return np.stack([outls, outss, outpercs])


@functools.lru_cache(maxsize=None)
def LK_to_LS(J: float, L: float, K: float, sc: float, so: float) -> np.array:
    ls = [L]
    ss = np.arange(abs(sc - so), abs(sc + so) + 1)
    outls = np.repeat(ls, len(ss))
    outss = np.tile(ss, len(ls))
    outpercs = np.array([np.sqrt((2 * K + 1) * (2 * outss[i] + 1)) * wigner6j(L, sc, K,
                                                                              so, J, outss[i]) for i in
                         range(len(outls))])
    return np.stack([outls, outss, outpercs])


if __name__ == '__main__':
    print(JJ_to_LS(3.5, 3.5, 0, 3, 0.5, 1, 1))

    print(JK_to_LS(0.5, 3.5, 1.5, 3, 0.5, 2, 1))
