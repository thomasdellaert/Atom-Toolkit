import numpy as np
import warnings
from .wigner import *


@functools.lru_cache(maxsize=None)
def tkq_transition_strength(I, k, q, J0, F0, M0, J1, F1, M1):
    prod = 1
    prod *= (2 * F0 + 1) * (2 * F1 + 2) * wigner6j(J0, J1, k,
                                                   F1, F0, I) ** 2
    prod *= wigner3j(F1, k, F0,
                     -M1, q, M0) ** 2
    return prod


# The tensor math for E2 (and somewhat E1) transitions is adapted from Tony's thesis (Ransford 2020)
@functools.lru_cache(maxsize=None)
def E1_transition_strength_geom(eps, I, J0, F0, M0, J1, F1, M1):
    eps = eps / np.linalg.norm(eps)
    tot = 0
    tot += tkq_transition_strength(I, 1, -1, J0, F0, M0, J1, F1, M1) * 0.5 * (eps[0] + eps[1]) ** 2
    tot += tkq_transition_strength(I, 1, 0, J0, F0, M0, J1, F1, M1) * eps[2] ** 2
    tot += tkq_transition_strength(I, 1, 1, J0, F0, M0, J1, F1, M1) * 0.5 * (eps[0] + eps[1]) ** 2
    return tot


@functools.lru_cache(maxsize=None)
def E1_transition_strength_avg(I, J0, F0, M0, J1, F1, M1):
    return sum([tkq_transition_strength(I, 1, q, J0, F0, M0, J1, F1, M1) for q in [-1, 0, 1]]) / 3.0


@functools.lru_cache(maxsize=None)
def M1_transition_strength_geom(eps, I, J0, F0, M0, J1, F1, M1):
    eps = eps / np.linalg.norm(eps)
    tot = 0
    tot += tkq_transition_strength(I, 1, -1, J0, F0, M0, J1, F1, M1) * \
           0.5 * (eps[0] ** 2 + eps[1] ** 2)
    tot += tkq_transition_strength(I, 1, 0, J0, F0, M0, J1, F1, M1) * eps[2] ** 2
    tot += tkq_transition_strength(I, 1, 1, J0, F0, M0, J1, F1, M1) * \
           0.5 * (eps[0] ** 2 + eps[1] ** 2)
    return tot


@functools.lru_cache(maxsize=None)
def M1_transition_strength_avg(I, J0, F0, M0, J1, F1, M1):
    return sum([tkq_transition_strength(I, 1, q, J0, F0, M0, J1, F1, M1) for q in [-1, 0, 1]]) / 3.0


@functools.lru_cache(maxsize=None)
def E2_transition_strength_geom(eps, k, I, J0, F0, M0, J1, F1, M1):
    eps = eps / np.linalg.norm(eps)
    k = k / np.linalg.norm(k)
    if np.dot(eps, k) != 0:
        warnings.warn("k-vector and polarization are not orthogonal")
    tot = 0
    tot += tkq_transition_strength(I, 2, -2, J0, F0, M0, J1, F1, M1) * \
           (eps[0] ** 2 + eps[1] ** 2) * (k[0] ** 2 + k[1] ** 2)
    tot += tkq_transition_strength(I, 2, -1, J0, F0, M0, J1, F1, M1) * \
           (eps[2] * k[0] + eps[0] * k[2]) ** 2 + (eps[1] * k[0] + eps[0] * k[1]) ** 2
    tot += tkq_transition_strength(I, 2, 0, J0, F0, M0, J1, F1, M1) * \
           (2. / 3.) * (3 * k[0] * eps[0] + 3 * k[1] * eps[1] + 2 * k[2] * eps[2]) ** 2
    tot += tkq_transition_strength(I, 2, 1, J0, F0, M0, J1, F1, M1) * \
           (eps[2] * k[0] + eps[0] * k[2]) ** 2 + (eps[1] * k[0] + eps[0] * k[1]) ** 2
    tot += tkq_transition_strength(I, 2, 2, J0, F0, M0, J1, F1, M1) * \
           (eps[0] ** 2 + eps[1] ** 2) * (k[0] ** 2 + k[1] ** 2)
    return tot


@functools.lru_cache(maxsize=None)
def E2_transition_strength_avg(I, J0, F0, M0, J1, F1, M1):
    tot = 0
    tot += tkq_transition_strength(I, 2, -2, J0, F0, M0, J1, F1, M1) * (3.0 / 29.0)
    tot += tkq_transition_strength(I, 2, -1, J0, F0, M0, J1, F1, M1) * (9.0 / 58.0)
    tot += tkq_transition_strength(I, 2, 0, J0, F0, M0, J1, F1, M1) * (14.0 / 29.0)
    tot += tkq_transition_strength(I, 2, 1, J0, F0, M0, J1, F1, M1) * (9.0 / 58.0)
    tot += tkq_transition_strength(I, 2, 2, J0, F0, M0, J1, F1, M1) * (3.0 / 29.0)
    return tot


@functools.lru_cache(maxsize=None)
def JJ_to_LS(J, Jc, Jo, lc, sc, lo, so):
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
def JK_to_LS(J, Jc, K, lc, sc, lo, so):
    ls = np.arange(abs(lc - lo), abs(lc + lo) + 1)
    ss = np.arange(abs(sc - so), abs(sc + so) + 1)
    outls = np.repeat(ls, len(ss))
    outss = np.tile(ss, len(ls))
    outpercs = np.array([float((-1) ** (-lc - lo - 2 * sc - so - K - outls[i] - J) * \
                               np.sqrt((2 * Jc + 1) * (2 * outls[i] + 1) * (2 * K + 1) * (2 * outss[i] + 1)) * \
                               wigner6j(sc, lc, Jc,
                                        lo, K, outls[i]) * \
                               wigner6j(outls[i], sc, K,
                                        so, J, outss[i])) for i in range(len(outls))])
    return np.stack([outls, outss, outpercs])


@functools.lru_cache(maxsize=None)
def LK_to_LS(J, L, K, sc, so):
    ls = [L]
    ss = np.arange(abs(sc - so), abs(sc + so) + 1)
    outls = np.repeat(ls, len(ss))
    outss = np.tile(ss, len(ls))
    outpercs = np.array([np.sqrt((2 * K + 1) * (2 * outss[i] + 1)) * wigner6j(L, sc, K,
                                                                              so, J, outss[i]) for i in range(len(outls))])
    return np.stack([outls, outss, outpercs])


if __name__ == '__main__':
    # I_0 = 2.5
    # L_0 = 3
    # L_1 = 3
    # S_0 = 0.5
    # S_1 = 0.5
    # J_0 = 3.5
    # J_1 = 3.5
    # F_0 = 4
    # F_1 = 3
    # G = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    # E = [-3, -2, -1, 0, 1, 2, 3]
    # print(wigner3j(0.0, 0.5, 0.5, 0.0, 0.5, -0.5))
    # print("M1")
    # flg = False
    # for mg in G:
    #     for me in E:
    #         s = M1_transition_strength_avg(I_0, J_0, F_0, mg, J_1, F_1, me)
    #         if s != 0:
    #             flg = True
    #             print("{0:.6f} {1:} {2:}".format(s, mg, me))
    # if not flg:
    #     print("no allowed transitions")
    # print("E1")
    # flg = False
    # for mg in G:
    #     for me in E:
    #         s = E1_transition_strength_avg(I_0, J_0, F_0, mg, J_1, F_1, me)
    #         if s != 0:
    #             flg = True
    #             print("{0:.6f} {1:} {2:}".format(s, mg, me))
    # if not flg:
    #     print("no allowed transitions")
    print(JJ_to_LS(3.5, 3.5, 0, 3, 0.5, 1, 1))

    print(JK_to_LS(0.5, 3.5, 1.5, 3, 0.5, 2, 1))
