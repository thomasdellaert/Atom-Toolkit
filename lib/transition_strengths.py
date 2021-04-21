import numpy as np
import warnings
try:
    # py3nj is a pain to install, and requires a fortran compiler among other things. Therefore it's optional,
    # and everything can work with sympy if it needs to. Py3nj is a bit faster, and it supports vectorized inputs,
    # so it's preferred.
    from py3nj import wigner3j, wigner6j, wigner9j
    nj = True
except ImportError:
    from sympy.physics.wigner import wigner_3j as wigner3j
    from sympy.physics.wigner import wigner_6j as wigner6j
    from sympy.physics.wigner import wigner_9j as wigner9j
    from sympy import N
    nj = False

def wignerPicker(func):
    def wrapper(*args):
        if nj:
            # py3nj takes the arguments doubled, and requires them to be integers
            return func(*map(lambda x: int(x*2), args))
        else:
            # sympy throws an error when selection rules are violated, but it needs to return 0
            try:
                return N(func(*args))
            except ValueError:
                return 0.0
    return wrapper


wigner3j = wignerPicker(wigner3j)
wigner6j = wignerPicker(wigner6j)

# noinspection PyTypeChecker
def tkq_LS_transition_strength(I, k, q, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=2):
    prod = 1
    if depth > 1:
        prod *= (S0 == S1) * (2 * J0 + 1) * (2 * J1 + 1) * \
                wigner6j(L0, L1, k,
                         J1, J0, S0) ** 2
    if depth > 0:
        prod *= (2 * F0 + 1) * (2 * F1 + 2) * \
                wigner6j(J0, J1, k,
                         F1, F0, I) ** 2
    prod *= wigner3j(F1,  k, F0,
                     -M1, q, M0) ** 2
    return prod


# The tensor math for E2 (and somewhat E1) transitions is adapted from Tony's thesis (Ransford 2020)

def E1_transition_strength_geom(eps, I, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=2):
    eps = eps / np.linalg.norm(eps)
    tot = 0
    tot += tkq_LS_transition_strength(I, 1, -1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=depth) * 0.5 * (eps[0] + eps[1]) ** 2
    tot += tkq_LS_transition_strength(I, 1, 0, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=depth) * eps[2] ** 2
    tot += tkq_LS_transition_strength(I, 1, 1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=depth) * 0.5 * (eps[0] + eps[1]) ** 2
    return tot


def E1_transition_strength_avg(I, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=2):
    if True:  # (L1 - L0) % 2 == 1:
        tot = 0
        tot += tkq_LS_transition_strength(I, 1, -1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=depth) * (1.0 / 3.0)
        tot += tkq_LS_transition_strength(I, 1, 0, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=depth) * (1.0 / 3.0)
        tot += tkq_LS_transition_strength(I, 1, 1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=depth) * (1.0 / 3.0)
        return tot
    else:
        return 0


def M1_transition_strength_geom(eps, I, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1):
    eps = eps / np.linalg.norm(eps)
    tot = 0
    if S1 == S0 and L1 == L0:
        tot += tkq_LS_transition_strength(I, 1, -1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=1) * \
               0.5 * (eps[0] ** 2 + eps[1] ** 2)
        tot += tkq_LS_transition_strength(I, 1,  0, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=1) * eps[2] ** 2
        tot += tkq_LS_transition_strength(I, 1,  1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=1) * \
               0.5 * (eps[0] ** 2 + eps[1] ** 2)
        return tot
    else:
        return 0


def M1_transition_strength_avg(I, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1):
    tot = 0
    if S1 == S0 and L1 == L0:
        tot += tkq_LS_transition_strength(I, 1, -1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=1) * (1.0 / 3.0)
        tot += tkq_LS_transition_strength(I, 1,  0, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=1) * (1.0 / 3.0)
        tot += tkq_LS_transition_strength(I, 1,  1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1, depth=1) * (1.0 / 3.0)
        return tot
    else:
        return 0


def E2_transition_strength_geom(eps, k, I, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1):
    eps = eps / np.linalg.norm(eps)
    k = k / np.linalg.norm(k)
    if np.dot(eps, k) != 0:
        warnings.warn("k-vector and polarization are not orthogonal")
    tot = 0
    tot += tkq_LS_transition_strength(I, 2, -2, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * \
        (eps[0] ** 2 + eps[1] ** 2) * (k[0] ** 2 + k[1] ** 2)
    tot += tkq_LS_transition_strength(I, 2, -1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * \
        (eps[2] * k[0] + eps[0] * k[2]) ** 2 + (eps[1] * k[0] + eps[0] * k[1]) ** 2
    tot += tkq_LS_transition_strength(I, 2, 0, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * \
        (2. / 3.) * (3 * k[0] * eps[0] + 3 * k[1] * eps[1] + 2 * k[2] * eps[2]) ** 2
    tot += tkq_LS_transition_strength(I, 2, 1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * \
        (eps[2] * k[0] + eps[0] * k[2]) ** 2 + (eps[1] * k[0] + eps[0] * k[1]) ** 2
    tot += tkq_LS_transition_strength(I, 2, 2, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * \
        (eps[0] ** 2 + eps[1] ** 2) * (k[0] ** 2 + k[1] ** 2)
    return tot


def E2_transition_strength_avg(I, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1):
    tot = 0
    tot += tkq_LS_transition_strength(I, 2, -2, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * (3.0 / 29.0)
    tot += tkq_LS_transition_strength(I, 2, -1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * (9.0 / 58.0)
    tot += tkq_LS_transition_strength(I, 2, 0, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * (14.0 / 29.0)
    tot += tkq_LS_transition_strength(I, 2, 1, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * (9.0 / 58.0)
    tot += tkq_LS_transition_strength(I, 2, 2, L0, S0, J0, F0, M0, L1, S1, J1, F1, M1) * (4.0 / 15.0)
    return tot

# TODO: Make this work outside of LS coupling. Probably involves converting
#  between LS, JJ, and JK couplings in a different .py file


if __name__ == '__main__':
    I_0 = 2.5
    L_0 = 3
    L_1 = 3
    S_0 = 0.5
    S_1 = 0.5
    J_0 = 3.5
    J_1 = 3.5
    F_0 = 4
    F_1 = 3
    G = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    E = [-3, -2, -1, 0, 1, 2, 3]
    print(wigner3j(0.0, 0.5, 0.5, 0.0, 0.5, -0.5))
    print("M1")
    flg = False
    for mg in G:
        for me in E:
            s = M1_transition_strength_avg(I_0, L_0, S_0, J_0, F_0, mg, L_1, S_1, J_1, F_1, me)
            if s != 0:
                flg = True
                print("{0:.6f} {1:} {2:}".format(s, mg, me))
    if not flg:
        print("no allowed transitions")
    print("E1")
    flg = False
    for mg in G:
        for me in E:
            s = E1_transition_strength_avg(I_0, L_0, S_0, J_0, F_0, mg, L_1, S_1, J_1, F_1, me)
            if s != 0:
                flg = True
                print("{0:.6f} {1:} {2:}".format(s, mg, me))
    if not flg:
        print("no allowed transitions")
