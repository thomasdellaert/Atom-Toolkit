from __future__ import annotations
from typing import List, Tuple

import numpy as np

from .atom import Atom, BaseLevel, BaseTransition, ZLevel
from .environment import Environment, LightField
from . import transition_strengths as ts
from . import hb, e, Hz


def compute_atomic_matrix_element(s0: BaseLevel, s1: BaseLevel, light: LightField):
    """
    Compute the geometric dipole matrix element amplitude connecting two states for a given LightField.
    :param s0: the first state in the matrix element
    :param s1: the second state in the matrix element
    :param light: a LightField object providing the perturbing interaction
    :return: dimensionless geometric matrix element amplitude
    """
    a = s0.atom
    t = a.transitions[(s0.name, s1.name)]
    pol = tuple(light.pol.xyz.astype(complex))
    if t.allowed_types[0]:
        return ts.E1_mel_geom(pol, a.I,
                              s0.term.J, s0.term.F, s0.term.mF,
                              s1.term.J, s1.term.F, s1.term.mF)
    elif t.allowed_types[1]:
        return ts.M1_mel_geom(pol, a.I,
                              s0.term.J, s0.term.F, s0.term.mF,
                              s1.term.J, s1.term.F, s1.term.mF)
    elif t.allowed_types[2]:
        return ts.E2_mel_geom(pol, tuple(light.k.astype(complex)), a.I,
                              s0.term.J, s0.term.F, s0.term.mF,
                              s1.term.J, s1.term.F, s1.term.mF)
    else:
        return 0


def mel_to_rabi_frequency(mel, light: LightField):
    """
    Convert the dipole matrix element to a Rabi frequency
    :param mel: the matrix element
    :param light: the LightField
    :return:
    """
    return e * mel * np.linalg.norm(light.E()) / hb


def generate_dynamics_matrix(atom: Atom,
                             states: List[BaseLevel],
                             environment: Environment = None) -> List:
    """
    Generate Lindblad collapse operators for open-system (master equation) dynamics.

    Each collapse operator corresponds to a spontaneous-decay channel
    ``|upper, mF_u⟩ → |lower, mF_l⟩`` and has the form
    ``√Γ |lower⟩⟨upper|``, where ``Γ`` (rad/s) is the partial decay rate
    for that channel summed over all emitted photon polarisations.

    Partial rates are computed from the parent ``Transition.A`` coefficient
    and the angular-momentum geometric factors, so **A must be set on every
    relevant ``Transition`` object** in the atom.  Channels whose parent
    transition has no A coefficient are silently skipped.

    The basis expansion follows the same rules as :func:`generate_hamiltonian`:
    non-Zeeman levels are recursively expanded to all their Zeeman sublevels,
    and the final list is sorted by energy.

    Usage with qutip master equation solver::

        H0, Hs = generate_hamiltonian(atom, states, fields)
        c_ops  = generate_dynamics_matrix(atom, states)
        result = qutip.mesolve([H0, *zip(Hs, envelopes)], rho0, tlist, c_ops)

    :param atom: the Atom object
    :param states: basis — any mix of EnergyLevel, HFLevel, or ZLevel
    :return: list of ``qutip.Qobj`` collapse operators
    :raises ImportError: if qutip is not installed
    :raises ValueError: if the expanded basis is empty
    """
    try:
        import qutip
    except ImportError:
        raise ImportError("qutip is required. Install with: pip install qutip")

    # Expand to ZLevels (same two-level expansion as generate_hamiltonian)
    z_states: List[ZLevel] = []
    for s in states:
        if isinstance(s, ZLevel):
            z_states.append(s)
        else:
            for sub in s.sublevels():
                if isinstance(sub, ZLevel):
                    z_states.append(sub)
                else:
                    z_states.extend(sub.sublevels())

    if not z_states:
        raise ValueError("No Zeeman sublevels found in the provided states")

    B_gauss = environment.B_gauss if environment is not None else 0.0
    z_states = sorted(z_states, key=lambda s: s.level_hz_at(B_gauss))
    n = len(z_states)
    I = atom.I

    # First pass: for each (upper ZLevel, parent Transition) pair, accumulate
    # the total angular-momentum strength so we can normalise partial rates.
    # Key: (upper_index, id(parent_Transition))  →  summed strength
    from collections import defaultdict
    upper_parent_norm = defaultdict(float)

    # channel_list: (lower_idx, upper_idx, parent_Transition, channel_strength, allowed_type)
    channel_list = []

    for j in range(n):        # j = upper
        sj = z_states[j]
        for i in range(j):    # i = lower
            si = z_states[i]

            try:
                zt = atom.transitions[(si.name, sj.name)]
            except (KeyError, TypeError):
                continue
            if zt is None or not np.any(zt.allowed_types):
                continue

            hft = zt.parent
            et = hft.parent if hft is not None else None
            if et is None or et.A is None:
                continue

            if zt.allowed_types[0]:     # E1
                strength = sum(
                    ts.tkq_transition_strength(I, 1, q,
                                               si.term.J, si.term.F, si.term.mF,
                                               sj.term.J, sj.term.F, sj.term.mF)
                    for q in (-1, 0, 1)
                )
                atype = 0
            elif zt.allowed_types[1]:   # M1 — same angular structure as E1
                strength = sum(
                    ts.tkq_transition_strength(I, 1, q,
                                               si.term.J, si.term.F, si.term.mF,
                                               sj.term.J, sj.term.F, sj.term.mF)
                    for q in (-1, 0, 1)
                )
                atype = 1
            else:
                continue

            if strength == 0:
                continue

            upper_parent_norm[(j, id(et))] += strength
            channel_list.append((i, j, et, strength, id(et)))

    # Second pass: build collapse operators
    collapse_ops = []
    for (i, j, et, strength, et_id) in channel_list:
        total = upper_parent_norm[(j, et_id)]
        if total == 0:
            continue

        partial_rate_hz = et.A.to('Hz').magnitude * strength / total
        if partial_rate_hz <= 0:
            continue

        rate_rad_s = 2 * np.pi * partial_rate_hz
        c_mat = np.zeros((n, n), dtype=complex)
        c_mat[i, j] = np.sqrt(rate_rad_s)   # |lower⟩⟨upper|
        collapse_ops.append(qutip.Qobj(c_mat))

    return collapse_ops


def generate_hamiltonian(atom: Atom,
                         states: List[BaseLevel],
                         environment: Environment) -> Tuple:
    """
    Generate a bare qutip Hamiltonian for a given atom, state basis, and environment.

    Returns ``(H0, coupling_Hs)`` where:

    - ``H0`` — diagonal ``qutip.Qobj`` of shape (n, n) with bare state energies in rad/s,
      zeroed to the lowest-energy state in the basis.
    - ``coupling_Hs`` — list of ``qutip.Qobj`` (one per field in the environment), each
      Hermitian with off-diagonal element [i, j] equal to Ω_ij/2 in rad/s (RWA convention).

    These can be assembled into a qutip time-dependent Hamiltonian::

        H = [H0, [coupling_Hs[0], envelope_func], ...]

    **Units**: ``field.intensity`` must carry pint units (e.g. ``Q_(1e4, 'W/m^2')``);
    a plain float will cause a unit error when multiplied against the dipole moment.

    :param atom: the Atom object
    :param states: basis — any mix of EnergyLevel, HFLevel, or ZLevel. Non-Zeeman levels
        are expanded to all their Zeeman sublevels. The final basis is sorted by energy.
    :param environment: an Environment object containing the magnetic field and LightFields
    :return: (H0, coupling_Hs)
    :raises ImportError: if qutip is not installed
    :raises ValueError: if the expanded basis is empty
    """
    try:
        import qutip
    except ImportError:
        raise ImportError("qutip is required. Install with: pip install qutip")

    # Expand any non-Zeeman levels recursively to their Zeeman sublevels
    z_states: List[ZLevel] = []
    for s in states:
        if isinstance(s, ZLevel):
            z_states.append(s)
        else:
            for sub in s.sublevels():
                if isinstance(sub, ZLevel):
                    z_states.append(sub)
                else:
                    z_states.extend(sub.sublevels())

    if not z_states:
        raise ValueError("No Zeeman sublevels found in the provided states")

    B_gauss = environment.B_gauss
    z_states = sorted(z_states, key=lambda s: s.level_hz_at(B_gauss))
    n = len(z_states)
    I = atom.I

    # H0: diagonal energies in rad/s relative to the lowest state
    ground_Hz = z_states[0].level_hz_at(B_gauss)
    H0 = qutip.Qobj(np.diag([2 * np.pi * (s.level_hz_at(B_gauss) - ground_Hz) for s in z_states]))

    coupling_Hs = []
    for field in environment.fields:
        H_mat = np.zeros((n, n), dtype=complex)
        pol = tuple(field.pol.xyz.astype(complex))
        E0 = np.linalg.norm(field.E())  # V/m (requires pint units on field.intensity)

        for i in range(n):
            for j in range(i + 1, n):
                si, sj = z_states[i], z_states[j]

                try:
                    zt = atom.transitions[(si.name, sj.name)]
                except (KeyError, TypeError):
                    continue
                if zt is None or not np.any(zt.allowed_types):
                    continue

                # Walk the transition hierarchy: ZTransition -> HFTransition -> Transition
                hft = zt.parent
                et = hft.parent if hft is not None else None
                if et is None or et.A is None:
                    continue

                freq_hz = zt.freq_Hz
                if freq_hz == 0:
                    continue

                # Reduced dipole matrix element |⟨J_lower||er||J_upper⟩| in a0·e
                d_red = ts.A_to_dipole_mel(et.A,
                                           et.E_lower.term.J,
                                           et.E_upper.term.J,
                                           2 * np.pi * freq_hz * Hz)

                # Signed geometric coupling amplitude (dimensionless)
                if zt.allowed_types[0]:    # E1
                    geom = ts.E1_mel_geom(pol, I,
                                          si.term.J, si.term.F, si.term.mF,
                                          sj.term.J, sj.term.F, sj.term.mF)
                elif zt.allowed_types[1]:  # M1
                    geom = ts.M1_mel_geom(pol, I,
                                          si.term.J, si.term.F, si.term.mF,
                                          sj.term.J, sj.term.F, sj.term.mF)
                elif zt.allowed_types[2]:  # E2
                    k_hat = tuple((field.k / np.linalg.norm(field.k)).astype(complex))
                    geom = ts.E2_mel_geom(pol, k_hat, I,
                                          si.term.J, si.term.F, si.term.mF,
                                          sj.term.J, sj.term.F, sj.term.mF)
                else:
                    continue

                if geom == 0:
                    continue

                # Ω = d_red · geom · E0 / ħ  (rad/s);  store Ω/2 in coupling matrix
                rabi_half = (d_red * geom * E0 / hb).to('Hz').magnitude / 2
                H_mat[i, j] = rabi_half
                H_mat[j, i] = np.conj(rabi_half)

        coupling_Hs.append(qutip.Qobj(H_mat))

    return H0, coupling_Hs
