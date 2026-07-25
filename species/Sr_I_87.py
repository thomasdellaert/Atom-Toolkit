"""
Atom configuration file for 87Sr I. This file builds a full atom and can save it as a JSON file.

87Sr, I = 9/2 -- the fermionic isotope and the workhorse optical lattice clock.
Hyperfine mixing of 3P*0 with 3P*1 makes the 698 nm 1S0 -> 3P*0 clock transition
weakly allowed (~1 mHz natural linewidth), which is why the odd isotope is used.
Its I = 9/2 also gives the largest nuclear spin of any common neutral clock atom,
and hence a 10-fold degenerate ground state.

Energy levels come from the NIST ASD snapshot cached in resources/SrI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '87Sr I'
I = 9/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/SrI.csv')

Sr87n = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Sr87n.gI = -0.24298  # mu_I = -1.0936 mu_N, I = 9/2; same nucleus as species/Sr_II_87.py

# === Hyperfine coefficients ===
# Heider & Brink, PRA 16, 1371 (1977) -- VERIFY
Sr87n.levels['5s.5p 3P*1'].hfA = Q_(-260.084, 'MHz')
Sr87n.levels['5s.5p 3P*1'].hfB = Q_(-35.658, 'MHz')

# === Forbidden clock transitions ===
# J=0 -> J=0 is forbidden for every multipole, so this is never populated
# automatically; it is only weakly allowed through hyperfine mixing in the
# odd isotopes, and has to be added by hand.
# the 698 nm clock transition, a BIPM secondary representation of the second.
# A is the ~1 mHz hyperfine-induced rate -- order of magnitude only
_t = Sr87n.add_transition((Sr87n.levels['5s2 1S0'], Sr87n.levels['5s.5p 3P*0']),
                         A=Q_(1e-3, 'Hz'))
_t.set_frequency(Q_(429.228004229873, 'THz'))

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 5s.5p 3P*2 and 1P*1 hyperfine A and B
#   - 5s.4d 3D1/2/3 hyperfine A and B

if __name__ == '__main__':
    Sr87n.save_json('Sr_I_87.json')
