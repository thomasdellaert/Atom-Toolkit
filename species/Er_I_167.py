"""
Atom configuration file for 167Er I. This file builds a full atom and can save it as a JSON file.

167Er, I = 7/2 -- the only stable fermionic erbium isotope. Its hyperfine B is
very large and negative, so the ground manifold departs strongly from the Lande
interval rule, much as in Ho I.

Energy levels come from the NIST ASD snapshot cached in resources/ErI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '167Er I'
I = 7/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/ErI.csv')

Er167 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Er167.gI = -0.16111  # mu_I = -0.5639 mu_N, I = 7/2 -- bare ratio

# === Hyperfine coefficients ===
# ground-state hyperfine A and B -- VERIFY
Er167.levels['4f12.6s2 3H6'].hfA = Q_(-120.487, 'MHz')
Er167.levels['4f12.6s2 3H6'].hfB = Q_(-4552.984, 'MHz')

# |B| >> |A| here, so the quadrupole term dominates and the linear-Zeeman form is a
# poor approximation. Diagonalise H_HF + H_Z exactly instead.
Er167.levels['4f12.6s2 3H6'].use_nonlinear_zeeman = True

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - hyperfine A and B for the 401 nm and 583 nm upper levels

if __name__ == '__main__':
    Er167.save_json('Er_I_167.json')
