"""
Atom configuration file for 173Yb I. This file builds a full atom and can save it as a JSON file.

173Yb, I = 5/2 -- the other stable fermionic ytterbium isotope. Its larger nuclear
spin gives a 6-fold degenerate ground state, useful for SU(N) physics.

Energy levels come from the NIST ASD snapshot cached in resources/YbI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '173Yb I'
I = 5/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/YbI.csv')

Yb173n = IO.generate_full_from_dataframe(_energy_level_df,
                                         name=NAME,
                                         I=I,
                                         num_levels=NUM_LEVELS,
                                         allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Yb173n.gI = -0.27196  # mu_I = -0.6799 mu_N, I = 5/2 -- bare ratio, not shielding-corrected

# === Hyperfine coefficients ===
# Yb I 3P1 hyperfine A and B -- VERIFY
Yb173n.levels['4f14.6s.6p 3P*1'].hfA = Q_(-1094.2, 'MHz')
Yb173n.levels['4f14.6s.6p 3P*1'].hfB = Q_(-826.5, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 4f14.6s.6p 3P*2 and 1P*1 hyperfine A and B

if __name__ == '__main__':
    Yb173n.save_json('Yb_I_173.json')
