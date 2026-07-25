"""
Atom configuration file for 199Hg I. This file builds a full atom and can save it as a JSON file.

199Hg, I = 1/2 -- the odd isotope used for the mercury optical lattice clock, on
the 265.6 nm 1S0 -> 3P*0 hyperfine-induced transition.

Energy levels come from the NIST ASD snapshot cached in resources/HgI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '199Hg I'
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('HgI.csv'))

Hg199n = IO.generate_full_from_dataframe(_energy_level_df,
                                         name=NAME,
                                         I=I,
                                         num_levels=NUM_LEVELS,
                                         allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Hg199n.gI = 1.01177  # mu_I = +0.5058855 mu_N, I = 1/2; same nucleus as species/Hg_II_199.py

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 5d10.6s.6p 3P*1 hyperfine A (and the 3P*2 / 1P*1 values)
#   - a precise frequency for the 265.6 nm clock transition

if __name__ == '__main__':
    Hg199n.save_json('Hg_I_199.json')
