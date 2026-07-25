"""
Atom configuration file for 67Zn II. This file builds a full atom and can save it as a JSON file.

67Zn+ is the only stable Zn isotope with nuclear spin (I = 5/2).

Energy levels come from the NIST ASD snapshot cached in resources/ZnII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '67Zn II'
I = 5/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/ZnII.csv')

Zn67 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Zn67.gI = 0.35019  # mu_I = +0.875479 mu_N, I = 5/2

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 4s 2S1/2 hyperfine A (the ground-state hyperfine splitting)
#   - 4p 2P*1/2 and 2P*3/2 hyperfine A and B

if __name__ == '__main__':
    Zn67.save_json('Zn_II_67.json')
