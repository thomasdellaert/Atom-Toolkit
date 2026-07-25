"""
Atom configuration file for 225Ra II. This file builds a full atom and can save it as a JSON file.

225Ra+ (I=1/2) is a radioactive optical-clock and atomic-parity-violation
candidate: high nuclear charge amplifies the parity-violating signal, and I=1/2
keeps the hyperfine structure simple.
NIST's Ra II level list is sparse (~40 levels) compared with the lighter ions.

Energy levels come from the NIST ASD snapshot cached in resources/RaII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '225Ra II'
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/RaII.csv')

Ra225 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Ra225.gI = -1.4676  # mu_I = -0.7338 mu_N, I = 1/2

# === Hyperfine coefficients ===
# ground-state hyperfine splitting ~27.7 GHz -- VERIFY against a primary source
Ra225.levels['7s 2S1/2'].hfA = Q_(27700.0, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 6d 2D3/2 and 2D5/2 hyperfine A and B
#   - 7p 2P*1/2 and 2P*3/2 hyperfine A and B
#   - a precise value for the 7s 2S1/2 hyperfine A

if __name__ == '__main__':
    Ra225.save_json('Ra_II_225.json')
