"""
Atom configuration file for 211Fr I. This file builds a full atom and can save it as a JSON file.

211Fr (I = 9/2) is the heaviest alkali and is radioactive (t_1/2 ~ 3 min). Its
large nuclear charge makes it a leading atomic-parity-violation candidate.
NIST's Fr I level list is sparse and the hyperfine data are far less precise
than for the lighter alkalis.

Energy levels come from the NIST ASD snapshot cached in resources/FrI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '211Fr I'
I = 9/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/FrI.csv')

Fr211 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Fr211.gI = 0.88889  # mu_I = +4.00 mu_N, I = 9/2 -- VERIFY, the moment is only known to ~1%

# === Hyperfine coefficients ===
# Grossman et al., PRL 83, 935 (1999) -- VERIFY
Fr211.levels['7s 2S1/2'].hfA = Q_(8713.9, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 7p 2P*1/2 and 2P*3/2 hyperfine A and B
#   - a better nuclear magnetic moment for 211Fr

if __name__ == '__main__':
    Fr211.save_json('Fr_I_211.json')
