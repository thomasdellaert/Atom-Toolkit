"""
Atom configuration file for 88Sr I. This file builds a full atom and can save it as a JSON file.

88Sr, I = 0 -- the bosonic strontium isotope. Blue MOT on the 460.9 nm 1S0 -> 1P*1
line, then a red MOT on the 689 nm 1S0 -> 3P*1 intercombination line (7.4 kHz wide).
With I = 0 the 1S0 -> 3P*0 clock transition is strictly forbidden, so 88Sr clocks
need external field mixing; see Sr_I_87 for the isotope that does not.

Energy levels come from the NIST ASD snapshot cached in resources/SrI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '88Sr I'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/SrI.csv')

Sr88n = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Sr88n.save_json('Sr_I_88.json')
