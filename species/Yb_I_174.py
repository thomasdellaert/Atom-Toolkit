"""
Atom configuration file for 174Yb I. This file builds a full atom and can save it as a JSON file.

174Yb, I = 0 -- the most abundant ytterbium isotope. Blue MOT on the 398.9 nm
1S0 -> 1P*1 line, then a green MOT on the 555.8 nm 1S0 -> 3P*1 intercombination
line (182 kHz wide). Heavily used for tweezer arrays and optical clocks.

Energy levels come from the NIST ASD snapshot cached in resources/YbI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '174Yb I'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/YbI.csv')

Yb174n = IO.generate_full_from_dataframe(_energy_level_df,
                                         name=NAME,
                                         I=I,
                                         num_levels=NUM_LEVELS,
                                         allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Yb174n.save_json('Yb_I_174.json')
