"""
Atom configuration file for 202Hg I. This file builds a full atom and can save it as a JSON file.

202Hg, I = 0. MOT on the 253.7 nm 1S0 -> 3P*1 intercombination line -- unusually,
mercury is cooled directly on the narrow line. Its very low sensitivity to
blackbody radiation makes it an attractive optical clock species.

Energy levels come from the NIST ASD snapshot cached in resources/HgI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '202Hg I'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/HgI.csv')

Hg202n = IO.generate_full_from_dataframe(_energy_level_df,
                                         name=NAME,
                                         I=I,
                                         num_levels=NUM_LEVELS,
                                         allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Hg202n.save_json('Hg_I_202.json')
