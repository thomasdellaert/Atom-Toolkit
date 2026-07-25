"""
Atom configuration file for 40Ca I. This file builds a full atom and can save it as a JSON file.

40Ca, I = 0. MOT on the 422.8 nm 1S0 -> 1P*1 line, with 657 nm 1S0 -> 3P*1 as the
narrow clock transition. Calcium was one of the first neutral optical frequency
standards.

Energy levels come from the NIST ASD snapshot cached in resources/CaI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '40Ca I'
I = 0
# NIST's Ca I table runs to ~450 levels, and populate_transitions is O(N^2) in
# Clebsch-Gordan evaluations, so the cap is set low. Everything below ~50000 cm-1,
# which is all of the cooling and clock physics, is comfortably inside it.
NUM_LEVELS = 150
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/CaI.csv')

Ca40n = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Ca40n.save_json('Ca_I_40.json')
