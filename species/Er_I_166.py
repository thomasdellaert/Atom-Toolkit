"""
Atom configuration file for 166Er I. This file builds a full atom and can save it as a JSON file.

166Er, I = 0 -- a strongly magnetic lanthanide (mu = 7 mu_B, 3H6 ground state)
used for dipolar quantum gases alongside dysprosium. Broad MOT at 401 nm,
narrow-line MOT at 583 nm.

Energy levels come from the NIST ASD snapshot cached in resources/ErI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '166Er I'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/ErI.csv')

Er166 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Er166.save_json('Er_I_166.json')
