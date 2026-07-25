"""
Atom configuration file for 164Dy I. This file builds a full atom and can save it as a JSON file.

164Dy, I = 0 -- the most magnetic neutral atom (mu = 10 mu_B in the 5I8 ground
state), which makes it a workhorse for dipolar quantum gases, quantum droplets and
supersolids. Broad MOT at 421.3 nm, narrow-line MOT at 626.1 nm.
The J = 8 ground state gives 17 Zeeman sublevels even with no nuclear spin.

Energy levels come from the NIST ASD snapshot cached in resources/DyI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '164Dy I'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/DyI.csv')

Dy164 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Dy164.save_json('Dy_I_164.json')
