"""
Atom configuration file for 24Mg II. This file builds a full atom and can save it as a JSON file.

24Mg+ is the spin-zero (I=0) isotope: no hyperfine structure, so the level
diagram is pure fine structure. Widely used as a sympathetic coolant.

Energy levels come from the NIST ASD snapshot cached in resources/MgII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '24Mg II'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/MgII.csv')

Mg24 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Mg24.save_json('Mg_II_24.json')
