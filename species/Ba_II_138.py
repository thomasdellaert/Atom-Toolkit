"""
Atom configuration file for 138Ba II. This file builds a full atom and can save it as a JSON file.

138Ba+ is the most abundant barium isotope and has I = 0, so it shows pure fine
structure. Commonly used where hyperfine structure would only be a nuisance.

Energy levels come from the NIST ASD snapshot cached in resources/BaII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '138Ba II'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/BaII.csv')

Ba138 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Ba138.save_json('Ba_II_138.json')
