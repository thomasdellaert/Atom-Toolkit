"""
Atom configuration file for 202Hg II. This file builds a full atom and can save it as a JSON file.

202Hg+ is the most abundant mercury isotope and has I = 0.

See the note in Hg_II_199.py about unparseable high-lying term symbols.

Energy levels come from the NIST ASD snapshot cached in resources/HgII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '202Hg II'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/HgII.csv')

Hg202 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Hg202.save_json('Hg_II_202.json')
