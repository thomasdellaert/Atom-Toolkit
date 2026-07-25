"""
Atom configuration file for 52Cr I. This file builds a full atom and can save it as a JSON file.

52Cr, I = 0 -- the first strongly dipolar species to be Bose-condensed
(mu = 6 mu_B, 7S3 ground state). MOT on the 425.6 nm 7S3 -> 7P*4 line.
A transition-metal rather than a lanthanide, so its structure is comparatively
simple for a magnetic atom.

Energy levels come from the NIST ASD snapshot cached in resources/CrI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '52Cr I'
I = 0
# Cr I has ~480 tabulated levels and populate_transitions is O(N^2), so the cap is
# set low; the MOT and metastable D states are all well inside it.
NUM_LEVELS = 150
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/CrI.csv')

Cr52 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Cr52.save_json('Cr_I_52.json')
