"""
Atom configuration file. This file should be able to build a full atom and save it as a .atom file.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')
# TRANSITIONS_PATH = absroot.joinpath('resources/Yb_II_Oscillator_Strengths.csv')
# HYPERFINE_PATH = absroot.joinpath('resources/171Yb_Hyperfine.csv')

NAME = '7Li I'
I = 1.5
NUM_LEVELS = 300
B_FIELD = Q_(0.0, 'G')
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data('Li I')
# _transition_df = IO.load_transition_data(
#     TRANSITIONS_PATH,
#     columns={
#             "conf_l": "LConfiguration", "conf_u": "UConfiguration",
#             "term_l": "LTerm", "term_u": "UTerm", "j_l": "LJ", "j_u": "UJ",
#             "A": "A DREAM"
#     }).dropna(subset=['A'])

Yb171 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        B=B_FIELD,
                                        allowed=ALLOWED_TRANSITIONS)

# Override the frequencies of existing transitions

# Add new transitions

# t.set_frequency(Q_(394.423900, 'THz'))

# Set any additional properties

if __name__ == '__main__':
    Yb171.save('7Li_I.atom')
