"""
Atom configuration file for 88Sr II. This file builds a full atom and can save it as a JSON file.

88Sr+ has I = 0 and a 674 nm S1/2 -> D5/2 quadrupole clock transition. Its
422 nm cooling light is convenient, making it a common optical-clock ion.

Energy levels come from the NIST ASD snapshot cached in resources/SrII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '88Sr II'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/SrII.csv')

Sr88 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

# === Quadrupole (E2) transitions ===
# ALLOWED_TRANSITIONS only auto-populates E1, so E2 clock transitions
# have to be added by hand.
# the 674 nm optical clock; A from the ~390 ms D5/2 lifetime
_t = Sr88.add_transition((Sr88.levels['4p6.5s 2S1/2'], Sr88.levels['4p6.4d 2D5/2']),
                         A=Q_(2.56, 'Hz'))
_t.set_frequency(Q_(444.779044095485, 'THz'))

if __name__ == '__main__':
    Sr88.save_json('Sr_II_88.json')
