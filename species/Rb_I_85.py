"""
Atom configuration file for 85Rb I. This file builds a full atom and can save it as a JSON file.

85Rb (I = 5/2) is the more abundant rubidium isotope, with a 3.036 GHz
ground-state hyperfine splitting.

Energy levels come from the NIST ASD snapshot cached in resources/RbI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '85Rb I'
I = 5/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/RbI.csv')

Rb85 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Rb85.gI = 0.539183  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 3.0357324390 GHz
Rb85.levels['4p6.5s 2S1/2'].hfA = Q_(1011.9108130, 'MHz')
# Steck, Rubidium 85 D Line Data
Rb85.levels['4p6.5p 2P*1/2'].hfA = Q_(120.527, 'MHz')
# Steck, Rubidium 85 D Line Data
Rb85.levels['4p6.5p 2P*3/2'].hfA = Q_(25.0020, 'MHz')
Rb85.levels['4p6.5p 2P*3/2'].hfB = Q_(25.790, 'MHz')

if __name__ == '__main__':
    Rb85.save_json('Rb_I_85.json')
