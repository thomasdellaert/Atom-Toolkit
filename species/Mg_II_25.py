"""
Atom configuration file for 25Mg II. This file builds a full atom and can save it as a JSON file.

25Mg+ is the only stable Mg isotope with nuclear spin (I=5/2), giving it a
1.789 GHz ground-state hyperfine splitting usable as a microwave qubit.

Energy levels come from the NIST ASD snapshot cached in resources/MgII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '25Mg II'
I = 5/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/MgII.csv')

Mg25 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Mg25.gI = -0.34218  # mu_I = -0.85545 mu_N, I = 5/2

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 1.7887 GHz
Mg25.levels['2p6.3s 2S1/2'].hfA = Q_(-596.254376, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 3p 2P*1/2 and 2P*3/2 hyperfine A (and B for 2P*3/2)

if __name__ == '__main__':
    Mg25.save_json('Mg_II_25.json')
