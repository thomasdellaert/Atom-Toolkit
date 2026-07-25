"""
Atom configuration file for 40K I. This file builds a full atom and can save it as a JSON file.

40K is the rare fermionic potassium isotope (I = 4, the largest nuclear spin in
this library). Its hyperfine A is negative, so the level ordering within the
ground-state manifold is inverted relative to 39K and 41K.

Energy levels come from the NIST ASD snapshot cached in resources/KI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '40K I'
I = 4
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/KI.csv')

K40 = IO.generate_full_from_dataframe(_energy_level_df,
                                      name=NAME,
                                      I=I,
                                      num_levels=NUM_LEVELS,
                                      allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
K40.gI = -0.324073  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# Arimondo et al., Rev. Mod. Phys. 49, 31 (1977)
K40.levels['3p6.4s 2S1/2'].hfA = Q_(-285.7308, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 4p 2P*1/2 and 2P*3/2 hyperfine A and B for 40K

if __name__ == '__main__':
    K40.save_json('K_I_40.json')
