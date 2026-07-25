"""
Atom configuration file for 6Li I. This file builds a full atom and can save it as a JSON file.

6Li is a fermionic alkali (I = 1) widely used in ultracold Fermi-gas experiments.
Its D1/D2 lines near 671 nm are only ~10 GHz apart and the excited-state
hyperfine structure is smaller than the natural linewidth.

Energy levels come from the NIST ASD snapshot cached in resources/LiI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '6Li I'
I = 1
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/LiI.csv')

Li6 = IO.generate_full_from_dataframe(_energy_level_df,
                                      name=NAME,
                                      I=I,
                                      num_levels=NUM_LEVELS,
                                      allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Li6.gI = 0.821945  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# Arimondo et al., Rev. Mod. Phys. 49, 31 (1977)
Li6.levels['1s2.2s 2S1/2'].hfA = Q_(152.1368407, 'MHz')
# Gerhardt et al. / Arimondo compilation
Li6.levels['1s2.2p 2P*1/2'].hfA = Q_(17.386, 'MHz')
# Arimondo compilation
Li6.levels['1s2.2p 2P*3/2'].hfA = Q_(-1.155, 'MHz')
Li6.levels['1s2.2p 2P*3/2'].hfB = Q_(-0.10, 'MHz')

if __name__ == '__main__':
    Li6.save_json('Li_I_6.json')
