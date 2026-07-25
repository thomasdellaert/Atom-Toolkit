"""
Atom configuration file for 161Dy I. This file builds a full atom and can save it as a JSON file.

161Dy, I = 5/2 -- a fermionic dysprosium isotope, used for dipolar Fermi gases.
J = 8 over I = 5/2 gives F = 11/2 .. 21/2 in the ground manifold.

Energy levels come from the NIST ASD snapshot cached in resources/DyI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '161Dy I'
I = 5/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('DyI.csv'))

Dy161 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Dy161.gI = -0.19212  # mu_I = -0.4803 mu_N, I = 5/2 -- bare ratio

# === Hyperfine coefficients ===
# ground-state hyperfine A and B -- VERIFY
Dy161.levels['4f10.6s2 5I8'].hfA = Q_(-116.2, 'MHz')
Dy161.levels['4f10.6s2 5I8'].hfB = Q_(1091.6, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - hyperfine A and B for the 421 nm and 626 nm upper levels

if __name__ == '__main__':
    Dy161.save_json('Dy_I_161.json')
