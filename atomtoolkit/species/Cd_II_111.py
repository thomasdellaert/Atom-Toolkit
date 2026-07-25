"""
Atom configuration file for 111Cd II. This file builds a full atom and can save it as a JSON file.

111Cd+ (I=1/2) has a 14.53 GHz ground-state hyperfine splitting used as a
microwave frequency standard. Included alongside Zn+ and Hg+ to complete group 12.

Energy levels come from the NIST ASD snapshot cached in resources/CdII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '111Cd II'
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('CdII.csv'))

Cd111 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Cd111.gI = -1.1886  # mu_I = -0.5943 mu_N, I = 1/2

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 14.5305 GHz; for I=J=1/2, dE_HF = A
Cd111.levels['4d10.5s 2S1/2'].hfA = Q_(14530.5, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 5p 2P*1/2 and 2P*3/2 hyperfine A and B

if __name__ == '__main__':
    Cd111.save_json('Cd_II_111.json')
