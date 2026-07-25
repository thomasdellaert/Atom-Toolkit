"""
Atom configuration file for 23Na I. This file builds a full atom and can save it as a JSON file.

23Na (I = 3/2) is the classic laser-cooled alkali; its D2 line at 589 nm and
1.772 GHz ground-state hyperfine splitting are textbook reference values.

Energy levels come from the NIST ASD snapshot cached in resources/NaI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '23Na I'
I = 3/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('NaI.csv'))

Na23 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Na23.gI = 1.477398  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 1771.6261288 MHz
Na23.levels['2p6.3s 2S1/2'].hfA = Q_(885.8130644, 'MHz')
# Arimondo et al., Rev. Mod. Phys. 49, 31 (1977)
Na23.levels['2p6.3p 2P*1/2'].hfA = Q_(94.44, 'MHz')
# Arimondo et al., Rev. Mod. Phys. 49, 31 (1977)
Na23.levels['2p6.3p 2P*3/2'].hfA = Q_(18.534, 'MHz')
Na23.levels['2p6.3p 2P*3/2'].hfB = Q_(2.724, 'MHz')

if __name__ == '__main__':
    Na23.save_json('Na_I_23.json')
