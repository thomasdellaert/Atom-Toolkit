"""
Atom configuration file for 39K I. This file builds a full atom and can save it as a JSON file.

39K (I = 3/2) is the dominant potassium isotope. Its excited-state hyperfine
splittings are smaller than the 6 MHz natural linewidth, which makes sub-Doppler
cooling on the D2 line notoriously awkward.

Energy levels come from the NIST ASD snapshot cached in resources/KI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '39K I'
I = 3/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('KI.csv'))

K39 = IO.generate_full_from_dataframe(_energy_level_df,
                                      name=NAME,
                                      I=I,
                                      num_levels=NUM_LEVELS,
                                      allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
K39.gI = 0.260596  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 461.7197202 MHz
K39.levels['3p6.4s 2S1/2'].hfA = Q_(230.8598601, 'MHz')
# Falke et al., PRA 74, 032503 (2006)
K39.levels['3p6.4p 2P*1/2'].hfA = Q_(28.85, 'MHz')
# Falke et al., PRA 74, 032503 (2006)
K39.levels['3p6.4p 2P*3/2'].hfA = Q_(6.093, 'MHz')
K39.levels['3p6.4p 2P*3/2'].hfB = Q_(2.786, 'MHz')

if __name__ == '__main__':
    K39.save_json('K_I_39.json')
