"""
Atom configuration file for 41K I. This file builds a full atom and can save it as a JSON file.

41K (I = 3/2) is the second stable bosonic potassium isotope.

Energy levels come from the NIST ASD snapshot cached in resources/KI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '41K I'
I = 3/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('KI.csv'))

K41 = IO.generate_full_from_dataframe(_energy_level_df,
                                      name=NAME,
                                      I=I,
                                      num_levels=NUM_LEVELS,
                                      allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
K41.gI = 0.143046  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 254.0138704 MHz
K41.levels['3p6.4s 2S1/2'].hfA = Q_(127.0069352, 'MHz')
# Falke et al., PRA 74, 032503 (2006)
K41.levels['3p6.4p 2P*1/2'].hfA = Q_(15.245, 'MHz')
# Falke et al., PRA 74, 032503 (2006)
K41.levels['3p6.4p 2P*3/2'].hfA = Q_(3.363, 'MHz')
K41.levels['3p6.4p 2P*3/2'].hfB = Q_(3.351, 'MHz')

if __name__ == '__main__':
    K41.save_json('K_I_41.json')
