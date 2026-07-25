"""
Atom configuration file for 87Rb I. This file builds a full atom and can save it as a JSON file.

87Rb (I = 3/2) is the most-used atom in cold-atom physics. The 6.835 GHz
ground-state hyperfine splitting underpins both the |1,-1>/|2,1> magic-field
clock qubit and Rb vapour-cell frequency standards.

Energy levels come from the NIST ASD snapshot cached in resources/RbI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '87Rb I'
I = 3/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('RbI.csv'))

Rb87 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Rb87.gI = 1.827330  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 6.83468261090429 GHz
Rb87.levels['4p6.5s 2S1/2'].hfA = Q_(3417.34130545215, 'MHz')
# Steck, Rubidium 87 D Line Data
Rb87.levels['4p6.5p 2P*1/2'].hfA = Q_(407.24, 'MHz')
# Steck, Rubidium 87 D Line Data
Rb87.levels['4p6.5p 2P*3/2'].hfA = Q_(84.7185, 'MHz')
Rb87.levels['4p6.5p 2P*3/2'].hfB = Q_(12.4965, 'MHz')

# The famous |F=1,mF=-1> <-> |F=2,mF=+1> clock transition has a magic field near
# 3.229 G, which only shows up if the Zeeman shift is computed nonlinearly.
Rb87.levels['4p6.5s 2S1/2'].use_nonlinear_zeeman = True

if __name__ == '__main__':
    Rb87.save_json('Rb_I_87.json')
