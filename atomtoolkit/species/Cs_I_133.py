"""
Atom configuration file for 133Cs I. This file builds a full atom and can save it as a JSON file.

133Cs (I = 7/2) defines the SI second: the 6s 2S1/2 F=3 -> F=4 transition is
exactly 9192631770 Hz, so its hyperfine A is exactly 2298.1579425 MHz.

Energy levels come from the NIST ASD snapshot cached in resources/CsI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '133Cs I'
I = 7/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('CsI.csv'))

Cs133 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Cs133.gI = 0.732339  # measured atomic g_I (shielding-corrected), Steck; NOT the bare mu_I/I ratio

# === Hyperfine coefficients ===
# EXACT by SI definition: dE_HF = 9192631770 Hz = 4A
Cs133.levels['5p6.6s 2S1/2'].hfA = Q_(2298.1579425, 'MHz')
# Steck, Cesium D Line Data
Cs133.levels['5p6.6p 2P*1/2'].hfA = Q_(291.9201, 'MHz')
# Steck, Cesium D Line Data
Cs133.levels['5p6.6p 2P*3/2'].hfA = Q_(50.28827, 'MHz')
Cs133.levels['5p6.6p 2P*3/2'].hfB = Q_(-0.4934, 'MHz')

Cs133.levels['5p6.6s 2S1/2'].use_nonlinear_zeeman = True

if __name__ == '__main__':
    Cs133.save_json('Cs_I_133.json')
