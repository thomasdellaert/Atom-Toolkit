"""
Atom configuration file for 112Cd I. This file builds a full atom and can save it as a JSON file.

112Cd, I = 0. MOT on the 228.8 nm 1S0 -> 1P*1 line, with a 326.1 nm 1S0 -> 3P*1
intercombination line. Cadmium is a proposed optical clock species with small
blackbody sensitivity.

Energy levels come from the NIST ASD snapshot cached in resources/CdI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '112Cd I'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('CdI.csv'))

Cd112 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Cd112.save_json('Cd_I_112.json')
