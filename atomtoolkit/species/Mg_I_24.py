"""
Atom configuration file for 24Mg I. This file builds a full atom and can save it as a JSON file.

24Mg, I = 0. MOT on the 285.2 nm 1S0 -> 1P*1 line; the 457.1 nm 1S0 -> 3P*1
intercombination line is the narrow-line cooling and clock candidate.
The deep-UV cooling wavelength is what makes magnesium awkward in practice.

Energy levels come from the NIST ASD snapshot cached in resources/MgI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '24Mg I'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('MgI.csv'))

Mg24n = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Mg24n.save_json('Mg_I_24.json')
