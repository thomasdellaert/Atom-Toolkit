"""
Atom configuration file for 64Zn II. This file builds a full atom and can save it as a JSON file.

64Zn+ has I = 0. Zn+ is a group-12 ion like Cd+ and Hg+: a filled d shell below
the valence s electron, which puts low-lying d-hole states (3d9.4s2 2D) above
the ground state rather than the alkaline-earth-like d states below the P levels.

Energy levels come from the NIST ASD snapshot cached in resources/ZnII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '64Zn II'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('ZnII.csv'))

Zn64 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Zn64.save_json('Zn_II_64.json')
