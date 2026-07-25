"""
Atom configuration file for 4He I. This file builds a full atom and can save it as a JSON file.

4He, I = 0 -- metastable helium.

IMPORTANT: the trapped state is not the ground state. He* experiments work with
the metastable 1s.2s 3S1 level (lifetime ~7900 s), cooled on the 1083 nm
1s.2s 3S1 -> 1s.2p 3P*2 line. This module builds the atom from the true 1s2 1S0
ground state as usual, so reference the metastable level explicitly:

    ms = He4.levels['1s.2s 3S1']

Its ~20 eV of internal energy is what allows single-atom detection on a
microchannel plate.

Energy levels come from the NIST ASD snapshot cached in resources/HeI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '4He I'
I = 0
# He I has ~840 tabulated levels, almost all high Rydberg states of no practical
# interest here, so the cap is set low.
NUM_LEVELS = 150
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('HeI.csv'))

He4 = IO.generate_full_from_dataframe(_energy_level_df,
                                      name=NAME,
                                      I=I,
                                      num_levels=NUM_LEVELS,
                                      allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    He4.save_json('He_I_4.json')
