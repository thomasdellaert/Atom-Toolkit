"""
Atom configuration file for 20Ne I. This file builds a full atom and can save it as a JSON file.

20Ne, I = 0 -- metastable neon.

IMPORTANT: as with He*, the trapped state is the metastable
2s2.2p5.(2P*<3/2>).3s 2[3/2]*2 level, not the 1S0 ground state. Cooling runs on
the 640.2 nm line out of that metastable state.

Neon's excited configurations are JK-coupled ('2[3/2]*'), which exercises a
different branch of atomtoolkit.term than the LS-coupled alkaline earths.

Energy levels come from the NIST ASD snapshot cached in resources/NeI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '20Ne I'
I = 0
# Only the low-lying 3s/3p manifolds matter here, so the cap is set low.
NUM_LEVELS = 150
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('NeI.csv'))

Ne20 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

if __name__ == '__main__':
    Ne20.save_json('Ne_I_20.json')
