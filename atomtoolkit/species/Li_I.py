"""
Atom configuration file. This file builds a full atom and can save it as a JSON file.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '7Li I'
I = 1.5
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('LiI.csv'))

Li7 = IO.generate_full_from_dataframe(_energy_level_df,
                                      name=NAME,
                                      I=I,
                                      num_levels=NUM_LEVELS,
                                      allowed=ALLOWED_TRANSITIONS)

# Override the frequencies of existing transitions

# Add new transitions

# t.set_frequency(Q_(394.423900, 'THz'))

# Set any additional properties

if __name__ == '__main__':
    Li7.save_json('7Li_I.json')
