"""
Atom configuration file. This file builds a full atom and can save it as a JSON file.
"""

from atomtoolkit import Q_, IO, resource_path

TRANSITIONS_PATH = resource_path('TmII_transitions_with_ASD_terms.csv')

NAME = '169Tm II'
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('TmII.csv'))
_transition_df = IO.load_transition_data(
    TRANSITIONS_PATH,
    columns={
            "conf_l": "lower_configuration", "conf_u": "upper_configuration",
            "term_l": "lower_term_symbol", "term_u": "upper_term_symbol",
            "j_l": "lower_J", "j_u": "upper_J",
            "A": "A"
    }).dropna(subset=['A'])

Tm169 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        transitions_df=_transition_df,
                                        allowed=ALLOWED_TRANSITIONS)

# Override the frequencies of existing transitions

# Add new transitions

# Set any additional properties

if __name__ == '__main__':
    Tm169.save_json('Tm_II_169.json')
