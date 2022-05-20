"""
Atom configuration file. This file should be able to build a full atom and save it as a .atom file.
"""

from atomtoolkit import Q_, IO
from atomtoolkit.atom import Transition
import os

RESOURCE_PATH = os.path.abspath('../resources')
TRANSITIONS_PATH = os.path.abspath('../resources/Yb_II_Oscillator_Strengths.csv')
HYPERFINE_PATH = os.path.abspath('../resources/171Yb_Hyperfine.csv')

NAME = '171Yb II'
I = 1/2
NUM_LEVELS = 300
B_FIELD = Q_(0.0, 'G')
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data('Yb II')
_transition_df = IO.load_transition_data(
    TRANSITIONS_PATH,
    columns={
            "conf_l": "LConfiguration", "conf_u": "UConfiguration",
            "term_l": "LTerm", "term_u": "UTerm", "j_l": "LJ", "j_u": "UJ",
            "A": "A DREAM"
    }).dropna(subset=['A'])

Yb171 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        B=B_FIELD,
                                        hf_csv=HYPERFINE_PATH,
                                        transitions_df=_transition_df,
                                        allowed=ALLOWED_TRANSITIONS)

# Override the frequencies of existing transitions
Yb171.transitions[('4f14.5d 2D3/2', '4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2')].set_frequency(Q_(320.567693, 'THz'))
Yb171.transitions[('4f14.6s 2S1/2', '4f14.6p 2P*1/2')].set_frequency(Q_(811.294110, 'THz'))

# Add new transitions
t = Transition(Yb171.levels['4f14.6s 2S1/2'], Yb171.levels['4f14.5d 2D5/2'], A=Q_(22, 'Hz'))
t.add_to_atom(Yb171)
t.set_frequency(Q_(729.476090, 'THz'))

# Set any additional properties
Yb171.levels['4f14.6s 2S1/2']['F=0']['mF=0'].quadratic_zeeman = -155.305
Yb171.levels['4f14.6s 2S1/2']['F=1']['mF=0'].quadratic_zeeman = 155.305

if __name__ == '__main__':
    Yb171.save('Yb_II_171.atom')
