"""
Atom configuration file. This file should be able to build a full atom and save it as a .atom file.
"""

from atomtoolkit import Q_, IO
from atomtoolkit.atom import Transition
import os

RESOURCE_PATH = os.path.abspath('../resources')
TRANSITIONS_PATH = os.path.abspath('../resources/Yb_II_Oscillator_Strengths.csv')
HYPERFINE_PATH = os.path.abspath('../resources/173Yb_Hyperfine.csv')

NAME = '173Yb II'
I = 5/2
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

Yb173 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        B=B_FIELD,
                                        hf_csv=HYPERFINE_PATH,
                                        transitions_df=_transition_df,
                                        allowed=ALLOWED_TRANSITIONS)

# Override the frequencies of existing transitions
Yb173.transitions[('4f14.5d 2D3/2', '4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2')].set_frequency(Q_(320.570339, 'THz'))
Yb173.transitions[('4f14.6s 2S1/2', '4f14.6p 2P*1/2')].set_frequency(Q_(811.292506, 'THz'))

# Add new transitions
t = Transition(Yb173.levels['4f14.6s 2S1/2'], Yb173.levels['4f14.5d 2D5/2'], A=Q_(22, 'Hz'))
t.add_to_atom(Yb173)
t.set_frequency(Q_(729.476090, 'THz'))
t = Transition(Yb173.levels['4f13.(2F*).6s2 2F*7/2'], Yb173.levels['4f13.(2F*<7/2>).5d.6s.(1D) 1[3/2]*3/2'], A=Q_(50, 'kHz'))
t.add_to_atom(Yb173)
# t.set_frequency(Q_(394.423900, 'THz'))

# Set any additional properties

if __name__ == '__main__':
    Yb173.save('Yb_II_173.atom')
