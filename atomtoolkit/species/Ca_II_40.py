"""
Atom configuration file for 40Ca II. This file builds a full atom and can save it as a JSON file.

40Ca+ is the most common trapped-ion species after Yb+. I = 0, so there is no
hyperfine structure; the qubit is either the 397 nm Zeeman qubit or the
729 nm S1/2 -> D5/2 optical quadrupole transition.

Energy levels come from the NIST ASD snapshot cached in resources/CaII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '40Ca II'
I = 0
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('CaII.csv'))

Ca40 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# I = 0: no hyperfine structure, so there is nothing to set here.

# === Quadrupole (E2) transitions ===
# ALLOWED_TRANSITIONS only auto-populates E1, so E2 clock transitions
# have to be added by hand.
# the 729 nm clock qubit; A from the 1.168 s D5/2 lifetime
_t = Ca40.add_transition((Ca40.levels['3p6.4s 2S1/2'], Ca40.levels['3p6.3d 2D5/2']),
                         A=Q_(0.856, 'Hz'))
_t.set_frequency(Q_(411.042129776393, 'THz'))

if __name__ == '__main__':
    Ca40.save_json('Ca_II_40.json')
