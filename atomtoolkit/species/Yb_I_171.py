"""
Atom configuration file for 171Yb I. This file builds a full atom and can save it as a JSON file.

171Yb, I = 1/2 -- an optical lattice clock species and a leading neutral-atom
qubit. I = 1/2 gives the simplest possible hyperfine structure, and the nuclear
spin itself serves as a long-lived qubit in tweezer arrays. The 578.4 nm
1S0 -> 3P*0 clock transition is hyperfine-induced, as in 87Sr.

Energy levels come from the NIST ASD snapshot cached in resources/YbI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '171Yb I'
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('YbI.csv'))

Yb171n = IO.generate_full_from_dataframe(_energy_level_df,
                                         name=NAME,
                                         I=I,
                                         num_levels=NUM_LEVELS,
                                         allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Yb171n.gI = -0.9951390  # Blaum et al. (2013); same nucleus as species/Yb_II_171.py

# === Hyperfine coefficients ===
# Yb I 3P1 hyperfine A -- VERIFY
Yb171n.levels['4f14.6s.6p 3P*1'].hfA = Q_(3957.97, 'MHz')

# === Forbidden clock transitions ===
# J=0 -> J=0 is forbidden for every multipole, so this is never populated
# automatically; it is only weakly allowed through hyperfine mixing in the
# odd isotopes, and has to be added by hand.
# the 578 nm clock transition, a BIPM secondary representation of the second
_t = Yb171n.add_transition((Yb171n.levels['4f14.6s2 1S0'], Yb171n.levels['4f14.6s.6p 3P*0']),
                         A=Q_(0.043, 'Hz'))
_t.set_frequency(Q_(518.2958365908636, 'THz'))

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 4f14.6s.6p 3P*2 and 1P*1 hyperfine A

if __name__ == '__main__':
    Yb171n.save_json('Yb_I_171.json')
