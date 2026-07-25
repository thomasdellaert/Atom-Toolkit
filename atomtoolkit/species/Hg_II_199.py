"""
Atom configuration file for 199Hg II. This file builds a full atom and can save it as a JSON file.

199Hg+ (I=1/2) hosts two frequency standards: the 40.507 GHz ground-state
hyperfine transition (a microwave clock) and the 282 nm 2S1/2 -> 2D5/2
electric-quadrupole transition (an optical clock).

Note: NIST's Hg II tables contain doubly-nested JJ/JK term symbols such as
'[(2D<5/2>,1/2)<3>,1/2]' that atomtoolkit.term cannot parse. Those levels are
skipped with a warning at build time; everything below ~80000 cm-1, including
both clock states, parses fine.

Energy levels come from the NIST ASD snapshot cached in resources/HgII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '199Hg II'
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('HgII.csv'))

Hg199 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Hg199.gI = 1.01177  # mu_I = +0.5058855 mu_N, I = 1/2

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 40.507347996841 GHz (microwave clock)
Hg199.levels['5d10.6s 2S1/2'].hfA = Q_(40507.347996841, 'MHz')

# === Quadrupole (E2) transitions ===
# ALLOWED_TRANSITIONS only auto-populates E1, so E2 clock transitions
# have to be added by hand.
# the 282 nm optical clock; A from the ~90 ms D5/2 lifetime
_t = Hg199.add_transition((Hg199.levels['5d10.6s 2S1/2'], Hg199.levels['5d9.6s2 2D5/2']),
                         A=Q_(11.1, 'Hz'))
_t.set_frequency(Q_(1064.721609899145, 'THz'))

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 5d9.6s2 2D5/2 hyperfine A and B

if __name__ == '__main__':
    Hg199.save_json('Hg_II_199.json')
