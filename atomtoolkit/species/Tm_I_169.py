"""
Atom configuration file for 169Tm I. This file builds a full atom and can save it as a JSON file.

169Tm, I = 1/2 -- neutral thulium, laser-cooled on the 410.6 nm line. Its
1.14 micron 4f13.6s2 2F*7/2 -> 2F*5/2 magnetic-dipole transition is an inner-shell
clock transition that is unusually well shielded from external fields by the
filled 6s2 shell, which is the main reason thulium is of interest.

Note this is neutral Tm; species/Tm_II_169.py holds the singly-ionised species,
which is not MOT-trapped.

NIST's Tm I table contains many JK-coupled levels the parser cannot handle; they
are skipped with a warning. The ground term and the 1.14 um upper state parse fine.

Energy levels come from the NIST ASD snapshot cached in resources/TmI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '169Tm I'
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('TmI.csv'))

Tm169n = IO.generate_full_from_dataframe(_energy_level_df,
                                         name=NAME,
                                         I=I,
                                         num_levels=NUM_LEVELS,
                                         allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Tm169n.gI = -0.4632  # mu_I = -0.2316 mu_N, I = 1/2 -- bare ratio

# === Hyperfine coefficients ===
# ground-state hyperfine A -- VERIFY
Tm169n.levels['4f13.(2F*).6s2 2F*7/2'].hfA = Q_(-374.137, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 4f13.(2F*).6s2 2F*5/2 hyperfine A (the 1.14 um clock upper state)
#   - an A coefficient for the 1.14 um M1 clock transition

if __name__ == '__main__':
    Tm169n.save_json('Tm_I_169.json')
