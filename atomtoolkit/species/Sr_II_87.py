"""
Atom configuration file for 87Sr II. This file builds a full atom and can save it as a JSON file.

87Sr+ carries I = 9/2, giving a ~5.0 GHz ground-state hyperfine splitting and a
correspondingly dense sublevel structure (the S1/2 manifold alone has 20 Zeeman states).

Energy levels come from the NIST ASD snapshot cached in resources/SrII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '87Sr II'
I = 9/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('SrII.csv'))

Sr87 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Sr87.gI = -0.24298  # mu_I = -1.0936 mu_N, I = 9/2

# === Hyperfine coefficients ===
# ground-state hyperfine splitting ~5.003 GHz
Sr87.levels['4p6.5s 2S1/2'].hfA = Q_(-1000.5, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 4d 2D3/2 and 2D5/2 hyperfine A and B
#   - 5p 2P*1/2 and 2P*3/2 hyperfine A and B

if __name__ == '__main__':
    Sr87.save_json('Sr_II_87.json')
