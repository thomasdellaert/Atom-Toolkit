"""
Atom configuration file for 9Be II. This file builds a full atom and can save it as a JSON file.

9Be+ is the lightest commonly trapped ion. Its 313 nm cooling transition and
1.25 GHz ground-state hyperfine splitting make it a workhorse for quantum logic
and for sympathetic cooling of heavier co-trapped ions.

Energy levels come from the NIST ASD snapshot cached in resources/BeII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '9Be II'
I = 3/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('BeII.csv'))

Be9 = IO.generate_full_from_dataframe(_energy_level_df,
                                      name=NAME,
                                      I=I,
                                      num_levels=NUM_LEVELS,
                                      allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Be9.gI = -0.78495  # mu_I = -1.177432 mu_N, I = 3/2 -- bare ratio, not shielding-corrected

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 1.2500 GHz; A = -dE_HF/(2I+1)
Be9.levels['1s2.2s 2S1/2'].hfA = Q_(-625.008837048, 'MHz')

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - 2p 2P*1/2 and 2P*3/2 hyperfine A (and B for 2P*3/2)

if __name__ == '__main__':
    Be9.save_json('Be_II_9.json')
