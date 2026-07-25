"""
Atom configuration file for 137Ba II. This file builds a full atom and can save it as a JSON file.

137Ba+ has I = 3/2 and an 8.038 GHz ground-state hyperfine splitting. Barium's
visible-wavelength transitions (493 nm, 650 nm) are unusually convenient.

Energy levels come from the NIST ASD snapshot cached in resources/BaII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '137Ba II'
I = 3/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('BaII.csv'))

Ba137 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Ba137.gI = 0.62491  # mu_I = +0.937365 mu_N, I = 3/2

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 8.037742 GHz
Ba137.levels['6s 2S1/2'].hfA = Q_(4018.871, 'MHz')
# Silverans et al., PRL 55, 1911 (1985)
Ba137.levels['5d 2D3/2'].hfA = Q_(189.7304, 'MHz')
Ba137.levels['5d 2D3/2'].hfB = Q_(44.5417, 'MHz')
# Silverans et al., PRL 55, 1911 (1985)
Ba137.levels['5d 2D5/2'].hfA = Q_(-12.028, 'MHz')
Ba137.levels['5d 2D5/2'].hfB = Q_(59.52, 'MHz')
# Villemoes et al., J. Phys. B 26, 4289 (1993)
Ba137.levels['6p 2P*1/2'].hfA = Q_(743.7, 'MHz')
# Villemoes et al., J. Phys. B 26, 4289 (1993)
Ba137.levels['6p 2P*3/2'].hfA = Q_(127.2, 'MHz')
Ba137.levels['6p 2P*3/2'].hfB = Q_(92.5, 'MHz')

Ba137.levels['6s 2S1/2'].use_nonlinear_zeeman = True

if __name__ == '__main__':
    Ba137.save_json('Ba_II_137.json')
