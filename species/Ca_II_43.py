"""
Atom configuration file for 43Ca II. This file builds a full atom and can save it as a JSON file.

43Ca+ is the only stable Ca isotope with nuclear spin (I=7/2). Its 3.226 GHz
ground-state hyperfine manifold supports first-order field-independent
('clock') qubit transitions, which is why it is favoured for long-coherence work.

Energy levels come from the NIST ASD snapshot cached in resources/CaII.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO
import pathlib

absroot = pathlib.Path(__file__).parents[1]
RESOURCE_PATH = absroot.joinpath('resources')

NAME = '43Ca II'
I = 7/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(absroot / 'resources/CaII.csv')

Ca43 = IO.generate_full_from_dataframe(_energy_level_df,
                                       name=NAME,
                                       I=I,
                                       num_levels=NUM_LEVELS,
                                       allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Ca43.gI = -0.37647  # mu_I = -1.317643 mu_N, I = 7/2

# === Hyperfine coefficients ===
# ground-state hyperfine splitting 3.2256 GHz
Ca43.levels['3p6.4s 2S1/2'].hfA = Q_(-806.402077, 'MHz')
# Benhelm et al., PRA 75, 032506 (2007)
Ca43.levels['3p6.3d 2D3/2'].hfA = Q_(-47.3, 'MHz')
Ca43.levels['3p6.3d 2D3/2'].hfB = Q_(-3.7, 'MHz')
# Benhelm et al., PRA 75, 032506 (2007)
Ca43.levels['3p6.3d 2D5/2'].hfA = Q_(-3.8931, 'MHz')
Ca43.levels['3p6.3d 2D5/2'].hfB = Q_(-4.241, 'MHz')
# Arbes et al., Z. Phys. D 31, 27 (1994)
Ca43.levels['3p6.4p 2P*1/2'].hfA = Q_(-145.4, 'MHz')
# Arbes et al., Z. Phys. D 31, 27 (1994)
Ca43.levels['3p6.4p 2P*3/2'].hfA = Q_(-31.0, 'MHz')
Ca43.levels['3p6.4p 2P*3/2'].hfB = Q_(-6.9, 'MHz')

# The S1/2 hyperfine manifold is the usual home of the 43Ca+ clock qubit, so turn on
# exact H_HF + H_Z diagonalisation rather than the linear gF*mF*muB approximation.
Ca43.levels['3p6.4s 2S1/2'].use_nonlinear_zeeman = True

if __name__ == '__main__':
    Ca43.save_json('Ca_II_43.json')
