"""
Atom configuration file for 165Ho I. This file builds a full atom and can save it as a JSON file.

165Ho -- the deliberately difficult species in this library.

Holmium is hard in every way this toolkit can be stressed:

  * The ground state is 4f11.6s2 4I*15/2 -- J = 15/2 with I = 7/2 gives F = 4..11,
    i.e. 128 hyperfine-Zeeman sublevels in the ground manifold alone. That is the
    largest ground-state Hilbert space of any neutral atom, and it is exactly why
    Ho is studied as a qudit platform.
  * Excited configurations mix 4f, 5d and 6s electrons, so NIST reports them in
    JK/JJ coupling ('4f10.(5I<8>).5d<3/2>.6s2 (8,3/2)') rather than LS. This
    exercises the non-LS branches of atomtoolkit.term and of compute_gJ.
  * Its hyperfine B coefficient is enormous and negative (-1668 MHz against
    A = +800 MHz), so the electric-quadrupole term is *not* a small correction to
    the magnetic-dipole term. The ground manifold spans ~45 GHz and badly violates
    the Lande interval rule: intervals run 4.31 -> 8.28 GHz (ratio 1.92) where the
    dipole-only rule would give a ratio of 2.2. The linear-Zeeman approximation
    correspondingly fails at modest fields.
  * Some high-lying levels carry term symbols the parser cannot handle; they are
    skipped with a warning rather than aborting the build.

Treat this file as a stress test as much as a data source.

Energy levels come from the NIST ASD snapshot cached in resources/HoI.csv.
Hyperfine coefficients and nuclear g-factors are from the literature; each is cited inline.
"""

from atomtoolkit import Q_, IO, resource_path

NAME = '165Ho I'
I = 7/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data_from_csv(resource_path('HoI.csv'))

Ho165 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# === Nuclear g-factor (nuclear magneton units, signed) ===
Ho165.gI = 1.19195  # mu_I = +4.173 mu_N, I = 7/2

# === Hyperfine coefficients ===
# Dankwort & Ferch, Z. Phys. 267, 229 (1974) -- VERIFY
Ho165.levels['4f11.6s2 4I*15/2'].hfA = Q_(800.583, 'MHz')
Ho165.levels['4f11.6s2 4I*15/2'].hfB = Q_(-1668.0, 'MHz')

# With |B| > 2|A| the quadrupole term dominates, so the linear gF*mF*muB form is a
# poor approximation almost immediately. Diagonalise H_HF + H_Z exactly instead.
Ho165.levels['4f11.6s2 4I*15/2'].use_nonlinear_zeeman = True

# === Data still needed ===
# The following are not yet filled in. Add them here (or via a
# resources/*_Hyperfine.csv and IO.apply_hf_csv) when you have values:
#   - hyperfine A and B for the excited 4f11.6s2 4I* levels (J = 13/2, 11/2, 9/2)
#   - transition A coefficients -- NIST has no usable line list for Ho I here

if __name__ == '__main__':
    Ho165.save_json('Ho_I_165.json')
