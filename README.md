# Atom-Toolkit

A Python package for manipulating atomic physics data in a notebook environment. It models atomic structure as a hierarchy of [NetworkX](https://networkx.org/) graphs — gross-structure energy levels at the top, with hyperfine and Zeeman sublevels generated automatically on demand — and provides a convenient interface for working with spectroscopic data, transitions, and selection rules.

## Features

- **Load data directly from NIST ASD** — fetch energy levels and transition data for any ion/neutral with a single call
- **Three-layer level hierarchy** — `EnergyLevel` → `HFLevel` → `ZLevel` (Zeeman), each layer generated lazily the first time it is accessed
- **Rich key access** for levels and transitions:
  ```python
  atom.levels['4f14.6s 2S1/2']                  # gross level
  atom.levels['4f14.6s 2S1/2 F=0']              # hyperfine sublevel
  atom.levels['4f14.6s 2S1/2 F=0 mF=0']         # flat shorthand
  atom.levels['4f14.6s 2S1/2']['F=0']           # hyperfine sublevel
  atom.levels['4f14.6s 2S1/2']['F=0']['mF=0']   # flat shorthand
  atom.levels[1][0]                             # indexing by energy
  atom.levels[0:4]                              # indexing by energy
  atom.transitions[level_a, level_b]            # order-independent
  ```
- **Automatic sublevel transitions** — setting a frequency or Einstein A coefficient on an `EnergyLevel` transition propagates down to all HF and Zeeman sub-transitions, applying the correct selection rules and angular-momentum weights
- **Support for LS, JJ, LK, and JK couplings**, with quantum numbers extracted automatically from NIST-format configuration and term strings
- **Branching ratios and lifetimes** computed from A coefficients
- **Grotrian diagrams and spectra** — matplotlib-based plots of level structure and spectral lineshapes (in development)
- **Serialisation** — save/load atoms to disk as JSON for fast repeated use

## Installation

```bash
pip install -e .
```

Dependencies: `pint`, `pint-pandas`, `pandas`, `numpy`, `networkx`, `sympy`, `matplotlib`, `scipy`, `tqdm`

## Loading NIST data

The primary loader fetches data from the NIST ASD web API:

```python
from atomtoolkit import IO
df = IO.load_NIST_data('Yb II')
```

If the API is unavailable (e.g. returns a 403 error), you can download a CSV manually instead:

1. Go to https://physics.nist.gov/PhysRefData/ASD/levels_form.html
2. Enter the species name (e.g. `Yb II`), enable all output fields, and set **Format = CSV**
3. Save the file and load it with:

```python
df = IO.load_NIST_data_from_csv('path/to/YbII.csv')
```

The CSVs in `atomtoolkit/resources/` were obtained this way and are used automatically by
the species modules, so nothing under `atomtoolkit/species/` needs network access.

A note on the web API: `load_NIST_data()` talks to `energy1.pl`, which is the ASD search
form's own backend rather than a documented, versioned API — NIST can change it without
notice. Treat it as a convenience for adding a *new* species (fetch once, save the CSV to
`atomtoolkit/resources/`, commit it) rather than something to call at runtime. It also sends an
identifying `User-Agent`, because NIST rejects the default `Python-urllib` one; please leave
that honest rather than swapping in a browser string.

## Quick start

```python
from atomtoolkit.atom import Atom
from atomtoolkit import IO, Q_, resource_path

# Load level data — from the API or a local CSV
df = IO.load_NIST_data('Yb II')               # live API
# df = IO.load_NIST_data_from_csv(resource_path('YbII.csv'))  # local fallback
a = Atom.from_dataframe(df, name='173Yb II', I=2.5)

# Set hyperfine coefficients on a gross level
d32 = a.levels['4f14.5d 2D3/2']
d32.hfA = Q_(-0.11, 'GHz')
d32.hfB = Q_(0.95, 'GHz')

b12 = a.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2']
b12.hfA = Q_(0.61, 'GHz')

# Set transition data — HF and Zeeman sub-transitions are updated automatically
repump_935 = a.transitions[b12, d32]
repump_935.A = Q_(120, 'kHz')

# Access individual sublevels
print(d32['F=2']['mF=-1'].level)      # Hz energy including HF shift
print(repump_935.subtransitions())    # all HF sub-transitions with selection rules
```

## Data model

```
Atom
├── EnergyLevel  (gross structure — from NIST, in cm^-1 or Hz)
│   └── HFLevel  (hyperfine — generated from A, B, C coefficients)
│       └── ZLevel   (Zeeman — generated from gF factor)
│
└── Transition   (gross — carries A coefficient, freq)
    └── HFTransition
        └── ZTransition  (carries geometric coupling factors)
```

All quantities use a shared [pint](https://pint.readthedocs.io/) unit registry with a spectroscopy context, so `.to('Hz')` and `.to('cm**-1')` conversions work everywhere. The `Q_()` constructor creates dimensioned quantities using this registry.

## Species files

The `atomtoolkit/species/` package contains ready-to-use atoms with hyperfine coefficients, nuclear
g-factors, and corrected transitions already applied. The registry is the easiest way in —
no module names or file paths required:

```python
import atomtoolkit

atomtoolkit.available_species()             # -> ['133Ba II', '112Cd I', '133Cs I', ...]
yb = atomtoolkit.load_species('Yb II 171')  # -> Atom
sr = atomtoolkit.load_species('87Sr I')
```

Name matching is forgiving: `'Yb II 171'`, `'171Yb II'`, `'Yb_II_171'` and `'yb ii 171'` all
resolve to the same species, and an unknown name raises a `KeyError` suggesting the closest
matches. Listing is instant — it reads one line from each file and imports nothing.

You can still import a module directly if you prefer; each exposes its `Atom` under a short
name:

```python
from atomtoolkit.species import Yb_II_171, Ca_II_43
a = Yb_II_171.Yb171
b = Ca_II_43.Ca43
```

Energy levels come from NIST ASD CSVs cached in `atomtoolkit/resources/`, so building a species never
touches the network.

`load_species()` prefers a prebuilt `.json` snapshot when one exists and is newer than its
`.py` — that turns a ~40 s Rb build into a ~1 s load — and otherwise builds from the CSV.
Snapshots are gitignored, so a fresh clone just builds; regenerate them with
`atomtoolkit.build_snapshots()` or by running a species file as `__main__`
(`python -m atomtoolkit.species.Ca_II_43`). Loaded atoms are
cached and shared, so pass `copy=True` if you intend to mutate one.

The species library and its data ship inside the package (`atomtoolkit/species/` and
`atomtoolkit/resources/`), so everything above works from an installed wheel as well as from
a checkout. Data files are located with `atomtoolkit.resource_path()` rather than by walking
`__file__`.

### Bundled species

**Trapped ions — alkaline-earth-like** (valence s electron over a closed shell)

| Module | Species | I | Module | Species | I |
|---|---|---|---|---|---|
| `Be_II_9` | ⁹Be⁺ | 3/2 | `Ba_II_133` | ¹³³Ba⁺ | 1/2 |
| `Mg_II_24` | ²⁴Mg⁺ | 0 | `Ba_II_137` | ¹³⁷Ba⁺ | 3/2 |
| `Mg_II_25` | ²⁵Mg⁺ | 5/2 | `Ba_II_138` | ¹³⁸Ba⁺ | 0 |
| `Ca_II_40` | ⁴⁰Ca⁺ | 0 | `Ra_II_225` | ²²⁵Ra⁺ | 1/2 |
| `Ca_II_43` | ⁴³Ca⁺ | 7/2 | `Yb_II_171` | ¹⁷¹Yb⁺ | 1/2 |
| `Sr_II_87` | ⁸⁷Sr⁺ | 9/2 | `Yb_II_173` | ¹⁷³Yb⁺ | 5/2 |
| `Sr_II_88` | ⁸⁸Sr⁺ | 0 | `Yb_II_174` | ¹⁷⁴Yb⁺ | 0 |

**Trapped ions — group 12** (d-hole states sit *above* the ground state, not below the P levels)

| Module | Species | I | Module | Species | I |
|---|---|---|---|---|---|
| `Zn_II_64` | ⁶⁴Zn⁺ | 0 | `Hg_II_199` | ¹⁹⁹Hg⁺ | 1/2 |
| `Zn_II_67` | ⁶⁷Zn⁺ | 5/2 | `Hg_II_202` | ²⁰²Hg⁺ | 0 |
| `Cd_II_111` | ¹¹¹Cd⁺ | 1/2 | | | |

**Alkali metals (neutral)**

| Module | Species | I | Module | Species | I |
|---|---|---|---|---|---|
| `Li_I` | ⁷Li | 3/2 | `K_I_41` | ⁴¹K | 3/2 |
| `Li_I_6` | ⁶Li | 1 | `Rb_I_85` | ⁸⁵Rb | 5/2 |
| `Na_I_23` | ²³Na | 3/2 | `Rb_I_87` | ⁸⁷Rb | 3/2 |
| `K_I_39` | ³⁹K | 3/2 | `Cs_I_133` | ¹³³Cs | 7/2 |
| `K_I_40` | ⁴⁰K | 4 | `Fr_I_211` | ²¹¹Fr | 9/2 |

**MOT-trapped neutrals — alkaline-earth-like** (¹S₀ ground state, two-stage broad/narrow MOT)

| Module | Species | I | Notes |
|---|---|---|---|
| `Mg_I_24` | ²⁴Mg | 0 | 285 nm MOT |
| `Ca_I_40` | ⁴⁰Ca | 0 | 423 nm MOT, 657 nm clock line |
| `Sr_I_88` | ⁸⁸Sr | 0 | 461 nm blue MOT → 689 nm red MOT |
| `Sr_I_87` | ⁸⁷Sr | 9/2 | 698 nm optical lattice clock |
| `Yb_I_174` | ¹⁷⁴Yb | 0 | 399 nm blue MOT → 556 nm green MOT |
| `Yb_I_171` | ¹⁷¹Yb | 1/2 | 578 nm lattice clock; nuclear-spin tweezer qubit |
| `Yb_I_173` | ¹⁷³Yb | 5/2 | SU(N) physics |
| `Hg_I_202` | ²⁰²Hg | 0 | 254 nm MOT, cooled directly on the narrow line |
| `Hg_I_199` | ¹⁹⁹Hg | 1/2 | 266 nm clock line |
| `Cd_I_112` | ¹¹²Cd | 0 | 229 nm MOT |

**MOT-trapped neutrals — strongly magnetic** (dipolar quantum gases)

| Module | Species | I | Notes |
|---|---|---|---|
| `Dy_I_164` | ¹⁶⁴Dy | 0 | μ ≈ 10 μ_B, the most magnetic neutral atom; 421/626 nm MOT |
| `Dy_I_161` | ¹⁶¹Dy | 5/2 | fermionic dipolar gas |
| `Er_I_166` | ¹⁶⁶Er | 0 | μ ≈ 7 μ_B; 401/583 nm MOT |
| `Er_I_167` | ¹⁶⁷Er | 7/2 | hyperfine *B* ≫ *A*, so nonlinear Zeeman is enabled |
| `Cr_I_52` | ⁵²Cr | 0 | μ ≈ 6 μ_B, the first dipolar BEC; 426 nm MOT |
| `Tm_I_169` | ¹⁶⁹Tm | 1/2 | 411 nm MOT; 1.14 μm inner-shell clock transition |

**MOT-trapped metastable noble gases**

⚠️ The trapped state is a metastable *excited* state, not the ground state. These modules
build from the true ground state as usual, so reference the metastable level explicitly:

```python
from atomtoolkit import load_species
metastable = load_species('4He I').levels['1s.2s 3S1']   # 19.8 eV above the ground state
```

| Module | Species | I | Notes |
|---|---|---|---|
| `He_I_4` | ⁴He\* | 0 | metastable 1s2s ³S₁ (τ ≈ 7900 s), cooled at 1083 nm |
| `Ne_I_20` | ²⁰Ne\* | 0 | metastable 3s ²[3/2]°₂, cooled at 640 nm; JK-coupled |

**Other**

| Module | Species | I | Notes |
|---|---|---|---|
| `Tm_II_169` | ¹⁶⁹Tm⁺ | 1/2 | lanthanide *ion*, ships with a transition list (see `Tm_I_169` for the MOT-trapped neutral) |
| `Ho_I_165` | ¹⁶⁵Ho | 7/2 | J=15/2 ground state → **128** ground-manifold Zeeman sublevels; JK/JJ-coupled excited configurations; hyperfine *B* larger than *A*. Useful as a stress test. |

Caveats worth knowing before using these for precision work:

- **Short names.** Modules expose their atom as element+mass (`Sr88`, `Dy164`). Where a
  neutral and an ion of the *same isotope* both exist, the neutral takes a trailing `n`:
  `Sr_II_87.Sr87` is the ion, `Sr_I_87.Sr87n` the neutral. Import the module, not the bare
  name, and the two never collide.
- **Forbidden clock transitions.** `¹S₀ → ³P°₀` is J=0 → J=0, which is forbidden for *every*
  multipole and only becomes weakly allowed in odd isotopes through hyperfine mixing. It is
  never populated automatically; `Sr_I_87` and `Yb_I_171` add it explicitly with its measured
  frequency, and its `allowed_types` correctly reads all-False.
- **Nuclear g-factors.** `atom.gI` is signed, in nuclear magneton units. The alkalis use
  measured, diamagnetically shielded atomic *g*<sub>I</sub> values; the ions mostly use the
  bare `μ_I / I` ratio, which runs ~0.5% larger. That is enough to move a magic field by
  several mG — replace it with a shielded value before relying on it.
- **Incomplete hyperfine data.** Ground-state coefficients are in place throughout, but
  several species are missing excited-state *A* and *B*. Each such file carries a
  `Data still needed` block listing exactly what is absent, and values I could not verify
  against a primary source are marked `VERIFY`.
- **Skipped levels.** A few high-lying levels in Hg II and Ho I use nested JJ/JK term symbols
  the parser does not handle; they are skipped with a warning at build time rather than
  aborting the build.

See `species/__init__.py` for the authoritative inventory.

## Citing the data

Energy levels bundled in `atomtoolkit/resources/` and fetched by `IO.load_NIST_data()` come from the
**NIST Atomic Spectra Database**. NIST asks that work using this data cite it:

> Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team. *NIST Atomic Spectra Database*
> (version 5.12). National Institute of Standards and Technology, Gaithersburg, MD.
> https://physics.nist.gov/asd — DOI: [10.18434/T4W30F](https://doi.org/10.18434/T4W30F)

The bundled CSVs were retrieved in July 2026; check the ASD for more recent evaluations
before publishing. NIST data is a work of the US government and is not subject to copyright.

Hyperfine coefficients, nuclear g-factors, and transition rates are **not** from ASD — they
come from the literature and are cited inline in each species file. Cite those sources
directly rather than this package.
