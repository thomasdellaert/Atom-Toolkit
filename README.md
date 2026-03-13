# Atom-Toolkit

A Python package for manipulating atomic physics data in a notebook environment. It models atomic structure as a hierarchy of [NetworkX](https://networkx.org/) graphs — gross-structure energy levels at the top, with hyperfine and Zeeman sublevels generated automatically on demand — and provides a convenient interface for working with spectroscopic data, transitions, and selection rules.

## Features

- **Load data directly from NIST ASD** — fetch energy levels and transition data for any ion/neutral with a single call
- **Three-layer level hierarchy** — `EnergyLevel` → `HFLevel` → `ZLevel` (Zeeman), each layer generated lazily the first time it is accessed
- **Rich key access** for levels and transitions:
  ```python
  atom.levels['4f14.6s 2S1/2']              # gross level
  atom.levels['4f14.6s 2S1/2']['F=0']       # hyperfine sublevel
  atom.levels['4f14.6s 2S1/2 F=0 mF=0']    # flat shorthand
  atom.transitions[level_a, level_b]        # order-independent
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

The bundled species in `resources/` were downloaded this way and are used automatically by the `species/` files.

## Quick start

```python
from atomtoolkit.atom import Atom
from atomtoolkit import IO, Q_

# Load level data — from the API or a local CSV
df = IO.load_NIST_data('Yb II')               # live API
# df = IO.load_NIST_data_from_csv('resources/YbII.csv')  # local fallback
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

The `species/` directory contains ready-to-use atoms (171Yb II, 173Yb II, 174Yb II, 133Ba II, 6Li I) with hyperfine coefficients and corrected transitions applied. Load them with:

```python
from species import Yb_II_171
a = Yb_II_171.atom        # pre-built Atom object
```