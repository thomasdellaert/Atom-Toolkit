pro# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install
```bash
pip install -e .
# or install dependencies directly:
pip install pint pandas pint-pandas numpy networkx sympy matplotlib scipy tqdm
```

### Run tests
```bash
pytest tests/
pytest tests/test_atoms.py          # single file
pytest tests/test_atoms.py::TestAtom::test_B  # single test
```

## Architecture

Atom-Toolkit models atomic physics data using **NetworkX graphs** as the backing store. The `Atom` class owns three layered graph models:
- `_levelsModel` — gross-structure `EnergyLevel` nodes + `Transition` edges
- `_hfModel` — hyperfine `HFLevel` nodes + `HFTransition` edges (auto-generated on access)
- `_zModel` — Zeeman `ZeemanLevel` nodes + `ZeemanTransition` edges (auto-generated on access)

### Key classes (`atomtoolkit/atom.py`)
- **`BaseLevel`** / **`EnergyLevel`** / **`HFLevel`** / **`ZeemanLevel`** — hierarchical energy level classes. Sublevels are generated *lazily* (via `@lazy_compute_sublevels`) the first time they are accessed.
- **`BaseTransition`** / **`Transition`** / **`HFTransition`** / **`ZeemanTransition`** — transitions at each layer. Setting `.freq` or `.A` on a sublevel propagates up/down to the parent and child transitions automatically.
- **`LevelStructure`** / **`TransitionStructure`** (`atomtoolkit/atom_helpers.py`) — dict-like facades over the graph that support rich key access:
  - `atom.levels['4f14.6s 2S1/2']` — gross level
  - `atom.levels['4f14.6s 2S1/2']['F=0']['mF=0']` — nested Zeeman sublevel
  - `atom.levels['4f14.6s 2S1/2 F=0 mF=0']` — flat string shorthand
  - `atom.transitions[('level_a', 'level_b')]` — order-independent (auto-sorted low→high)

### Supporting modules
- **`atomtoolkit/term.py`** — `Term` / `MultiTerm`: parse and represent spectroscopic term symbols across LS, JJ, LK, JK couplings. Quantum numbers are extracted automatically from the NIST-format configuration + term strings.
- **`atomtoolkit/IO.py`** — data loading/saving. `IO.load_NIST_data('Yb II')` fetches directly from the NIST ASD web API. `IO.generate_full_from_dataframe(df, name, I)` builds a complete atom. Supports CSV round-trips for hyperfine coefficients.
- **`atomtoolkit/__init__.py`** — sets up a shared `pint` unit registry (`ureg`, `Q_`) with a spectroscopy context that enables wavenumber ↔ frequency conversions. Import `Q_`, `Hz`, `kHz`, `MHz`, `GHz`, `THz`, `mu_B` from here.
- **`atomtoolkit/util.py`** — spectroscopic string helpers (`frac_to_float`, `float_to_frac`, `get_term_coupling`, `get_quantum_nums`).
- **`atomtoolkit/transition_strengths.py`** — Wigner symbols, reduced matrix elements, E1/M1/E2 geometric coupling factors.
- **`atomtoolkit/wigner.py`** — Wigner 3j and 6j symbol computation.
- **`atomtoolkit/environment.py`** — `Polarization`, `LightField`, `GaussianBeam` for computing coupling to external fields.
- **`atomtoolkit/dynamics.py`** — (in progress) dipole matrix elements and dynamics matrix generation.
- **`atomtoolkit/render/`** — plotting: `grotrian.py` (Grotrian diagrams via matplotlib), `spectra.py` (spectral lineshapes), `lineshapes.py`, `arrows.py`.

### Species files (`species/`)
Each file (e.g. `Yb_II_171.py`, `Ba_II_133.py`) builds a concrete `Atom` instance from NIST data, applies hyperfine coefficients from a CSV, and adds/corrects transitions. Serialised `.atom` files (pickle) are saved/loaded with `Atom.save()` / `Atom.load()`.

### Unit conventions
All physical quantities use `pint` with the shared registry from `atomtoolkit`. Levels are stored as both `cm⁻¹` and `Hz`. Hyperfine coefficients (`hfA`, `hfB`, `hfC`) are in `Hz`-compatible units. The spectroscopy context is always enabled, allowing `.to('Hz')` and `.to('cm**-1')` conversions.