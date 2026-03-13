# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install
```bash
pip install -e .
```

### Run tests
```bash
pytest tests/                                          # all fast tests
pytest tests/ -m slow                                  # species integration tests (NIST network/CSV)
pytest tests/ -m "not slow"                            # skip species tests
pytest tests/test_atoms.py::TestAtom::test_init        # single test
```

### Build and publish
```bash
python3 -m build
python3 -m twine upload dist/atom_toolkit-0.2.0*
```

## Architecture

Atom-Toolkit models atomic physics data using **NetworkX graphs** as the backing store. The `Atom` class owns three layered graph models:
- `_levelsModel` — gross-structure `EnergyLevel` nodes + `Transition` edges
- `_hfModel` — hyperfine `HFLevel` nodes + `HFTransition` edges (auto-generated on access)
- `_zModel` — Zeeman `ZLevel` nodes + `ZTransition` edges (auto-generated on access)

### Key classes (`atomtoolkit/atom.py`)
- **`BaseLevel`** / **`EnergyLevel`** / **`HFLevel`** / **`ZLevel`** — hierarchical energy level classes. Sublevels are generated *lazily* (via `@lazy_compute_sublevels`) the first time they are accessed.
- **`BaseTransition`** / **`Transition`** / **`HFTransition`** / **`ZTransition`** — transitions at each layer. Setting `.freq` or `.A` propagates up/down automatically. Transitions must be explicitly added with `atom.add_transition(t)` — there is no auto-add on construction.
- **`AllowedTypes`** — `NamedTuple(E1, M1, E2)` on each transition; access as `.allowed_types.E1` etc., never by index.
- **`LevelStructure`** / **`TransitionStructure`** (`atomtoolkit/atom_helpers.py`) — dict-like facades over the graph with rich key access:
  - `atom.levels['4f14.6s 2S1/2']` — gross level by name
  - `atom.levels['4f14.6s 2S1/2']['F=0']['mF=0']` — chained sublevel access
  - `atom.levels['4f14.6s 2S1/2 F=0 mF=0']` — flat string shorthand
  - `atom.transitions[('level_a', 'level_b')]` — order-independent (auto-sorted low→high)
- **Atom convenience properties**: `atom.hflevels`, `atom.zlevels`, `atom.hftransitions`, `atom.ztransitions` return dicts of the populated sublevel graphs (only populated after the lazy accessors have been triggered).
- **Serialisation**: `atom.save_json(path)` / `Atom.load_json(path)` — JSON with schema versioning (`_ATOM_JSON_SCHEMA = 2`) and a migration chain in `_migrate_atom_dict()`. The old pickle-based `save()`/`load()` is deprecated.
- **`B` field removed from `Atom`** — there is no `atom.B`. Zeeman energies are computed on demand: `zlevel.shift_hz_at(B_gauss)` and `zlevel.level_hz_at(B_gauss)`. Pass `B` via an `Environment` object to dynamics functions.

### Supporting modules
- **`atomtoolkit/term.py`** — `Term` / `MultiTerm`: parse and represent spectroscopic term symbols across LS, JJ, LK, JK couplings.
  - `Term.__eq__` is strict same-type: never matches `MultiTerm`. Use `term.equal_to(other, level='J'|'F'|None)` for cross-type or coarser comparisons.
  - `Term.__hash__` is consistent with `__eq__` (includes `conf`, `term`, `quantum_nums`, `F`, `mF`).
- **`atomtoolkit/IO.py`** — data loading/saving.
  - `IO.load_NIST_data('Yb II')` — fetches from the NIST ASD web API. Raises a `RuntimeError` with download instructions on 403.
  - `IO.load_NIST_data_from_csv('resources/YbII.csv')` — loads from a locally saved NIST ASD CSV (same format; use when the API is unavailable).
  - `IO.generate_full_from_dataframe(df, name, I)` — builds a complete atom from a dataframe.
  - Supports CSV round-trips for HF coefficients (`generate_hf_csv` / `apply_hf_csv`) and transitions (`generate_transition_csv` / `apply_transition_csv`).
- **`atomtoolkit/environment.py`** — `Environment(B_gauss, fields)` bundles magnetic field and `LightField` objects for dynamics. `Polarization`, `LightField`, `GaussianBeam` handle field geometry.
- **`atomtoolkit/dynamics.py`** — `generate_hamiltonian(atom, states, environment)` and `generate_dynamics_matrix(atom, states, environment)` build qutip operators. Requires `qutip` (optional dependency).
- **`atomtoolkit/__init__.py`** — shared `pint` unit registry (`ureg`, `Q_`) with spectroscopy context. Import `Q_`, `Hz`, `kHz`, `MHz`, `GHz`, `THz`, `mu_B` from here.
- **`atomtoolkit/util.py`** — spectroscopic string helpers (`frac_to_float`, `float_to_frac`, `get_term_coupling`, `get_quantum_nums`).
- **`atomtoolkit/transition_strengths.py`** — Wigner symbols, reduced matrix elements, E1/M1/E2 geometric coupling factors.
- **`atomtoolkit/wigner.py`** — Wigner 3j and 6j symbol computation.
- **`atomtoolkit/render/`** — plotting: `grotrian.py` (Grotrian diagrams), `spectra.py` (spectral lineshapes), `lineshapes.py`, `arrows.py`.

### Species files (`species/`)
Each file (e.g. `Yb_II_171.py`, `Ba_II_133.py`) builds a concrete `Atom` instance, loads HF coefficients from a CSV in `resources/`, and adds/corrects transitions. Bundled NIST CSVs in `resources/` (`YbII.csv`, `BaII.csv`, `LiI.csv`) are used directly via `IO.load_NIST_data_from_csv()`. Run a species file as `__main__` to regenerate its `.json` snapshot.

Species integration tests are in `tests/test_species.py`, marked `@pytest.mark.slow`.

### Unit conventions
All physical quantities use `pint` with the shared registry. Levels are stored as both `cm⁻¹` and `Hz`. HF coefficients (`hfA`, `hfB`, `hfC`) are in `Hz`-compatible units. The spectroscopy context is always active, so `.to('Hz')` and `.to('cm**-1')` conversions work throughout.