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

Current state: 744 tests pass, 4 skipped (qutip-dependent). Of these, 47 are marked `slow`.

## Architecture

Atom-Toolkit models atomic physics data using **NetworkX graphs** as the backing store. The `Atom` class owns three layered graph models:
- `_levelsModel` — gross-structure `EnergyLevel` nodes + `Transition` edges
- `_hfModel` — hyperfine `HFLevel` nodes + `HFTransition` edges (auto-generated on access)
- `_zModel` — Zeeman `ZLevel` nodes + `ZTransition` edges (auto-generated on access)

### Key classes (`atomtoolkit/atom.py`)
- **`BaseLevel`** / **`EnergyLevel`** / **`HFLevel`** / **`ZLevel`** — hierarchical energy level classes. Sublevels are generated *lazily* (via `@lazy_compute_sublevels`) the first time they are accessed.
- **`BaseTransition`** / **`Transition`** / **`HFTransition`** / **`ZTransition`** — transitions at each layer. Setting `.freq` or `.A` propagates up/down automatically. Transitions must be explicitly added with `atom.add_transition(t)` — there is no auto-add on construction.
- **`AllowedTypes`** — `NamedTuple(E1, M1, E2)` on each transition; access as `.allowed_types.E1` etc., never by index.
- **Transition lab-quantity helpers** (on `BaseTransition`, so they work identically on gross `Transition`s and on HF/Z subtransitions):
  - `.total_A` — climbs the parent chain to the root `Transition` and returns the *gross* natural linewidth Γ, which is what governs saturation and scattering even for a single sublevel pair.
  - `.I_sat` — saturation intensity `π h ν³ Γ / (3 c² S)`, where `S = rel_strength`. Subtransitions have a *higher* `I_sat` than the cycling transition by `1/S`.
  - `.saturation_parameter(intensity)` → `s = I / I_sat` (already sublevel-corrected).
  - `.rabi_frequency(intensity)` → `Ω/2π = Γ√(s/2)` in Hz.
  - `.scattering_rate(intensity, detuning=None)` → `(Γ/2)·s / (1 + s + (2Δ/Γ)²)` in photons/s.
- **`LevelStructure`** / **`TransitionStructure`** (`atomtoolkit/atom_helpers.py`) — dict-like facades over the graph with rich key access:
  - `atom.levels['4f14.6s 2S1/2']` — gross level by name
  - `atom.levels['4f14.6s 2S1/2']['F=0']['mF=0']` — chained sublevel access
  - `atom.levels['4f14.6s 2S1/2 F=0 mF=0']` — flat string shorthand
  - `atom.transitions[('level_a', 'level_b')]` — order-independent (auto-sorted low→high)
- **Atom convenience properties**: `atom.hflevels`, `atom.zlevels`, `atom.hftransitions`, `atom.ztransitions` return dicts of the populated sublevel graphs (only populated after the lazy accessors have been triggered).
- **Serialisation**: `atom.save_json(path)` / `Atom.load_json(path)` — JSON with schema versioning (`_ATOM_JSON_SCHEMA = 3`) and a migration chain in `_migrate_atom_dict()`. Schema 3 added `gI`; the migration supplies `gI: 0.0` so older files still load. The old pickle-based `save()`/`load()` is deprecated.
- **`B` field removed from `Atom`** — there is no `atom.B`. Zeeman energies are computed on demand: `zlevel.shift_hz_at(B_gauss)` and `zlevel.level_hz_at(B_gauss)`. Pass `B` via an `Environment` object to dynamics functions.
- **`atom.gI`** — signed nuclear g-factor in nuclear-magneton units, defaults to `0.0`. It feeds the nuclear Zeeman term in `HFLevel.compute_gF()` and in the exact Hamiltonian in `zeeman.py`. Set it per species when hyperfine accuracy matters.
- **Unimplemented `Atom` stubs** — `transitions_near()`, `find_clock_transitions()`, `ac_stark_shift()`, `doppler_cooling_limit()` are deliberate placeholders that raise `NotImplementedError`, with TODO comments describing the intended physics. Tests assert they raise. Don't treat them as bugs; implement them properly or leave them.

### Nonlinear Zeeman effect (`atomtoolkit/zeeman.py`)
By default `ZLevel.level_hz_at(B)` uses the linear form `gF·mF·μB·B`. For fields where the Zeeman energy is comparable to the hyperfine splitting, set `energy_level.use_nonlinear_zeeman = True` and the level dispatches to exact diagonalisation of `H = H_HF + H_Z` in the `|F, mF⟩` basis.

- `mF` is conserved, so the matrix is block-diagonal in `mF`; each block goes through `numpy.linalg.eigh`. Eigenvalues are mapped back to `(F, mF)` labels by adiabatic correspondence — sorting by ascending B=0 hyperfine energy.
- `zeeman_sweep(level, B_array)` → `{(F, mF): ndarray}` of absolute Hz. This is the **preferred entry point** for Zeeman diagrams and B-sweeps.
- `diagonalize_hfz(level, B_gauss)` / `diagonalize_hfz_batch(level, B_array)` — lower-level access to eigenvalues *and* eigenvectors. `lru_cache`-backed, so one diagonalisation per manifold per unique B serves all `2F+1` sublevels.
- `breit_rabi(level, F, mF, B_gauss)` — analytic Breit-Rabi energy for J=1/2 levels; used to validate the numerics.
- `EnergyLevel.precompute_zeeman(B_min, B_max, n_points=200)` builds a `CubicSpline` LUT over every `(F, mF)` pair so per-level lookups in a sweep loop are ~free. Queries outside the range fall back to live diagonalisation *with a warning*. `clear_zeeman_cache()` discards it.
- `ZLevel.zeeman_sensitivity_hz_per_gauss(B_gauss=0.0, delta=1e-3)` — exact `gF·mF·μB` in linear mode, central finite difference of the eigenenergy in nonlinear mode. Scan for zero crossings to find magnetically-insensitive clock states.

### Supporting modules
- **`atomtoolkit/term.py`** — `Term` / `MultiTerm`: parse and represent spectroscopic term symbols across LS, JJ, LK, JK couplings.
  - `Term.markup_runs()` returns `(text, role)` runs with roles `normal`/`super`/`sub`; `Term.to_markup(fmt)` renders them as `unicode`, `latex`, `html` or `plain`. Renderer-agnostic on purpose — the Grotrian SVG backend emits the runs as `<tspan>`s, and LaTeX output is independently useful. `MultiTerm` delegates both to its leading term (its `__getattribute__` forwards only `__slots__`, so methods need explicit delegation).
  - `Term.__eq__` is strict same-type: never matches `MultiTerm`. Use `term.equal_to(other, level='J'|'F'|None)` for cross-type or coarser comparisons.
  - `Term.__hash__` is consistent with `__eq__` (includes `conf`, `term`, `quantum_nums`, `F`, `mF`).
- **`atomtoolkit/IO.py`** — data loading/saving.
  - `IO.load_NIST_data('Yb II')` — fetches from the NIST ASD web API. Requests `format=2`, the same Excel-quoted CSV the ASD web form produces, so the API and local-CSV paths feed the parser identical input. Raises a `RuntimeError` with manual-download instructions on 403.
    - It sets `_NIST_USER_AGENT`, an honest tool identifier, because NIST 403s the default `Python-urllib/3.x` string. Any non-default UA works — **don't "fix" a future block by spoofing a browser string**; that makes the traffic unattributable and is not why the filter exists.
    - `energy1.pl` is the web form's backend, not a versioned API. It can break without notice, so it's a convenience for adding a *new* species (fetch once, save the CSV to `atomtoolkit/resources/`, commit it) rather than something to depend on at runtime. Everything in `atomtoolkit/species/` loads from cached CSVs and never touches the network.
  - `IO.load_NIST_data_from_csv(resource_path('YbII.csv'))` — loads from a locally saved NIST ASD CSV (same format; use when the API is unavailable).
  - `atomtoolkit.resource_path('YbII.csv')` — locates a bundled CSV via `importlib.resources`. **Always use this rather than `__file__` arithmetic**; it is what makes the data resolve inside an installed wheel.
  - `IO.generate_full_from_dataframe(df, name, I)` — builds a complete atom from a dataframe.
  - Supports CSV round-trips for HF coefficients (`generate_hf_csv` / `apply_hf_csv`) and transitions (`generate_transition_csv` / `apply_transition_csv`).
- **`atomtoolkit/environment.py`** — `Environment(B=Q_(5.36, 'G'), fields=[...])` bundles magnetic field and `LightField` objects for dynamics. `B` is a pint Quantity (defaults to 0 G); the plain float is exposed as `env.B_gauss`. `Polarization`, `LightField`, `GaussianBeam` handle field geometry.
- **`atomtoolkit/dynamics.py`** — `generate_hamiltonian(atom, states, environment)` and `generate_dynamics_matrix(atom, states, environment)` build qutip operators. Requires `qutip` (optional dependency, not in `install_requires`; its tests skip when absent).
- **`atomtoolkit/__init__.py`** — shared `pint` unit registry (`ureg`, `Q_`) with spectroscopy context. Import `Q_`, `Hz`, `kHz`, `MHz`, `GHz`, `THz`, `G`, `c`, `e`, `hbar`, `a0`, `mu_B`, `mu_N` from here.
- **`atomtoolkit/util.py`** — spectroscopic string helpers (`frac_to_float`, `float_to_frac`, `get_term_coupling`, `get_quantum_nums`).
- **`atomtoolkit/transition_strengths.py`** — Wigner symbols, reduced matrix elements, E1/M1/E2 geometric coupling factors.
- **`atomtoolkit/wigner.py`** — Wigner 3j and 6j symbol computation.
- **`atomtoolkit/render/`** — `spectra.py` / `lineshapes.py` are matplotlib-based. **`grotrian/` is not** — it emits SVG directly, as a `spec → layout → render` pipeline:
  - `spec.py` — `LevelSpec`, `LevelStyle`, `Theme` (`LIGHT`/`DARK`), `ColumnPolicy`. Pure declaration, no geometry.
  - `scale.py` — `LinearScale` (default, literal) and `CompressedScale` (allocates space by level density; keeps crowded regions legible but distances are no longer proportional).
  - `layout.py` — pure functions producing a `Scene` of positioned primitives. `resolve_overlaps()` de-overlaps labels via PAVA isotonic regression (optimal minimum-displacement, unlike a greedy sweep). `Scene.anchors` maps each level to its *drawn* box — transitions must resolve against those, never raw energies.
  - `svg.py` — `Scene → str` via `svgwrite`. Deliberately dumb so golden-file tests mean something.
  - `Grotrian.render()` is **pure and idempotent** (the matplotlib version doubled its state on every call). `Grotrian.auto(atom)` is the zero-config entry point and the honest test of the defaults.
  - Labels never need real font metrics because `Theme.label_gutter` reserves fixed space per column — see `docs/grotrian-svg-plan.md` §7.
  - **Substructure** (`detail='hyperfine'` / `'zeeman'`) is drawn in place on **one global magnification** shared by the whole diagram. Never size it from the smallest gap: ¹³⁷Ba⁺ 5d ²D₅/₂ has F=4/F=3 just 0.496 MHz apart inside a 135 MHz span, so that rule demands a band thousands of px tall. A shared scale also keeps splittings comparable between manifolds, which is what a reader reads off the figure.
  - `hf_scale='auto'` (default) sizes to the largest **excited-state** splitting, skipping the lowest manifold — ground-state hyperfine runs 5–8× larger than anything else (5.5× in ⁴³Ca⁺, 8.4× in ⁸⁷Rb), so sizing to it squashes everything else to a sliver. `'widest'` restores ground-state sizing. Also accepts px/Hz or a `Quantity` ("this splitting fills the band"). The ground state then overflows proportionally, so the plot area is inset by the worst fan overhang to keep it on-canvas.
  - **Per-manifold overrides**: `add_level(..., hf_scale='fit' | px_per_Hz | Quantity, hf_exaggerate=N)` and the `z_` equivalents. `None` inherits the diagram scale, which is the default and what keeps manifolds comparable. Any manifold drawn off the shared scale is **automatically annotated with its true span** so the figure cannot mislead. The canvas grows around exaggerated fans rather than compressing the energy span — squeezing the span would silently distort the gross axis.
  - `z_scale='match'` (default) keeps Zeeman faithful to the hyperfine scale — and therefore sub-pixel at ordinary fields. `z_scale='auto'` exaggerates it to fill `theme.z_band`, and the scale bar then reports the factor. The vertical offset is decomposed into hyperfine and Zeeman parts so exaggerating one never rescales the other.
  - Sublevels are ordered **by energy, not by F** — a negative hyperfine A inverts the ladder. The fan is not symmetric about the parent line either: the hyperfine centre of gravity is degeneracy-weighted.
  - **Lines are physical, labels relax.** Near-degenerate sublevels overlap honestly while their `F` labels de-overlap onto leaders.
  - Zeeman uses the *same* vertical scale as hyperfine (its sublevels carry the hyperfine shift plus a much smaller Zeeman one) and fans horizontally by m_F. `B=` routes through `ZLevel.level_hz_at`, so `use_nonlinear_zeeman` manifolds get exact diagonalisation for free.
  - Manifolds with no hyperfine data — still the majority — get an evenly spaced schematic fan with a **dashed** bracket, so missing data never reads as real structure.
  - **Transitions** (`add_transition((lower, upper), ...)`, order-normalised; also accepts a `Transition` object) resolve endpoints via `Scene.anchors`, never raw energies — an arrow into a magnified fan must land on the line actually drawn. Missing endpoints warn and skip. A fanned manifold's anchor spans the band, so arrows attach to the *edge facing their partner* rather than the centre, which falls between sublevel lines.
  - Arrow colour comes from the wavelength (`color.py`, Bruton's fit), fading to grey outside 380–780 nm; non-visible lines auto-dash. `wavy=True` (spontaneous decay) suppresses auto-dashing. `route='arc'` bulges perpendicular to the chord — offsetting along +x barely bends a diagonal.
  - Anything emitting SVG must keep `render()` pure: `svgwrite` auto-ids break byte-equality, so marker ids are derived deterministically from appearance.
  - `examples/grotrian_gallery.py` renders a visual regression gallery; run it after touching the renderer.

### Species files (`species/`)
Each file builds a concrete `Atom` instance from a cached NIST CSV, then layers hyperfine coefficients, a nuclear g-factor, and measured transition data on top, with sources cited inline. Run a species file as `__main__` to regenerate its `.json` snapshot alongside it. **`species/__init__.py` holds the authoritative inventory** — 49 isotopes covering the common trapped ions (Be⁺, Mg⁺, Ca⁺, Sr⁺, Ba⁺, Ra⁺, Zn⁺, Cd⁺, Hg⁺, Yb⁺), the alkali metals (Li, Na, K, Rb, Cs, Fr), the MOT-trapped neutrals (Mg, Ca, Sr, Yb, Hg, Cd, plus the magnetic Dy, Er, Cr, Tm and metastable He\*/Ne\*), Tm⁺, and Ho I.

Bundled NIST ASD level CSVs live in `atomtoolkit/resources/` (`YbII.csv`, `CaII.csv`, `RbI.csv`, …), reached via `atomtoolkit.resource_path(name)` and loaded with `IO.load_NIST_data_from_csv()`, so builds are reproducible offline. To add a species, fetch its CSV with `IO.load_NIST_data('Xx II')` and save it there.

Conventions that bite when editing these files:
- **`gI` sign and units** — signed, in *nuclear magneton* units. The alkalis use measured diamagnetically-shielded atomic `g_I` values; the bare `μ_I / I` ratio from nuclear moment tables is ~0.5% larger and shifts the ⁸⁷Rb magic field by ~12 mG. Steck-convention values (Bohr magnetons, opposite sign) convert as `gI_ours = -gI_Steck * mu_B/mu_N`.
- **`ALLOWED_TRANSITIONS = (True, False, False)` populates E1 only.** Quadrupole clock transitions (Ca⁺ 729 nm, Sr⁺ 674 nm, Hg⁺ 282 nm) must be added explicitly with `atom.add_transition(...)` — indexing `atom.transitions[...]` for one that was never added returns `None`, not a KeyError.
- **`Transition.transition_allowed` enforces the full triangle condition** `|J₀−J₁| ≤ k ≤ J₀+J₁`. The lower bound matters: it forbids J=0→J=0 for every multipole and J=0→J=1 for E2. So the neutral `¹S₀ → ³P°₀` clock transitions are correctly all-False and are added by hand in `Sr_I_87`/`Yb_I_171` — they are hyperfine-induced, not a gross-structure multipole. The HF/Zeeman layers get this right via Wigner symbols, but they inherit `init` from the parent, so a wrong gross-level verdict propagates down.
- **Neutral/ion short-name collisions.** Modules expose their atom as element+mass; where a neutral and an ion of the same isotope both exist, the neutral takes a trailing `n` (`Sr_II_87.Sr87` vs `Sr_I_87.Sr87n`).
- **`NUM_LEVELS` is not always 300.** `populate_transitions` is O(N²) in Clebsch-Gordan evaluations, so Ca I, Cr I, He I and Ne I cap at 150 — the difference is ~70 s vs ~5 s per build, and the excluded levels are high Rydberg states. Each such file says so in a comment.
- **He\*/Ne\* are trapped in a metastable *excited* state**, not the ground state. Those modules build from the true ground state, so the metastable level must be referenced explicitly.
- **Unparseable levels are skipped, not fatal.** `Atom.from_dataframe` warns and continues when a term symbol defeats the parser (doubly-nested JJ/JK symbols like `[(2D<5/2>,1/2)<3>,1/2]` in Hg II, and some high-lying Ho I levels). Expect ~33 skipped levels in Hg II and ~34 in Ho I.

**`Ho_I_165.py` is a deliberate stress test**, not just data: J=15/2 ground state with I=7/2 gives 128 ground-manifold Zeeman sublevels, excited configurations are JK/JJ-coupled, and `hfB` (−1668 MHz) is larger than `hfA` (+801 MHz), so the quadrupole term is not a perturbation. Use it when changing `term.py`, `compute_gJ`, or the hyperfine code.

**Registry** (`species/__init__.py`): `species.available_species()` lists all 49 display names by reading one `NAME =` line per file — it deliberately imports nothing, since importing a species module builds its atom. `species.load_species('Yb II 171')` resolves forgiving name forms (`'171Yb II'`, `'Yb_II_171'`, `'yb ii 171'`), prefers a `.json` snapshot when it is newer than its `.py`, falls back to building from CSV, and caches the result. Atoms are shared, so `copy=True` is needed before mutating. `species.build_snapshots()` regenerates snapshots.

**Packaging:** `species/` and `resources/` live *inside* the package (`atomtoolkit/species/`, `atomtoolkit/resources/`) and are declared in `setup.py` via `packages` + `package_data`, so both ship in the wheel. Data is reached through `atomtoolkit.resource_path()`; never reintroduce `Path(__file__).parents[1]` there, since it resolves only in a checkout. `atomtoolkit.load_species` / `available_species` / `species_info` / `build_snapshots` are re-exported lazily through a PEP 562 module `__getattr__`, which is required — species modules import `atomtoolkit`, so an eager import would cycle. Background and the rejected alternatives are in `docs/packaging-proposal.md`.

Species integration tests are in `tests/test_species.py`, marked `@pytest.mark.slow`; registry tests are in `tests/test_species_registry.py` (fast, apart from a slow-marked loading class).

### Tests (`tests/`)
`test_atoms.py` (levels, transitions, serialisation, lab-quantity helpers, stub behaviour), `test_zeeman.py` (diagonalisation, Breit-Rabi cross-checks, spline LUT, sensitivity), `test_term.py`, `test_util.py`, `test_transition_strengths.py`, `test_environment.py`, `test_dynamics.py` (qutip-gated), `test_species.py` (slow). The `slow` marker is registered in `tests/conftest.py`, not in a pytest config file.

Note: `tests/test_atoms.py` exercises the deprecated pickle path by writing `dummy.atom` into the **repo root** as a side effect. It's gitignored.

### Repo layout conventions
- `scratch/` — gitignored dumping ground for test outputs, one-off figures, `Untitled*.ipynb`, and external `.dat` files. Nothing here is referenced by the library.
- `analysis/` — gitignored one-off research notebooks and scripts (`hf_fit.py`, `zeeman_demo_173Yb.ipynb`, `critical_field_calc.ipynb`, …). Kept separate from `scratch/` because it's real work, just not library content.
- Tracked notebooks at the repo root (`grotrians.ipynb`, `spectra.ipynb`, `testing.ipynb`, `transforms_tutorial.ipynb`) are demo/documentation material — leave them tracked.
- `species/*.json` snapshots are gitignored (14 MB, fully derived). Regenerate them by running the species `.py` file, never by editing the JSON by hand.

### Unit conventions
All *public* quantities use `pint` with the shared registry, and the spectroscopy context is always active, so `.to('Hz')` and `.to('cm**-1')` conversions work throughout.

Internally, hot paths store bare floats in Hz — `_level_Hz`, `hfA_Hz`/`hfB_Hz`/`hfC_Hz`, and everything in `zeeman.py` — with pint `Quantity` wrappers exposed as properties (`.level`, `.hfA`, …). When adding code to a hot loop, follow that convention: work in float Hz and only wrap at the API boundary. `mu_B` and `mu_N` are defined directly in Hz/G for the same reason.

Magnetic fields are floats in **gauss** wherever a bare number is taken (`shift_hz_at(B_gauss)`, `zeeman_sweep`, `precompute_zeeman`); `Environment` is the exception and takes a pint Quantity.