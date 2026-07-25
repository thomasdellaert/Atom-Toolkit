# Packaging the species library

**Status: resolved. Option A was implemented.** The rest of this document is kept as the
record of why, and of what was rejected.

Written up in response to the long-standing TODO in `species/__init__.py`:

> Look more deeply into namespace packaging and see if it's possible to include this
> package in the namespace of atomtoolkit in a nice way

## What was done

- `species/` → `atomtoolkit/species/`, `resources/` → `atomtoolkit/resources/`.
- Data is located with `atomtoolkit.resource_path()`, built on `importlib.resources`,
  replacing `pathlib.Path(__file__).parents[1] / 'resources'` in all 49 species modules.
- `setup.py` gained `atomtoolkit.species` in `packages` and
  `package_data={'atomtoolkit': ['resources/*.csv']}`; a `MANIFEST.in` includes the CSVs
  and excludes the derived `.json` snapshots.
- `atomtoolkit.load_species` / `available_species` / `species_info` / `build_snapshots`
  are re-exported through a PEP 562 module-level `__getattr__`. This has to be lazy:
  species modules import `atomtoolkit`, so an eager import would be circular.

Verified by building a wheel and installing it into a clean virtualenv:
109 files, 50 species modules, 38 CSVs, no snapshots, 0.6 MB, and
`atomtoolkit.load_species('43Ca II')` works from `site-packages`.

That check immediately paid for itself by catching an unrelated latent bug: the clean
environment resolved `pandas` to 3.0.5, where `DataFrame.applymap` has been removed, so
*every* species failed to load. `IO._parse_NIST_df` now uses `DataFrame.map` when
available. This had nothing to do with the move — it would have broken any user who
installed against current pandas — and it is exactly the class of problem a from-scratch
install test exists to find.

## The actual problem is bigger than the import path

The TODO reads as an ergonomics wish — `atomtoolkit.species` instead of a bare top-level
`species`. But surveying it turned up something more serious:

```python
# setup.py
packages=['atomtoolkit', 'atomtoolkit.render'],
```

Neither `species/` nor `resources/` is listed, and there is no `package_data`,
`include_package_data`, or `MANIFEST.in`. So:

- **`pip install atom-toolkit` ships no species library and no NIST CSVs at all.**
- It works in this repo only because the working directory is on `sys.path`, which makes
  the top-level `species` package importable by accident of layout.
- Even if `species/` were added to `packages`, every species module resolves its data as
  `pathlib.Path(__file__).parents[1] / 'resources/CaII.csv'` — a path that exists in a
  checkout and does not exist under `site-packages`.

So any fix has to solve three things together: the import path, package inclusion, and
data-file location. Fixing only the first would give a tidy-looking import that still
fails on an installed copy.

## The literal suggestion — PEP 420 namespace packages — does not work here

A native namespace package requires that **no** portion of the namespace contain an
`__init__.py`. `atomtoolkit/__init__.py` is 31 lines and does essential work: it builds
the shared `pint` registry, installs the spectroscopy context, and defines `Q_`, `Hz`,
`mu_B` and friends. Effectively everything imports from it.

Because `atomtoolkit` is a *regular* package, a separately distributed
`atomtoolkit.species` cannot be merged into it by PEP 420. The choice is between shipping
species inside the same distribution, or giving it a different top-level name.

(The legacy `pkgutil`/`pkg_resources` style namespace packages can extend a regular
package, but they are deprecated, interact badly with wheels and editable installs, and
would mean adding boilerplate to `atomtoolkit/__init__.py` that fights the registry setup.
Not worth it.)

## Option A — move species and resources inside the package *(recommended)*

```
atomtoolkit/
    __init__.py
    atom.py
    species/
        __init__.py          # the registry, unchanged
        Yb_II_171.py
        ...
    resources/
        YbII.csv
        ...
```

```python
# setup.py
packages=['atomtoolkit', 'atomtoolkit.render', 'atomtoolkit.species'],
package_data={'atomtoolkit': ['resources/*.csv']},
include_package_data=True,
```

Data access moves from filesystem paths to `importlib.resources`, which works inside a
wheel, a zipimport, or a checkout:

```python
from importlib.resources import files
_energy_level_df = IO.load_NIST_data_from_csv(files('atomtoolkit') / 'resources/CaII.csv')
```

**For:** conventional, single distribution, `import atomtoolkit.species` reads naturally,
and `atomtoolkit.load_species()` becomes a one-line re-export. Solves all three problems.

**Against:** moves 49 species files plus 18 CSVs; the `absroot` line in every species file
changes (mechanical — it is one line per file, and the files are generated from a table
anyway). Adds ~3.4 MB to the wheel. Breaks `from species import ...`, though the only
importers in the tree are `tests/test_species.py` and `tests/test_species_registry.py`, so
the blast radius is small.

The 3.4 MB is the one real judgement call. For comparison that is roughly a tenth of a
typical scipy wheel, and the data is the point of the package, so I would just ship it.

## Option B — a separate `atom-toolkit-species` distribution

Species and CSVs become their own package, installed as an extra:

```bash
pip install atom-toolkit[species]
```

**For:** keeps the core library small; data can be versioned and re-released on its own
cadence as NIST evaluations are updated, without cutting a new core release.

**Against:** two distributions to build, version and release, and a version-compatibility
matrix between them. It cannot be `atomtoolkit.species` for the PEP 420 reason above, so
it would import as something like `atomtoolkit_species` — which is precisely the
ergonomics the TODO wanted to avoid. Worth revisiting only if the data grows by an order
of magnitude.

## Option C — leave the layout, add a lazy accessor

```python
# atomtoolkit/__init__.py
def load_species(name, **kwargs):
    try:
        import species
    except ImportError:
        raise ImportError(
            'The species library is not installed. It ships with the source '
            'repository but not with the PyPI wheel; see docs/packaging-proposal.md.'
        ) from None
    return species.load_species(name, **kwargs)
```

**For:** no file churn at all, and `atomtoolkit.load_species('Yb II 171')` works today for
anyone running from a checkout. The lazy import avoids a cycle, since `species` imports
`atomtoolkit`.

**Against:** cosmetic. An installed copy still has no species library, so this papers over
the packaging gap rather than closing it. Also leaves `species` squatting a very generic
top-level import name — plausible collision material on PyPI.

## Outcome

**Option A was chosen and implemented**, on the reasoning that 3.4 MB uncompressed
(0.6 MB in the wheel) is a fair price for the data being the point of the package. If the
data grows by an order of magnitude, Option B is the fallback and this document is the
starting point for it.

`tests/test_species_registry.py::TestPackaging` guards the parts that are cheap to check
without building a wheel: that every `resource_path(...)` a species file references
actually exists, that no species file has reintroduced `__file__` arithmetic, that
`setup.py` still declares both the subpackage and the data glob, and that the lazy
re-export works.

Still worth doing, not yet done:

1. A CI job that builds the wheel, installs it into a clean virtualenv, and runs
   `load_species('Yb II 171')`. The in-repo tests cannot catch a `package_data` typo,
   because in a checkout the files are present either way. Doing this manually is what
   surfaced the pandas 3 breakage above.
2. Pinning a lower bound on `pandas` in `install_requires` (it is currently unpinned), and
   deciding a supported range now that pandas 3 is out.
