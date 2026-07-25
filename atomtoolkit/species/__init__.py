"""
Species library: one module per isotope, each building a concrete :class:`Atom`.

Run any of these as ``__main__`` to regenerate its ``.json`` snapshot alongside it::

    python -m atomtoolkit.species.Ca_II_43

Every module loads energy levels from a cached NIST ASD CSV in ``atomtoolkit/resources/``
(so the
build is reproducible offline), then layers hyperfine coefficients, nuclear g-factors
and measured transition data on top, with sources cited inline.

Available species
-----------------
Trapped ions, alkaline-earth-like (valence s electron over a closed shell):
    Be_II_9      I=3/2   Mg_II_24    I=0     Mg_II_25    I=5/2
    Ca_II_40     I=0     Ca_II_43    I=7/2
    Sr_II_87     I=9/2   Sr_II_88    I=0
    Ba_II_133    I=1/2   Ba_II_137   I=3/2   Ba_II_138   I=0
    Ra_II_225    I=1/2
    Yb_II_171    I=1/2   Yb_II_173   I=5/2   Yb_II_174   I=0

Trapped ions, group 12 (d-hole states above the ground state rather than below the P levels):
    Zn_II_64     I=0     Zn_II_67    I=5/2
    Cd_II_111    I=1/2
    Hg_II_199    I=1/2   Hg_II_202   I=0

Alkali metals (neutral):
    Li_I (7Li)   I=3/2   Li_I_6      I=1
    Na_I_23      I=3/2
    K_I_39       I=3/2   K_I_40      I=4     K_I_41      I=3/2
    Rb_I_85      I=5/2   Rb_I_87     I=3/2
    Cs_I_133     I=7/2
    Fr_I_211     I=9/2

MOT-trapped neutrals, alkaline-earth-like (1S0 ground state, two-stage broad/narrow MOT):
    Mg_I_24      I=0     Ca_I_40     I=0
    Sr_I_88      I=0     Sr_I_87     I=9/2   -- 698 nm lattice clock
    Yb_I_174     I=0     Yb_I_171    I=1/2   -- 578 nm lattice clock, tweezer qubit
                         Yb_I_173    I=5/2
    Hg_I_202     I=0     Hg_I_199    I=1/2
    Cd_I_112     I=0

MOT-trapped neutrals, strongly magnetic (dipolar quantum gases):
    Dy_I_164     I=0     Dy_I_161    I=5/2   -- mu = 10 mu_B, the most magnetic neutral atom
    Er_I_166     I=0     Er_I_167    I=7/2   -- mu = 7 mu_B
    Cr_I_52      I=0     -- mu = 6 mu_B, the first dipolar BEC
    Tm_I_169     I=1/2   -- 1.14 um inner-shell clock transition

MOT-trapped metastable noble gases (NOTE: the trapped state is a metastable *excited*
state, not the ground state -- reference it explicitly, e.g. He4.levels['1s.2s 3S1']):
    He_I_4       I=0     -- metastable 1s.2s 3S1, cooled at 1083 nm
    Ne_I_20      I=0     -- metastable 3s 2[3/2]*2, cooled at 640 nm

Other:
    Tm_II_169    I=1/2   -- lanthanide ion with a transition list (the *ion*; see Tm_I_169
                            for the MOT-trapped neutral)
    Ho_I_165     I=7/2   -- deliberately difficult: J=15/2 ground state, 128 ground-manifold
                            Zeeman sublevels, JK/JJ-coupled excited configurations, and a
                            hyperfine B larger than its A. Use it as a stress test.

Conventions and caveats
-----------------------
* **Short names.** Each module exposes its Atom as element+mass (``Sr88``, ``Dy164``). Where a
  neutral and an ion of the *same isotope* both exist, the neutral takes a trailing ``n`` to
  keep the two apart: ``Sr_II_87.Sr87`` is the ion, ``Sr_I_87.Sr87n`` the neutral. Import the
  module rather than the bare name and the distinction never bites you.
* **Forbidden clock transitions.** ``1S0 -> 3P*0`` is J=0 -> J=0, forbidden for every
  multipole, and only weakly allowed in odd isotopes via hyperfine mixing. It is therefore
  never auto-populated and is added by hand in Sr_I_87 and Yb_I_171, where its
  ``allowed_types`` correctly reads all-False. The same applies to the ions' E2 clock lines,
  which are forbidden only at E1 and so also need adding explicitly.
* ``atom.gI`` is the *signed nuclear g-factor in nuclear magneton units*. The alkalis use
  measured, diamagnetically shielded atomic g_I values (converted from the Bohr-magneton
  convention used by Steck's alkali data documents). The ions mostly use the bare
  ``mu_I / I`` ratio, which is typically ~0.5% larger; that is enough to move a magic
  field by a few mG, so replace it with a shielded value before doing precision work.
* Levels whose term symbols the parser cannot handle (deeply nested JJ/JK couplings in
  Hg II and Ho I) are skipped at build time with a warning rather than aborting the build.
* ``ALLOWED_TRANSITIONS = (True, False, False)`` populates E1 transitions only, so
  quadrupole clock transitions (Ca+ 729 nm, Sr+ 674 nm, Hg+ 282 nm) are added explicitly.
* Several files carry a "Data still needed" block listing hyperfine coefficients that are
  not yet filled in.

Registry
--------
You do not need to know module names or file paths::

    from atomtoolkit import species

    species.available_species()            # -> ['133Ba II', '137Ba II', '138Ba II', ...]
    yb = species.load_species('Yb II 171') # -> Atom

Name matching is forgiving: ``'Yb II 171'``, ``'171Yb II'``, ``'Yb_II_171'`` and
``'yb ii 171'`` all resolve to the same species, and an unknown name raises a
:class:`KeyError` suggesting the closest matches.

:func:`load_species` prefers a prebuilt ``.json`` snapshot when one exists and is newer
than its ``.py``, which turns a ~40 s Rb build into a ~1 s load. Otherwise it imports the
module and builds from the cached NIST CSV. Snapshots are gitignored, so a fresh clone
simply builds; regenerate them with :func:`build_snapshots` or by running a species file
as ``__main__``. The same four functions are re-exported from ``atomtoolkit`` itself,
so ``atomtoolkit.load_species('Yb II 171')`` works too.

.. warning::
    Loaded atoms are cached and shared, so mutating one is visible to every later caller
    in the same process. Pass ``copy=True`` if you intend to modify it.
"""

from __future__ import annotations

import difflib
import importlib
import pathlib
import re
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from atomtoolkit.atom import Atom

__all__ = ['available_species', 'load_species', 'species_info', 'build_snapshots']

_SPECIES_DIR = pathlib.Path(__file__).parent
_NAME_RE = re.compile(r'''^NAME\s*=\s*['"](?P<name>[^'"]+)['"]''', re.MULTILINE)
_ROMAN = ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII')


def _norm(text: str) -> str:
    """Collapse a species name to a comparison key: lowercase alphanumerics only."""
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _split_species(text: str) -> Optional[Tuple[str, str, str]]:
    """Split a species name into (element, ionisation stage, mass number).

    Handles both orderings, e.g. '171Yb II' and 'Yb_II_171'. Returns None if the
    string doesn't look like a species name at all.
    """
    tokens = [t for t in re.split(r'[^A-Za-z0-9]+', text) if t]
    element = stage = mass = ''
    for token in tokens:
        upper = token.upper()
        # A roman numeral is only a stage if we've already seen the element; that keeps
        # a hypothetical iodine ('I') or vanadium ('V') entry from being misread.
        if upper in _ROMAN and element and not stage:
            stage = upper
        elif token.isdigit():
            mass = token
        else:
            # '171Yb' arrives as one token; peel the leading mass number off
            match = re.fullmatch(r'(\d*)([A-Za-z]{1,2})(\d*)', token)
            if not match:
                return None
            lead, symbol, trail = match.groups()
            if symbol.upper() in _ROMAN and element and not stage:
                stage = symbol.upper()
                continue
            element = symbol.capitalize()
            mass = lead or trail or mass
    if not element:
        return None
    return element, stage or 'I', mass


def _aliases(stem: str, display: str) -> List[str]:
    """Every spelling that should resolve to this species."""
    keys = {_norm(stem), _norm(display)}
    parts = _split_species(display) or _split_species(stem)
    if parts:
        element, stage, mass = parts
        keys.add(_norm(f'{mass}{element}{stage}'))    # 171Yb II
        keys.add(_norm(f'{element}{stage}{mass}'))    # Yb II 171
        keys.add(_norm(f'{element}{mass}{stage}'))    # Yb171 II
    return [k for k in keys if k]


_registry_cache: Optional[Dict[str, str]] = None
_display_cache: Optional[Dict[str, str]] = None
# Loaded atoms, keyed by module stem. Both the snapshot path and the module-import path
# populate this, so load_species() has the same sharing semantics either way -- otherwise
# whether a mutation leaked to the next caller would depend on whether a .json happened
# to be lying around.
_atom_cache: Dict[str, 'Atom'] = {}


def _registry() -> Tuple[Dict[str, str], Dict[str, str]]:
    """(alias -> module stem, module stem -> display name), built by reading NAME lines.

    Deliberately does *not* import anything: importing a species module builds its atom,
    which takes seconds to tens of seconds.
    """
    global _registry_cache, _display_cache
    if _registry_cache is None:
        aliases: Dict[str, str] = {}
        displays: Dict[str, str] = {}
        for path in sorted(_SPECIES_DIR.glob('*.py')):
            if path.stem == '__init__':
                continue
            match = _NAME_RE.search(path.read_text(encoding='utf-8'))
            display = match.group('name') if match else path.stem
            displays[path.stem] = display
            for key in _aliases(path.stem, display):
                aliases.setdefault(key, path.stem)
        _registry_cache, _display_cache = aliases, displays
    return _registry_cache, _display_cache


def available_species() -> List[str]:
    """Display names of every bundled species, e.g. ``['133Ba II', '137Ba II', ...]``.

    Cheap: reads one line out of each species file and imports nothing.
    """
    _, displays = _registry()
    return sorted(displays.values())


def species_info() -> Dict[str, str]:
    """Map display name -> module name, for callers that want the underlying module."""
    _, displays = _registry()
    return {display: stem for stem, display in sorted(displays.items())}


def _resolve(name: str) -> str:
    """Resolve a user-supplied name to a module stem, or raise a helpful KeyError."""
    aliases, displays = _registry()
    stem = aliases.get(_norm(name))
    if stem is not None:
        return stem
    close = difflib.get_close_matches(_norm(name), list(aliases), n=4, cutoff=0.6)
    suggestions = sorted({displays[aliases[c]] for c in close})
    hint = f' Did you mean: {", ".join(suggestions)}?' if suggestions else ''
    raise KeyError(
        f'Unknown species {name!r}.{hint} '
        f'Call species.available_species() for the full list of {len(displays)}.'
    )


def _atom_from_module(module) -> 'Atom':
    from atomtoolkit.atom import Atom
    atoms = [v for v in vars(module).values() if isinstance(v, Atom)]
    if not atoms:
        raise RuntimeError(f'{module.__name__} defines no Atom object.')
    return atoms[0]


def load_species(name: str, use_snapshot: bool = True, copy: bool = False) -> 'Atom':
    """Load a bundled species by name.

    :param name: any reasonable spelling -- ``'Yb II 171'``, ``'171Yb II'``, ``'Yb_II_171'``
    :param use_snapshot: prefer a prebuilt ``.json`` snapshot when it exists and is newer
        than its ``.py``. Much faster (a ~40 s Rb build becomes a ~1 s load). Set False to
        always rebuild from the NIST CSV.
    :param copy: return an independent copy rather than the shared cached atom. Use this
        if you are going to mutate it.
    :return: the :class:`~atomtoolkit.atom.Atom`
    :raises KeyError: if the name doesn't match a bundled species (with suggestions)

    ::

        yb = species.load_species('Yb II 171')
        sr = species.load_species('87Sr I')
    """
    from atomtoolkit.atom import Atom

    stem = _resolve(name)

    if stem not in _atom_cache:
        snapshot = _SPECIES_DIR / f'{stem}.json'
        source = _SPECIES_DIR / f'{stem}.py'
        atom = None
        if use_snapshot and snapshot.exists():
            if snapshot.stat().st_mtime >= source.stat().st_mtime:
                atom = Atom.load_json(str(snapshot))
            else:
                warnings.warn(
                    f'{snapshot.name} is older than {source.name}; rebuilding from source. '
                    f'Run `python -m atomtoolkit.species.{stem}` to refresh the snapshot.',
                    stacklevel=2,
                )
        if atom is None:
            atom = _atom_from_module(importlib.import_module(f'.{stem}', __name__))
        _atom_cache[stem] = atom

    atom = _atom_cache[stem]
    return Atom.from_dict(atom.to_dict()) if copy else atom


def build_snapshots(names: Optional[List[str]] = None, force: bool = False) -> List[str]:
    """Regenerate the ``.json`` snapshots that :func:`load_species` prefers.

    :param names: species to build; defaults to all of them. Building everything takes
        several minutes, most of it in the alkalis.
    :param force: rebuild even when the snapshot is already newer than its source
    :return: the display names actually rebuilt
    """
    _, displays = _registry()
    stems = [_resolve(n) for n in names] if names else sorted(displays)
    built = []
    for stem in stems:
        snapshot = _SPECIES_DIR / f'{stem}.json'
        source = _SPECIES_DIR / f'{stem}.py'
        if not force and snapshot.exists() and snapshot.stat().st_mtime >= source.stat().st_mtime:
            continue
        atom = _atom_from_module(importlib.import_module(f'.{stem}', __name__))
        atom.save_json(str(snapshot))
        built.append(displays[stem])
    return built


# The namespace/packaging TODO that used to live here is done: this package moved from a
# top-level `species` to `atomtoolkit.species`, the CSVs moved to atomtoolkit/resources/,
# setup.py declares both, and data is reached through atomtoolkit.resource_path() rather
# than __file__ arithmetic so it resolves inside an installed wheel. PEP 420 namespace
# packages were never an option here -- atomtoolkit is a regular package with an essential
# __init__.py, and a regular package cannot be extended by a separate distribution.
# See docs/packaging-proposal.md for the reasoning and the alternatives considered.

# TODO: Maybe if you try to import something that doesn't exist it could just throw a warning
#  and try to do a quick NIST import? load_species() currently raises KeyError with close
#  matches; falling back to a live NIST fetch would make a typo silently slow instead of
#  loud, so it probably wants to be opt-in (load_species(..., fetch_missing=True)).
