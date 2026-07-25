"""
Tests for the species registry (species/__init__.py).

The name-resolution tests are fast: the registry reads one NAME line out of each species
file and imports nothing. Tests that actually build an atom are marked `slow`.
"""

import pathlib

import pytest

import species
from atomtoolkit.atom import Atom


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:

    def test_lists_every_species_file(self):
        """available_species() must cover exactly the species modules on disk."""
        on_disk = {p.stem for p in pathlib.Path(species.__file__).parent.glob('*.py')}
        on_disk.discard('__init__')
        assert len(species.available_species()) == len(on_disk)

    def test_names_are_sorted_and_unique(self):
        names = species.available_species()
        assert names == sorted(names)
        assert len(names) == len(set(names))

    def test_info_maps_display_name_to_module(self):
        info = species.species_info()
        assert info['171Yb II'] == 'Yb_II_171'
        assert info['87Sr I'] == 'Sr_I_87'

    def test_discovery_imports_nothing(self):
        """Listing must not import species modules -- that would build every atom."""
        import sys
        for name in list(sys.modules):
            if name.startswith('species.'):
                del sys.modules[name]
        species.available_species()
        assert not [n for n in sys.modules if n.startswith('species.')]


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

class TestNameResolution:

    @pytest.mark.parametrize('name', [
        'Yb II 171', '171Yb II', 'Yb_II_171', 'yb ii 171', 'Yb171 II', 'YB_II_171',
    ])
    def test_equivalent_spellings(self, name):
        assert species._resolve(name) == 'Yb_II_171'

    def test_neutral_and_ion_of_same_isotope_are_distinct(self):
        """The Sr_I_87 / Sr_II_87 pair is exactly the case the 'n' suffix exists for."""
        assert species._resolve('87Sr I') == 'Sr_I_87'
        assert species._resolve('87Sr II') == 'Sr_II_87'

    def test_module_name_without_mass_number(self):
        """Li_I is named '7Li I' but its module carries no mass number."""
        assert species._resolve('Li_I') == 'Li_I'
        assert species._resolve('7Li I') == 'Li_I'

    def test_every_name_round_trips(self):
        """No alias may resolve to the wrong module -- checks all 49 at once."""
        for display, stem in species.species_info().items():
            assert species._resolve(display) == stem
            assert species._resolve(stem) == stem

    def test_unknown_name_raises_with_suggestions(self):
        with pytest.raises(KeyError) as excinfo:
            species.load_species('Yb II 172')
        message = str(excinfo.value)
        assert 'Did you mean' in message
        assert '171Yb II' in message

    def test_unrecognisable_name_still_raises_cleanly(self):
        with pytest.raises(KeyError) as excinfo:
            species.load_species('not an element')
        assert 'available_species' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLoading:
    """Ra II 225 is the smallest bundled species (~43 levels), so it builds in ~0.1 s."""

    def test_load_returns_an_atom(self):
        atom = species.load_species('225Ra II', use_snapshot=False)
        assert isinstance(atom, Atom)
        assert atom.name == '225Ra II'
        assert atom.levels['7s 2S1/2'] is not None

    def test_repeat_loads_are_shared(self):
        a = species.load_species('225Ra II')
        b = species.load_species('Ra_II_225')
        assert a is b

    def test_copy_is_independent(self):
        original = species.load_species('225Ra II')
        duplicate = species.load_species('225Ra II', copy=True)
        assert duplicate is not original
        assert len(duplicate.levels) == len(original.levels)

    def test_copy_mutation_does_not_leak(self):
        from atomtoolkit import Q_
        original = species.load_species('225Ra II')
        before = original.levels['7s 2S1/2'].hfA
        duplicate = species.load_species('225Ra II', copy=True)
        duplicate.levels['7s 2S1/2'].hfA = Q_(12345.0, 'MHz')
        assert original.levels['7s 2S1/2'].hfA == before
