import pint
import pint_pandas

# set up the unit registry
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

pint.set_application_registry(ureg)
pint_pandas.PintType.ureg = ureg

# add wavenumbers to the preexisting pint spectroscopy context
_c = pint.Context('spectroscopy')
_c.add_transformation('[wavenumber]', '[frequency]',
                      lambda u, x: x * u.speed_of_light)
_c.add_transformation('[frequency]', '[wavenumber]',
                      lambda u, x: x / u.speed_of_light)
ureg.enable_contexts('spectroscopy')

c = ureg.speed_of_light
e = ureg.elementary_charge
hb = hbar = ureg.planck_constant / (2*ureg.pi)
Hz = ureg.Hz
kHz = Hz * 1e3
MHz = Hz * 1e6
GHz = Hz * 1e9
THz = Hz * 1e12
G = ureg.G
a0 = 5.29177210544e-11 * ureg.meter

mu_B = 1.39962449361e6 * Hz / G
mu_N = mu_B / 1836.15267343  # nuclear magneton in Hz/G


def resource_path(*parts):
    """Locate a bundled data file under ``atomtoolkit/resources/``.

    Uses :mod:`importlib.resources` rather than ``__file__`` arithmetic, so it resolves
    correctly whether the package is a source checkout, an editable install, or an
    installed wheel::

        IO.load_NIST_data_from_csv(resource_path('YbII.csv'))

    :param parts: path components below ``resources/``
    :return: a path to the resource
    """
    from importlib.resources import files
    path = files(__name__) / 'resources'
    for part in parts:
        path = path / part
    return path


# The species library imports this module, so it cannot be imported eagerly here.
# PEP 562 module-level __getattr__ defers it to first use, which keeps
# `atomtoolkit.load_species(...)` working without an import cycle.
_SPECIES_EXPORTS = ('available_species', 'load_species', 'species_info', 'build_snapshots')


def __getattr__(name):
    if name in _SPECIES_EXPORTS:
        from . import species
        return getattr(species, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(list(globals()) + list(_SPECIES_EXPORTS))
