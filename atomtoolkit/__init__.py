import pint
import pint_pandas

# setup the unit registry
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

pint.set_application_registry(ureg)
pint_pandas.PintType.ureg = ureg

# add wavenumbers to the pint spectroscopy context
_c = pint.Context('spectroscopy')
_c.add_transformation('[wavenumber]', '[frequency]',
                      lambda ureg, x: x * ureg.speed_of_light)
_c.add_transformation('[frequency]', '[wavenumber]',
                      lambda ureg, x: x / ureg.speed_of_light)
ureg.enable_contexts('spectroscopy')

c = ureg.speed_of_light
Hz = ureg.Hz
kHz = ureg.Hz * 1e3
MHz = ureg.Hz * 1e6
GHz = ureg.Hz * 1e9
THz = ureg.Hz * 1e12
G = ureg.G
