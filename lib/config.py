import pandas as pd
import pint
import pint_pandas

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

pint.set_application_registry(ureg)
pint_pandas.PintType.ureg = ureg
pd.options.display.max_rows = 999

_c = pint.Context('spectroscopy')
_c.add_transformation('[wavenumber]', '[frequency]',
                     lambda ureg, x: x * ureg.speed_of_light)
_c.add_transformation('[frequency]', '[wavenumber]',
                     lambda ureg, x: x / ureg.speed_of_light)
ureg.enable_contexts('spectroscopy')
