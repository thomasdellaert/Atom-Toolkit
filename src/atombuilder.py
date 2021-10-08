from . import *
from main import Term, EnergyLevel, Atom

# def energy_level_from_df(df, i):
#     t = Term(df["Configuration"][i], df["Term"][i], df["J"][i], percentage=df["Leading percentages"])
#     e = EnergyLevel(t, df["Level (cm-1)"][i], lande=df["Lande"][i], hfA=10 * ureg('gigahertz'))
#     return e
#
# def atom_from_df(df, name, I=0.0, num_levels=None, B=Q_(0.0, 'G')):
#     a = Atom(name, I=I, B=B)
#     for i in range(num_levels):
#         try:
#             e = energy_level_from_df(df, i)
#             a.add_level(e)
#         except KeyError:
#             pass
#     return a

"""Placeholder for when I'd like to instantiate atoms in more complicated ways, adding transitions and 
hyperfine coefficients, for instance"""
