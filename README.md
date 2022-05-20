# Atom-Toolkit
A python package that seeks to provide a convenient way to manipulate atomic physics data in a notebook environment, based on Networkx graph structures. The package allows you to:

 - Create an Atom object that contains information about spectroscopic energy levels
   - Hyperfine and Zeeman structure is automatically generated as needed
   - Support for LS, JJ, LK, and J1K couplings, with quantum numbers automatically extracted from the spectroscopic notation
 - Intuitively access and iterate through energy levels and sublevels
   - ```python 
     # Examples with hypothetical atoms:
     Be.levels['1s2.2s2 1S0']
     BaII_138.levels['5d 2D5/2']['mJ=1/2']
     CaII_43.levels['3p6.3d 2D3/2']['F=1']['mF=0']
     NdII_145.levels['5d9.6s2(2D<5/2>).6p 2[3/2]*2']['F=3/2']
     ```
 - Create transitions between energy levels
     - Transitions between any sublevels are also computed, with appropriate selection rules and angular momentum considerations
     - When setting the frequency of a transition, associated parent- and sub-transitions are also updated, as are the energies of the associated levels
 - Compute branching ratios and lifetimes
 - Make diagrams of atomic structure (in development)
 - Plot transition spectra, with different lineshapes and widths

### Example usage:
```python
from atomtoolkit.atom import Atom
from atomtoolkit import IO, Q_

df = IO.load_NIST_data('Yb II')
a = Atom.from_dataframe(df, name='173Yb II', I=2.5)

d32 = a.levels['4f14.5d 2D3/2']
d32.hfA = Q_(-0.11, 'GHz')
d32.hfB = Q_(0.95, 'GHz')

b12 = a.levels['4f13.(2F*<7/2>).5d.6s.(3D) 3[3/2]*1/2']
b12.hfA = Q_(0.61, 'GHz')

repump_935 = a.transitions[b12, d32]
repump_935.A = Q_(120, 'kHz')

print(d32['F=2']['mF=-1'].level)
print(repump_935.subtransitions())
```
