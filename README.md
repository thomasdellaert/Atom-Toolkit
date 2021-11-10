# Atom-Toolkit
A python package that seeks to provide a convenient way to manipulate atomic physics data in a notebook environment. The package allows you to:

 - create an Atom object that contains information about spectroscopic energy levels
	 - Hyperfine and Zeeman structure is automatically generated
 - create transitions between energy levels
	 - transitions between any sublevels are also computed, with appropriate selection rules and angular momentum considerations
	 - Make diagrams of atomic structure
	 - Plot transition spectra

### Very basic example usage:

    from atom-toolkit.atom import Atom
    from atom-toolkit import IO
    import pandas as pd

	df = IO.load_NIST_data('Yb II')
    a = Atom.from_dataframe(df, name='171Yb II', I=0.5)

	print(a.levels)
