from pathlib import Path
from setuptools import setup

setup(
    name='Atom-Toolkit',
    version='0.2.0',
    packages=['atomtoolkit', 'atomtoolkit.render', 'atomtoolkit.species'],
    # The bundled NIST ASD CSVs are data, not modules, so they need declaring explicitly
    # or the wheel ships a species library with nothing to build from.
    package_data={'atomtoolkit': ['resources/*.csv']},
    include_package_data=True,
    url='https://github.com/thomasdellaert/Atom-Toolkit',
    license='GNU GPLv3',
    author='Thomas Dellaert',
    author_email='dellaert.thomas@gmail.com',
    description='A package for handling and displaying data for atomic physics calculations',
    long_description=Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    install_requires=[
          'pint', 'pandas', 'pint-pandas', 'numpy', 'networkx', 'sympy', 'matplotlib', 'scipy', 'tqdm',
          'svgwrite'
      ],
)
