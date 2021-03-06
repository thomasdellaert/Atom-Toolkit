from setuptools import setup

setup(
    name='Atom-Toolkit',
    version='0.2.0.dev1',
    packages=['atomtoolkit', 'atomtoolkit.render'],
    url='https://github.com/thomasdellaert/Atom-Toolkit',
    license='GNU GPLv3',
    author='Thomas Dellaert',
    author_email='dellaert.thomas@gmail.com',
    description='A package for handling and displaying data for atomic physics calculations',
    python_requires='>=3.8',
    install_requires=[
          'pint', 'pandas', 'pint-pandas', 'numpy', 'networkx', 'sympy', 'matplotlib', 'scipy', 'tqdm'
      ],
)
