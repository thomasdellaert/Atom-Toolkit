"""
Wigner functions used throughout the package, wrapped in a way that's convenient, and cached for speed.
"""

import functools

try:
    # py3nj is a pain to install, and requires a fortran compiler among other things. Therefore it's optional,
    # and everything can work with sympy if it needs to. Py3nj is a bit faster, and it supports vectorized inputs,
    # so it's preferred.
    from py3nj import wigner3j as _wigner3j
    from py3nj import wigner6j as _wigner6j
    from py3nj import wigner9j as _wigner9j
    PY3NJ_INSTALLED = True
except ImportError:
    from sympy.physics.wigner import wigner_3j as _wigner3j
    from sympy.physics.wigner import wigner_6j as _wigner6j
    from sympy.physics.wigner import wigner_9j as _wigner9j
    from sympy import N
    PY3NJ_INSTALLED = False

def triangle(a, b, c):
    return a+b >= c, b+c >= a, c+a >= b

def wignerPicker(func):
    def wrapper(*args):
        if PY3NJ_INSTALLED:
            # py3nj takes the arguments doubled, and requires them to be integers
            try:
                return func(*map(lambda x: int(x * 2), args))
            except ValueError:
                return 0.0
            except IndexError:
                return 0.0
        else:
            # sympy throws an error when selection rules are violated, but it needs to return 0
            try:
                return N(func(*args))
            except ValueError:
                return 0.0

    return wrapper


wigner3j = (functools.lru_cache(maxsize=None))(wignerPicker(_wigner3j))
wigner6j = (functools.lru_cache(maxsize=None))(wignerPicker(_wigner6j))
wigner9j = (functools.lru_cache(maxsize=None))(wignerPicker(_wigner9j))
