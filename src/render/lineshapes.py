"""
A set of lineshape functions meant to be used with the spectra module. The functions
themselves are standard, and the LineShape class is meant as a container to encompass
terms like 'width' and 'shape' in the unique terms of each lineshape.

For instance, a LorentzianLineShape has a different meaning for its 'width' than a
GaussianLineShape might
"""

import numpy as np
from scipy.special import jv
from scipy.special import voigt_profile

def lorentzian(x, x0, gamma: float, ampl: float = 1.0):
    return ampl * gamma ** 2 / (4 * (x - x0) ** 2 + gamma ** 2)

def gaussian(x, x0, sigma: float, ampl: float = 1.0):
    return ampl * np.exp(-(x - x0)**2 / (2 * sigma ** 2))

def voigt(x, x0, sigma: float, gamma: float, ampl: float = 1.0):
    return ampl * voigt_profile(x - x0, sigma, gamma)

def mod_lorentzian(x, x0: float, gamma: float, ampl: float = 1.0, depth: float = 0.0, num_sidebands: int = None, mod_freq_GHz: float = 1.0):
    tot = np.zeros_like(x)
    tot += lorentzian(x, x0, gamma, ampl=ampl * float(jv(0, depth) ** 2))
    for i in range(1, num_sidebands + 1):
        mod_ampl = ampl * float(jv(i, depth) ** 2)
        tot += lorentzian(x, x0 + i * mod_freq_GHz, gamma, ampl=mod_ampl)
        tot += lorentzian(x, x0 - i * mod_freq_GHz, gamma, ampl=mod_ampl)
    return tot

class LineShape:
    def __init__(self):
        pass

    @staticmethod
    def shape_func(x, x0, **kwargs):
        raise NotImplementedError

    def width_func(self, **kwargs):
        raise NotImplementedError

    def compute(self, x0, num_points: int = 1000, **kwargs):
        width = self.width_func(**kwargs)
        x_values = np.linspace(x0 - width, x0 + width, num_points)
        y_values = self.shape_func(x_values, x0, **kwargs)
        return x_values, y_values

class LorentzianLineShape(LineShape):
    def __init__(self, gamma, ampl: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def shape_func(self, x, x0, **kwargs):
        width = self.gamma
        if 'A_coeff' in kwargs:
            width += kwargs['A_coeff']
        ampl = 1.0
        if 'ampl' in kwargs:
            ampl = kwargs['ampl']
        return lorentzian(x, x0, width, ampl=ampl)

    def width_func(self, padding=20.0, **kwargs):
        if 'A_coeff' in kwargs:
            return padding*(self.gamma + kwargs['A_coeff'])
        return padding * self.gamma

class ModLorentzianLineShape(LineShape):
    def __init__(self, gamma, depth, mod_freq_GHz, ampl: float = 1.0, num_sidebands=None):
        super().__init__()
        self.gamma = gamma
        self.mod_freq_GHz = mod_freq_GHz
        self.depth = depth
        self.num_sidebands = 0
        if num_sidebands is None:
            while (self.depth/2)**self.num_sidebands > 0.01 * np.math.factorial(self.num_sidebands):
                self.num_sidebands += 1
        else:
            self.num_sidebands = num_sidebands

    def shape_func(self, x, x0, **kwargs):
        if 'ampl' not in kwargs:
            ampl = 1.0
        else:
            ampl = kwargs['ampl']
        width = self.gamma
        if 'A_coeff' in kwargs:
            width += kwargs['A_coeff'] / (1e6 * 2 * np.pi)
        tot = np.zeros_like(x)
        tot += lorentzian(x, x0, width, ampl=ampl*float(jv(0, self.depth)**2))
        for i in range(1, self.num_sidebands+1):
            mod_ampl = ampl*float(jv(i, self.depth)**2)
            tot += lorentzian(x, x0+i*self.mod_freq_GHz, width, ampl=mod_ampl)
            tot += lorentzian(x, x0-i*self.mod_freq_GHz, width, ampl=mod_ampl)
        return tot

    def width_func(self, padding=20.0, **kwargs):
        width = self.gamma
        if 'A_coeff' in kwargs:
            width += kwargs['A_coeff'] / (1e6 * 2 * np.pi)
        return self.mod_freq_GHz * self.num_sidebands + padding * width

class GaussianLineShape(LineShape):
    def __init__(self, sigma, ampl: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def shape_func(self, x, x0, **kwargs):
        width = self.sigma * 2.355
        # TODO: technically with a line/laser of finite linewidth this can broaden into a Voigt profile
        ampl = 1.0
        if 'ampl' in kwargs:
            ampl = kwargs['ampl']
        return gaussian(x, x0, self.sigma, ampl=ampl)

    def width_func(self, padding=20.0, **kwargs):
        return padding * self.sigma