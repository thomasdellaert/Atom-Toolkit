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
        if x == x0:
            return 1.0
        return 0.0

    def width_func(self, **kwargs):
        return 1.0

    def compute(self, x0, num_points: int = 1000, **kwargs):
        width = self.width_func(**kwargs)
        x_values = np.linspace(x0 - width, x0 + width, num_points)
        y_values = self.shape_func(x_values, x0, **kwargs)
        return x_values, y_values

class LorentzianLineShape(LineShape):
    def __init__(self, gamma, ampl):
        super().__init__()
        self.gamma = gamma
        self.ampl = ampl

    def shape_func(self, x, x0, **kwargs):
        width = self.gamma
        if 'A_coeff' in kwargs:
            width += kwargs['A_coeff']
        return lorentzian(x, x0, width, self.ampl)

    def width_func(self, padding=20.0, **kwargs):
        if 'A_coeff' in kwargs:
            return padding*(self.gamma + kwargs['A_coeff'])
        return padding * self.gamma

class ModLorentzianLineShape(LineShape):
    def __init__(self, gamma, ampl, depth, mod_freq_GHz, num_sidebands=None):
        super().__init__()
        self.gamma = gamma
        self.ampl = ampl
        self.mod_freq_GHz = mod_freq_GHz
        self.depth = depth
        self.num_sidebands = 0
        if num_sidebands is None:
            while (self.depth/2)**self.num_sidebands > 0.01 * np.math.factorial(self.num_sidebands):
                self.num_sidebands += 1
        else:
            self.num_sidebands = num_sidebands

    def shape_func(self, x, x0, **kwargs):
        width = self.gamma
        if 'A_coeff' in kwargs:
            width += kwargs['A_coeff'] / (1e6 * 2 * np.pi)
        tot = np.zeros_like(x)
        tot += lorentzian(x, x0, width, ampl=self.ampl*float(jv(0, self.depth)**2))
        for i in range(1, self.num_sidebands+1):
            mod_ampl = self.ampl*float(jv(i, self.depth)**2)
            tot += lorentzian(x, x0+i*self.mod_freq_GHz, width, ampl=mod_ampl)
            tot += lorentzian(x, x0-i*self.mod_freq_GHz, width, ampl=mod_ampl)
        return tot

    def width_func(self, padding=20.0, **kwargs):
        width = self.gamma
        if 'A_coeff' in kwargs:
            width += kwargs['A_coeff'] / (1e6 * 2 * np.pi)
        return self.mod_freq_GHz * self.num_sidebands + padding * width
