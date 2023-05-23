from . import Q_, ureg
import numpy as np

c = ureg.speed_of_light


class LightField:
    def __init__(self, k: np.array, i: float, pol: np.array = np.array([0., 0., 1.], dtype=np.complex_)):
        """

        :param k:
        :param i:
        :param pol: the polarization of the light: a complex array in the lab frame [x, y, z]
            linear vertical = [0, 0, 1]
            linear in x = [1, 0, 0]
            right circular = 2**(-0.5) * [1, 1j, 0]
        """
        self.k = k
        assert np.linalg.norm(pol) - 1 < 1e-6, "polarization must be a unit vector"
        self.pol = pol
        self.intensity = i

    def E(self, **_):
        return self.pol * (2 * self.intensity / (c * ureg.vacuum_permittivity)) ** 0.5

    def B(self, **_):
        return 1 / c * np.cross(self.k, self.E())


# class LinewidthMixin:
#     def __init__(self, linewidth):
#         self.linewidth = linewidth.to('Hz')
#         self.linewidth_hz = self.linewidth.magnitude
#         self.q = self.k.to('Hz') / self.linewidth.to('Hz')
#
#     def i_at(self):
#         return

class GaussianBeam(LightField):
    def __init__(self, k, pow, pol, w0):
        self.i0 = 2 * pow / (np.pi * w0 ** 2)
        self.w0 = w0
        super(LightField).__init__(k, self.i0, pol)
        self.E0 = super(LightField).E()
        self.zr = np.pi * w0 ** 2 * c * k

    def E(self, z, r=0.0):
        return self.pol * self.E0 * self.w0 / self.w(z) * np.exp(-r * r / self.w(z) ** 2)

    def B(self, z, r=0.0):
        return 1 / c * np.cross(self.E(z, r), self.pol)

    def phase_at(self, z):
        return np.arctan(z / self.zr)

    def radius_at(self, z):
        return z * (1 + self.zr ** 2 / z ** 2)

    def w(self, z):
        return self.w0 * np.sqrt(1 + z ** 2 / self.zr ** 2)
