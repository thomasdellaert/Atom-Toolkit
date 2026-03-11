from . import Q_, ureg
import numpy as np

c = ureg.speed_of_light


class Polarization:
    def __init__(self, pol):
        if not isinstance(pol, np.ndarray):
            pol = np.array(pol)
        pol = pol.astype(np.complex64)
        pol = pol/np.linalg.norm(pol)
        assert np.linalg.norm(pol) - 1 < 1e-6
        self._pol = pol

    @property
    def xyz(self):
        return self._pol

    @xyz.setter
    def xyz(self, pol):
        self._pol = pol

    @property
    def spherical(self):
        return self.xyz_to_sigma(self._pol)

    @spherical.setter
    def spherical(self, pol_spherical):
        self._pol = self.sigma_to_xyz(pol_spherical)

    @staticmethod
    def xyz_to_sigma(pol):
        """
        Convert between the frame defined by the external magnetic field,
        and the spherical tensor frame of sigma-, pi, sigma+
        :param pol: the polarization of the input light in the xyz frame, with the z axis defined as the quantization axis
        :return: the polarization components as spherical tensors
        """
        return np.dot(np.array([[2 ** (-0.5), -2j ** (-0.5), 0],
                                [0, 0, 1],
                                [2 ** (-0.5), 2j ** (-0.5), 0]]), pol)

    @staticmethod
    def sigma_to_xyz(pol):
        """
        Convert between the frame defined by spherical tensor frame of sigma-, pi, sigma+,
        and the one defined by the external magnetic field in cartesian coordinates
        :param pol: the polarization of the input light in the xyz frame, with the z axis defined as the quantization axis
        :return: the polarization components as spherical tensors
        """
        return np.dot(np.array([[0.5, 0, 0.5],
                                [0.5j, 0, -0.5j],
                                [0, 1, 0]]), pol)


class LightField:
    def __init__(self, k: np.array, i: float, pol: Polarization = Polarization([0, 0, 1])):
        """

        :param k:
        :param i:
        :param pol: the polarization of the light: a complex array in the lab frame [x, y, z]
            linear vertical = [0, 0, 1]
            linear in x = [1, 0, 0]
            right circular = 2**(-0.5) * [1, 1j, 0]
        """
        self.k = k
        self.pol = pol
        self.intensity = i

    def E(self, **_):
        return self.pol.xyz * (2 * self.intensity / (c * ureg.vacuum_permittivity)) ** 0.5

    def B(self, **_):
        E = self.E()
        return np.cross(self.k, E.magnitude) * E.units / c


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
        return self.pol.xyz * self.E0 * self.w0 / self.w(z) * np.exp(-r * r / self.w(z) ** 2)

    def B(self, z, r=0.0):
        return 1 / c * np.cross(self.E(z, r), self.pol.xyz)

    def phase_at(self, z):
        return np.arctan(z / self.zr)

    def radius_at(self, z):
        return z * (1 + self.zr ** 2 / z ** 2)

    def w(self, z):
        return self.w0 * np.sqrt(1 + z ** 2 / self.zr ** 2)
