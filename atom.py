import csv
import pickle
import re
import itertools
import math
from typing import List

import networkx as nx
import numpy as np
import pint
from indexedproperty import indexedproperty
from tqdm import tqdm

from config import Q_, ureg
from transition_strengths import wigner3j, wigner6j

Hz = ureg.hertz


class Term:
    def __init__(self,
                 conf: str, term: str, J: float or str,
                 F: float or str = None, mF: float or str = None,
                 percentage=100.0):
        """
        A Term contains all the good quantum numbers of an EnergyLevel, as well as any ancestor terms that can be
        extracted from the configuration. These ancestor terms can (sometimes? often?) be used to convert between
        different couplings, among other things.

        :param conf: the configuration of the term, formatted according to NIST's conventions
        :param term: The LS (Russell-Saunders), LK, JK, or JJ coupled term symbol, excluding the J value
        :param J: The J value for the term symbol
        :param F: The F value for the term symbol
        :param mF: The m_F value for the term symbol
        :param percentage: The leading percentage of the term. Correlates with how well the term's behavior
                            physically corresponds with what you expect
        """
        self.conf = conf
        self.term = term
        self.percentage = percentage
        self.parity = (1 if '*' in self.term else 0)

        self.J = self.frac_to_float(J)
        self.F = self.frac_to_float(F)
        self.mF = self.frac_to_float(mF)

        self.term_name = f'{self.term}{self.J_frac}'
        if self.F is not None:
            self.term_name += f' F={self.F_frac}'
        if self.mF is not None:
            self.term_name += f' mF={self.mF_frac}'

        self.name = f'{self.conf} {self.term_name}'

        self.coupling = self.get_coupling()

        self.lc, self.sc, self.lo, self.so, self.jc, self.jo, self.l, self.s, self.k = self.get_quantum_nums()

    # region frac properties

    @property
    def J_frac(self):
        return self.float_to_frac(self.J)

    @J_frac.setter
    def J_frac(self, value):
        self.J = self.frac_to_float(value)

    @property
    def F_frac(self):
        return self.float_to_frac(self.F)

    @F_frac.setter
    def F_frac(self, value):
        self.F = self.frac_to_float(value)

    @property
    def mF_frac(self):
        return self.float_to_frac(self.mF)

    @mF_frac.setter
    def mF_frac(self, value):
        self.mF = self.frac_to_float(value)

    # endregion

    def get_quantum_nums(self):
        """
        Calls the appropriate term parsing method in order to (hopefully) extract all usable information
        :return: a tuple of useful quantum numbers
        """
        lc = sc = lo = so = jc = jo = l = s = k = None
        if self.coupling == 'LS':
            l, s = self.parse_LS_term()
        elif self.coupling == 'JK':
            lc, sc, lo, so, jc, k = self.parse_JK_term()
        elif self.coupling == 'JJ':
            lc, sc, lo, so, jc, jo = self.parse_JJ_term()
        elif self.coupling == 'LK':
            lc, sc, lo, so, l, k = self.parse_LK_term()
        return lc, sc, lo, so, jc, jo, l, s, k

    # region parsing functions

    def get_coupling(self):
        """
        :return: A string corresponding to the coupling detected in the term symbol
        """
        if '[' in self.term:
            if self.term[0] in 'SPDFGHIKMNOPQRTUVWXYZ':
                return 'LK'
            else:
                return 'JK'
        elif '(' in self.term:
            return 'JJ'
        elif len(self.term) > 1:
            return 'LS'
        else:
            return "unknown"

    def parse_LS_term(self):
        """
        Parses an LS-coupled term. Looks for the following forms in self.term:
            {2S+1}{L}
        Examples:
            2F, 3P*, 1S, 6L
        :return: L, S
        """
        # find the following forms: 2F, 3P*, and extract the relevant substrings
        [(ss, ls)] = re.findall(r'(\d+)([A-Z])', self.term)
        s = (float(ss) - 1) / 2
        l = self.let_to_l(ls)
        return l, s

    def parse_JK_term(self):
        """
        Parses a JK-coupled term.
        Looks for the following forms in self.term:
            {2S+1}[{K}]
        Examples:
            2[3/2], 3[4]*, 1[11/2]
        Looks for the following forms in self.conf:
            {2s+1}{L}<{J}>
        Examples:
            3D<2>, 2F<7/2>

        :return: Lc, Sc, Lo, So, Jc, K
        """
        # find the following forms: 3D<2>, and extract the relevant substrings
        relevant_parts = re.findall(r'(\d+)([A-Z])\*?(?:<(.+?)>)?', self.conf)
        if len(relevant_parts) == 2:
            [(scs, lcs, jcs), (_, los, _)] = relevant_parts
        else:
            [(scs, lcs, jcs)] = relevant_parts
            los = self.conf[-1]
        # find the following forms: 2[3/2], 3[4]*, and extract the relevant substrings
        [(sos, ks)] = re.findall(r'(\d+)\[(.+?)]', self.term)

        jc = self.frac_to_float(jcs)
        k = self.frac_to_float(ks)
        sc = (float(scs) - 1) / 2
        so = (float(sos) - 1) / 2
        lc = self.let_to_l(lcs)
        lo = self.let_to_l(los)

        return lc, sc, lo, so, jc, k

    def parse_JJ_term(self):
        """
         Parses a JJ-coupled term.
         Looks for the following forms in self.conf:
             {2S+1}{L}<{J}>
             ({Jc, Jo})<{J}>
         Examples:
             3D<2>, 2F<7/2>
             (2, 1/2)<5/2>

         :return: Lc, Sc, Lo, So, Jc, Jo
         """
        # find the following forms: 3D<2>, 7p<3/2>, (8,5/2)*<21/2>, and extract the relevant substrings
        relevant_parts = re.findall(r'(?:(\d+)([A-Za-z])|\(.+?\))\*?<(.+?)>', self.conf)
        if len(relevant_parts) == 0:  # sometimes the ancestor terms are in the term, not in the config
            relevant_parts = re.findall(r'(?:(\d+)([A-Za-z])|\(.+?\))\*?<(.+?)>', self.term)
        [(scs, lcs, jcs), (sos, los, jos)] = relevant_parts

        jc = self.frac_to_float(jcs)
        jo = self.frac_to_float(jos)
        lc = self.let_to_l(lcs)
        lo = self.let_to_l(los)
        if lcs.isupper():
            sc = (float(scs) - 1) / 2
        elif lcs.islower():
            sc = 0.0
        else:
            sc = None
        if los.isupper():
            so = (float(sos) - 1) / 2
        elif los.islower():
            so = 0.0
        else:
            so = None

        return lc, sc, lo, so, jc, jo

    def parse_LK_term(self):
        """
        Parses a JK-coupled term.
        Looks for the following forms in self.term:
            {L} {2S+1}[{K}]
        Examples:
            P 2[3/2], D 3[4]*, G 1[11/2]
        Looks for the following forms in self.conf:
            {2s+1}{L}<{J}>
        Examples:
            3D, 2F*

        :return: Lc, Sc, Lo, So, L, K
        """
        # find the following forms: 3D, and extract the relevant substrings
        relevant_parts = re.findall(r'\((\d+)([A-Z])\*?\)', self.conf)
        if len(relevant_parts) == 2:
            [(scs, lcs), (_, los)] = relevant_parts
        else:
            [(scs, lcs)] = relevant_parts
            los = self.conf[-1]
        # find the following forms: D 2[3/2], P* 3[4]*, and extract the relevant substrings
        [(ls, sos, ks)] = re.findall(r'([A-Z])\*? ?(\d+)\[(.+?)]', self.term)

        k = self.frac_to_float(ks)
        sc = (float(scs) - 1) / 2
        so = (float(sos) - 1) / 2
        l = self.let_to_l(ls)
        lc = self.let_to_l(lcs)
        lo = self.let_to_l(los)

        return lc, sc, lo, so, l, k

    # endregion

    @classmethod
    def from_dataframe(cls, df, i=0):
        return Term(df["Configuration"][i], df["Term"][i], df["J"][i], percentage=df["Leading percentages"])

    @staticmethod
    def let_to_l(let: str) -> int or None:
        """
        :param let: a single character in "SPDFGHIKLMNOQRTUVWXYZ" or ""
        :return: the L-value corresponding to the letter in spectroscopic notation
        """
        if let == '':
            return None
        if len(let) > 1:
            raise ValueError('Argument must be a single character in "SPDFGHIKLMNOQRTUVWXYZ" or an empty string')
        return 'SPDFGHIKLMNOQRTUVWXYZ'.index(let.upper())

    @staticmethod
    def l_to_let(l: int) -> str:
        """
        :param l: a positive integer
        :return: the corresponding L-value in spectroscopic notation
        """
        return 'SPDFGHIKLMNOQRTUVWXYZ'[l]

    @staticmethod
    def frac_to_float(frac: float or str) -> float or None:
        """

        :param frac: a string formatted as "1/2", "5/2", "3", etc
        :return: the corresponding float
        """
        if frac is None or '':
            return None
        if type(frac) == str:
            if '/' in frac:
                (f1, f2) = frac.split('/')
                return float(f1) / float(f2)
            else:
                try:
                    return float(frac)
                except TypeError:
                    raise ValueError("Please input a fraction-formatted string or a float")
        return frac

    @staticmethod
    def float_to_frac(f: float or str) -> str or None:
        """
        :param f: a half-integer float
        :return: a string formatted as "1/2", "5/2", "3", etc
        """
        if f is None:
            return None
        # Assumes n/2, since all the fractions that appear in term symbols are of that form
        if type(f) == str:
            if '/' in f:
                return f
            try:
                f = float(f)
            except TypeError:
                raise ValueError("Please input either a float or a fraction-formatted string")
        if (2 * f) % 2 == 0:
            return str(int(f))
        else:
            return str(int(f * 2)) + '/2'


class BaseLevel:

    def __init__(self, term: Term, parent):
        self.parent = parent
        self.term = term
        self.atom = self.get_atom()
        self.manifold = self.get_manifold()
        self.name = self.term.name

        self._sublevels = dict()

    def get_atom(self):
        raise NotImplementedError()

    def get_manifold(self):
        raise NotImplementedError()

    def populate_sublevels(self):
        raise NotImplementedError()

    @property
    def level_Hz(self):
        raise NotImplementedError()

    @property
    def level(self):
        return self.level_Hz * Hz

    @level.setter
    def level(self, value: pint.Quantity):
        self.level_Hz = value.to(Hz).magnitude

    @property
    def shift_Hz(self):
        raise NotImplementedError()

    @property
    def shift(self):
        return self.shift_Hz * Hz

    # region dict-like methods

    def __len__(self):
        return len(self._sublevels)

    def __getitem__(self, key):
        return self._sublevels[key]

    def __setitem__(self, key, level):
        level.parent = self
        self._sublevels[key] = level

    def __delitem__(self, key):
        del self._sublevels[key]

    def __iter__(self):
        return iter(self._sublevels)

    def values(self):
        return self._sublevels.values()

    def sublevels(self):
        return self._sublevels.values()

    def keys(self):
        return self._sublevels.keys()

    # endregion


class EnergyLevel(BaseLevel):
    def __init__(self, term: Term, level: pint.Quantity, lande: float = None, parent=None,
                 hfA=Q_(0.0, 'gigahertz'), hfB=Q_(0.0, 'gigahertz'), hfC=Q_(0.0, 'gigahertz')):
        """
        An EnergyLevel represents a single fine-structure manifold in an atom. It contains a number of sublevels,
        which are instances of the HFLevel subclass. An example of a fully instantiated EnergyLevel looks like:
        EnergyLevel
            HFLevel
                ZLevel
                (...)
            HFLevel
                ZLevel
                (...)
            (...)
        EnergyLevels serve as the nodes in the internal graph of an Atom object. Its level can be shifted to be
        consistent with a transition that the level participates in

        :param term: a Term object containing the level's quantum numbers
        :param level: the energy of the (center of mass of) the level
        :param lande: the lande-g value of the level
        :param parent: the atom that the level is contained in
        :param hfA: the hyperfine A-coefficient
        :param hfB: the hyperfine B-coefficient
        :param hfC: the hyperfine C-coefficient
        """
        super().__init__(term, parent)
        self._level_Hz = level.to(Hz).magnitude
        self.hfA_Hz, self.hfB_Hz, self.hfC_Hz = hfA.to(Hz).magnitude, hfB.to(Hz).magnitude, hfC.to(Hz).magnitude
        if lande is None:
            try:
                self.lande = self.compute_gJ()
            except NotImplementedError:
                self.lande = 0  # TODO: think about a placeholder value instead?
        else:
            self.lande = lande

    def get_manifold(self):
        return self

    def get_atom(self):
        if self.parent is not None:
            return self.parent
        else:
            return None

    def populate_sublevels(self):
        """
        Populates the sublevels dict with the appropriate hyperfine sublevels for the atom that the EnergyLevel is in
        """
        if isinstance(self.parent, Atom):
            for f in np.arange(abs(self.term.J - self.atom.I), self.term.J + self.atom.I + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=f)
                e = HFLevel(term=t, parent=self)
                self[f'F={f}'] = e

    @property
    def level_Hz(self):
        return self._level_Hz

    @level_Hz.setter
    def level_Hz(self, value: float):
        self._level_Hz = value

    @property
    def shift_Hz(self):
        return self.level_Hz

    # region HF coefficients

    @property
    def hfA(self):
        return self.hfA_Hz * Hz

    @hfA.setter
    def hfA(self, value):
        self.hfA_Hz = value.to(Hz).magnitude

    @property
    def hfB(self):
        return self.hfB_Hz * Hz

    @hfB.setter
    def hfB(self, value):
        self.hfB_Hz = value.to(Hz).magnitude

    @property
    def hfC(self):
        return self.hfC_Hz * Hz

    @hfC.setter
    def hfC(self, value):
        self.hfC_Hz = value.to(Hz).magnitude

    # endregion

    def compute_gJ(self):
        """
        Computes the Lande g-value of an LS-coupled term.
        :return: gJ
        """
        if self.term.coupling != 'LS':
            raise NotImplementedError("Unable to compute g_J for non-LS-coupled terms")
        J = self.term.J
        L = self.term.l
        S = self.term.s
        return 1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

    def populate_internal_transitions(self, include_zeeman=True):
        if self.atom is None:
            raise RuntimeError('EnergyLevel needs to be contained in an atom to add transitions')
        hf_pairs = list(itertools.product(list(self.values()), repeat=2))
        for pair in hf_pairs:
            # TODO: Think about a way to implement relative transition strengths in an elegant way
            # TODO: Check that there aren't any wacky selection rules here
            transition = Transition(pair[0], pair[1])
            self.atom.add_transition(transition)
            if include_zeeman:
                for z0 in list(pair[0].values()):
                    for z1 in list(pair[1].values()):
                        if abs(z0.term.mF - z1.term.mF) <= 1:
                            transition = Transition(z0, z1)
                            self.atom.add_transition(transition)

    @classmethod
    def from_dataframe(cls, df, i=0):
        t = Term.from_dataframe(df, i)
        return EnergyLevel(t, df["Level (cm-1)"][i], lande=df["Lande"][i])


class HFLevel(BaseLevel):
    def __init__(self, term: Term, parent=None):
        """
        An HFLevel represents a hyperfine sublevel of an energy level. It lives inside an EnergyLevel,
        and contains Zeeman-sublevel ZLevel objects.
        An HFLevel's level is assigned in the constructor, but in practice, the level defines where it
        is based on its shift relative to its parent level

        :param term: a Term object containing the level's quantum numbers
        :param parent: the atom that the level is contained in
        """
        super().__init__(term, parent)
        self.gF = self.compute_gF()

    def get_manifold(self):
        return self.parent

    def get_atom(self):
        return self.parent.atom

    def populate_sublevels(self):
        """Populates the sublevels dict with the appropriate Zeeman sublevels"""
        if isinstance(self.parent, EnergyLevel):
            for mf in np.arange(-self.term.F, self.term.F + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=self.term.F, mF=mf)
                e = ZLevel(term=t, parent=self)
                self[f'mF={mf}'] = e

    def compute_hf_shift(self):
        """
        Computes the hyperfine shift of a level given the EnergyLevel's hyperfine coefficients and an F level
        :return: the shift of the level
        """
        J = self.term.J
        I = self.atom.I
        F = self.term.F

        IdotJ = 0.5 * (F * (F + 1) - J * (J + 1) - I * (I + 1))

        FM1 = IdotJ

        if J <= 0.5 or I <= 0.5:
            FE2 = 0
        else:
            FE2 = (3 * IdotJ ** 2 + 1.5 * IdotJ - I * (I + 1) * J * (J + 1)) / \
                  (2.0 * I * (2.0 * I - 1.0) * J * (2.0 * J - 1.0))

        if J <= 1 or I <= 1:
            FM3 = 0
        else:
            FM3 = (10 * IdotJ ** 3 + 20 * IdotJ ** 2 + 2 * IdotJ * (
                    -3 * I * (I + 1) * J * (J + 1) + I * (I + 1) + J * (J + 1) + 3)
                   - 5 * I * (I + 1) * J * (J + 1)) / (I * (I - 1) * J * (J - 1) * (2 * J - 1))
        return self.manifold.hfA_Hz * FM1 + self.manifold.hfB_Hz * FE2 + self.manifold.hfC_Hz * FM3

    @property
    def shift_Hz(self):
        return self.compute_hf_shift()

    @property
    def level(self):
        """When asked, sublevels calculate their position relative to their parent level"""
        return (self.parent.level_Hz + self.shift_Hz) * Hz

    @property
    def level_Hz(self):
        return self.parent.level_Hz + self.shift_Hz

    @level.setter
    def level(self, value: pint.Quantity):
        """
        When the level of a sublevel is changed (for example, by a transition being added), the sublevel tells its
        parent to move by the same amount instead of moving itself
        :param value: the new level of the hyperfine level
        """
        shift = value - self.level
        self.parent.level += shift

    def compute_gF(self):
        """
        Computes the Lande g-value for the hyperfine level
        :return: gF
        """
        F = self.term.F
        J = self.term.J
        I = self.atom.I
        if F != 0:
            return self.manifold.lande * (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
        return 0.0


class ZLevel(HFLevel):
    def get_manifold(self):
        return self.parent.parent

    def populate_sublevels(self):
        """A Zeeman sublevel will have no further sublevels."""
        pass

    @property
    def shift_Hz(self):
        return self.gF * self.term.mF * 1.39962449361e6 * self.atom.B_gauss


class Transition:
    def __init__(self, E1: EnergyLevel, E2: EnergyLevel, freq=None, A: pint.Quantity = None,
                 name=None, update_mode='upper', parent=None):
        """
        A transition contains information about the transition between two EnergyLevels. When instantiated
        with a set frequency, it can move one of the EnergyLevels in order to make the energy difference
        between them consistent with the transition. This allows you to accommodate isotope shifts/more
        precise spectroscopy, etc.

        :param E1: EnergyLevel 1
        :param E2: EnergyLevel 2
        :param freq: a pint Quantity representing the transition frequency. (can be any quantity that pint can
                     convert to a frequency, such as a wavenumber or a wavelength)
        :param A: The Einstein A coefficient of the transition, corresponding to the natural linewidth
        :param name: the name of the transition
        :param update_mode: how the transition should deal with moving EnergyLevels to resolve inconsistencies:
            upper: move the upper level
            lower: move the lower level
            ignore: ignore the conflict
        """
        self.E_1 = E1
        self.E_2 = E2
        self.A = A
        self.parent = parent
        self.allowed_types = self.transition_allowed()
        if self.E_2.level > self.E_1.level:
            self.E_upper = self.E_2
            self.E_lower = self.E_1
        else:
            self.E_upper = self.E_1
            self.E_lower = self.E_2
        self.name = name
        if self.name is None:
            self.name = f'{self.E_1.name} -> {self.E_2.name}'
        self.set_freq = freq

    @property
    def freq_Hz(self):
        return abs(self.E_1.level_Hz - self.E_2.level_Hz)

    @property
    def freq(self):
        return self.freq_Hz * Hz

    @property
    def wl(self):
        return self.freq.to(ureg.nanometer)

    def transition_allowed(self):
        I = self.E_1.atom.I
        J0, J1 = self.E_1.term.J, self.E_2.term.J
        p0, p1 = self.E_1.term.parity, self.E_2.term.parity
        if type(self.E_1) != type(self.E_2):
            return 0b000
        return 0b001 * (np.abs(J1 - J0) <= 1.0 and p0 != p1) | \
               0b010 * (np.abs(J1 - J0) <= 1.0 and p0 == p1) | \
               0b100 * (np.abs(J1 - J0) <= 2.0 and p0 == p1)

class HFTransition(Transition):
    def transition_allowed(self):
        I = self.E_1.atom.I
        J0, J1 = self.E_1.term.J, self.E_2.term.J
        F0, F1 = self.E_1.term.F, self.E_2.term.F
        init = self.parent.allowed_types
        ret = 0b000
        if init & 1:
            ret |= 0b001 * wigner6j(J0, J1, 1, F1, F0, I) != 0.0
        if (init >> 1) & 1:
            ret |= 0b010 * wigner6j(J0, J1, 1, F1, F0, I) != 0.0
        if (init >> 2) & 1:
            ret |= 0b100 * wigner6j(J0, J1, 2, F1, F0, I) != 0.0
        return ret

class ZTransition(Transition):
    def transition_allowed(self):
        F0, F1 = self.E_1.term.F, self.E_2.term.F
        mF0, mF1 = self.E_1.term.mF, self.E_2.term.mF
        init = self.parent.allowed_types
        ret = 0b000
        if init & 1:
            ret |= 0b001 * sum([wigner3j(F1, 1, F0, -mF1, q, mF0) for q in [-1, 0, 1]]) != 0.0
        if (init >> 1) & 1:
            ret |= 0b010 * sum([wigner3j(F1, 1, F0, -mF1, q, mF0) for q in [-1, 0, 1]]) != 0.0
        if (init >> 2) & 1:
            ret |= 0b100 * sum([wigner3j(F1, 2, F0, -mF1, q, mF0) for q in [-2, -1, 0, 1, 2]]) != 0.0
        return ret

# noinspection PyPropertyDefinition
class Atom:
    def __init__(self, name: str, I: float = 0.0, B=Q_(0.0, 'G'),
                 levels: List[EnergyLevel] = None, transitions: List[Transition] = None):
        """
        TODO: docstring

        :param name: The Atom's name
        :param I: A half-integer (0.5, 2.0) , or a string formatted like one ('3', '7/2') representing the
            nuclear spin of the atom
        :param levels: Any EnergyLevel objects you want to instantiate the Atom with
        :param transitions: Any Transition objects you want to instantiate the Atom with. Transitions when
            added will bring their EnergyLevels with them
        """
        self.name = name
        self.I = Term.frac_to_float(I)
        self.B_gauss = B.to(ureg.gauss).magnitude
        self.levelsModel = nx.Graph()
        self.hfModel = nx.Graph()
        self.zModel = nx.Graph()
        if levels is not None:
            for level in levels:
                self.add_level(level)
        if transitions is not None:
            for transition in transitions:
                self.add_transition(transition)

    def add_level(self, level: EnergyLevel, key=None):
        """
        Adds a level to the internal graph model. When the level is added, it calculates its sublevels.
        Add the sublevels to the internal graphs as well.
        TODO: accept a list of levels
        :param level: the EnergyLevel to be added
        :param key: A custom name, if desired. Otherwise defaults to the EnergyLevel's name
        :return:
        """
        if key is None:
            key = level.name
        if type(level) != EnergyLevel:
            level = level.manifold
        level.atom = self
        level.parent = self
        level.populate_sublevels()
        self.levelsModel.add_node(key, level=level)
        for sublevel in list(level.values()):
            self.hfModel.add_node(sublevel.name, level=sublevel)
            sublevel.populate_sublevels()
            for z_level in list(sublevel.values()):
                self.zModel.add_node(z_level.name, level=z_level)

    def add_transition(self, transition: Transition, subtransitions=False):
        """
        Add a transition between two EnergyLevels in the atom. If either of the referenced levels aren't in the atom
        yet, they will be added.
        TODO: accept a list of Transitions
        :param transition: the Transition to be added
        :param subtransitions: what types of sub-transitions to add
        :return:
        """
        #TODO: when adding a transition with a fixed freq, move the energy levels appropriately

        if type(transition.E_1) == EnergyLevel:
            self.levelsModel.add_edge(transition.E_1.name, transition.E_2.name, transition=transition)
            if subtransitions:
                for pair in list(itertools.product(list(transition.E_1.sublevels()), list(transition.E_2.sublevels()))):
                    t = HFTransition(pair[0], pair[1], parent=transition)
                    if t.allowed_types & transition.allowed_types:
                        self.add_transition(t, subtransitions=subtransitions)
        elif type(transition.E_1) == HFLevel:
            self.hfModel.add_edge(transition.E_1.name, transition.E_2.name, transition=transition)
            if subtransitions:
                for pair in list(itertools.product(list(transition.E_1.sublevels()), list(transition.E_2.sublevels()))):
                    t = ZTransition(pair[0], pair[1], parent=transition)
                    if t.allowed_types & transition.allowed_types:
                        self.add_transition(t, subtransitions=subtransitions)
        elif type(transition.E_1) == ZLevel:
            self.zModel.add_edge(transition.E_1.name, transition.E_2.name, transition=transition)

    @property
    def B(self):
        return self.B_gauss * ureg.gauss

    @B.setter
    def B(self, value: pint.Quantity):
        self.B_gauss = value.to(ureg.gauss).magnitude

    # region levels property methods

    @indexedproperty
    def levels(self, key):
        return nx.get_node_attributes(self.levelsModel, 'level')[key]

    @levels.deleter
    def levels(self, key):
        self.levelsModel.remove_node(key)

    @levels.setter
    def levels(self, key, value):
        self.add_level(value, key=key)

    @levels.append
    def levels(self, value):
        self.add_level(value)

    @levels.values
    def levels(self):
        return nx.get_node_attributes(self.levelsModel, 'level').values()

    @levels.keys
    def levels(self):
        return nx.get_node_attributes(self.levelsModel, 'level').keys()

    @levels.iter
    def levels(self):
        return nx.get_node_attributes(self.levelsModel, 'level').__iter__()

    @levels.len
    def levels(self):
        return len(nx.get_node_attributes(self.levelsModel, 'level'))

    # endregion

    # region transitions property methods

    @indexedproperty
    def transitions(self, key):
        return nx.get_edge_attributes(self.levelsModel, 'transition')[key]

    @transitions.setter
    def transitions(self, value):
        self.add_transition(value)

    @transitions.deleter
    def transitions(self, level1, level2):
        self.levelsModel.remove_edge(level1, level2)

    @transitions.append
    def transitions(self, value):
        self.add_transition(value)

    @transitions.len
    def transitions(self):
        return len(nx.get_edge_attributes(self.levelsModel, 'transition'))

    @transitions.values
    def transitions(self):
        return nx.get_edge_attributes(self.levelsModel, 'transition').values()

    @transitions.iter
    def transitions(self):
        return nx.get_edge_attributes(self.levelsModel, 'transition').__iter__()

    @transitions.keys
    def transitions(self):
        return nx.get_edge_attributes(self.levelsModel, 'transition').keys()

    # endregion

    # region loading/unloading methods

    def to_pickle(self, filename):
        if filename is None:
            filename = self.name
        try:
            if filename.split(".", -1)[1] != "atom":
                filename = filename + ".atom"
        except IndexError:
            filename = filename + ".atom"
        file = open(filename, "wb")
        pickle.dump(self, file)
        file.close()

    @classmethod
    def from_pickle(cls, filename):
        file = open(filename, "rb")
        p = pickle.load(file)
        file.close()
        return p

    @classmethod
    def from_dataframe(cls, df, name, I=0.0, num_levels=None, B=Q_(0.0, 'G'), **kwargs):
        if num_levels is None:
            num_levels = len(df)
        a = Atom(name, I=I, B=B)
        rows = tqdm(range(num_levels))
        for i in rows:
            try:
                e = EnergyLevel.from_dataframe(df, i)
                rows.set_description(f'adding level {e.name:109}')
                a.add_level(e)
            except KeyError:
                pass
        return a

    @classmethod
    def generate_full_from_dataframe(cls, df, name, I=0.0, **kwargs):
        """

        :param df:
        :param name:
        :param I:
        :param kwargs:
            'transitions_csv', 'transitions_df', 'hf_csv', 'subtransitions'
        :return:
        """
        a = Atom.from_dataframe(df, name, I, **kwargs)
        if 'hf_csv' in kwargs:
            try:
                a.apply_hf_csv(kwargs['hf_csv'])
            except FileNotFoundError:
                pass
        if 'transitions_csv' in kwargs:
            a.apply_transition_csv(kwargs['transitions_csv'])
        elif 'transitions_df' in kwargs:
            if kwargs['transitions_df'] is not None:
                a.populate_transitions_df(kwargs['transitions_df'], **kwargs)
            else:
                a.populate_transitions(**kwargs)
        else:
            a.populate_transitions(allowed=0b001, **kwargs)
        a.populate_internal_transitions()
        return a

    # endregion

    def populate_transitions(self, allowed=0b111, subtransitions=True, **kwargs):
        """
        Iterate through every pair of levels in the atom, checking whether a given transition is 'allowed'
        and adding it if it is. Since this involves calculating Clebsch-Gordan coefficients for every possible
        pair of levels, it's slow and scales horribly with atom size. When possible, give a dataframe of transitions.

        :param allowed: a bit string. 0b[E2][M1][E1]
                        also accepts the following formats:  #TODO make this true
                            - a tuple of bools:  ([E1][M1][E2])
                            - a dict: {'E1': [E1], 'M1': [M1], 'E2': [E2]}
        :param subtransitions: whether to generate subtransitions when the transitions are added
        :param kwargs: none
        """
        max_to_try = 20

        js = {J: [lvl for lvl in self.levels.values() if lvl.term.J == J]
              for J in np.arange(0, max_to_try+1, 0.5)}
        for delta_j in np.arange(0, int(math.log(allowed, 2)/2)+2):  # (len(allowed)+2)/2): #FIXME
            set0 = (js[j] for j in list(js.keys()))
            set1 = (js[j+delta_j] for j in list(js.keys())[:int(-(delta_j*2+1))])
            j_pairs = zip(set0, set1)
            level_pairs = tqdm(list(itertools.chain.from_iterable([itertools.product(j1, j2) for j1, j2 in j_pairs])))
            for pair in level_pairs:
                if ((pair[0].name, pair[1].name) not in self.transitions.keys()) and \
                        ((pair[0].name, pair[1].name) not in self.transitions.keys()):
                    t = Transition(pair[0], pair[1])
                    if t.allowed_types & allowed == 0:
                        del t
                    else:
                        level_pairs.set_description(f'processing ΔJ={delta_j:3} transition {t.name:93}')
                        self.add_transition(t, subtransitions)

    def populate_internal_transitions(self):
        """
        Iterates over each energy level and populates the internal M1 transitions of each one
        """
        levels = tqdm(list(self.levels.values()))
        for level in levels:
            level.populate_internal_transitions()
            levels.set_description(f'adding internal transitions to {level.name:91}')

    def populate_transitions_df(self, df, subtransitions=True, **kwargs):
        """
        Load transitions into the Atom from a dataframe generated from the IO module's load_transition_data function
        """
        rows = tqdm(list(df.iterrows()))
        for _, row in rows:
            j_l = Term.float_to_frac(float(row['j_l']))
            j_u = Term.float_to_frac(float(row['j_u']))
            try:
                e1 = self.levels[f'{row["conf_l"]} {row["term_l"]}{j_l}']
                e2 = self.levels[f'{row["conf_u"]} {row["term_u"]}{j_u}']
                freq = row["freq"]
                A = row["A"]
                t = Transition(e1, e2, freq=freq, A=A)
                rows.set_description(f'adding transition {t.name:104}')
                self.add_transition(t, subtransitions=subtransitions)
            except KeyError:
                pass

    def generate_hf_csv(self, filename=None, blank=False, def_A=Q_(0.0, 'gigahertz')):
        if filename is None:
            filename = f'{self.name}_Hyperfine.csv'
        if not blank:
            rows_to_write = [
                [level.name, (level.hfA if level.hfA != Q_(0.0, 'gigahertz') else def_A), level.hfB, level.hfC]
                for level in list(self.levels.values())]
        else:
            rows_to_write = [[level.name, Q_(0.0, 'GHz'), Q_(0.0, 'GHz'), Q_(0.0, 'GHz')] for level in
                             list(self.levels.values())]
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

    def apply_hf_csv(self, filename):
        import csv
        with open(filename, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    name, hfA, hfB, hfC = row
                    self.levels[name].hfA = Q_(hfA)
                    self.levels[name].hfB = Q_(hfB)
                    self.levels[name].hfC = Q_(hfC)
                except KeyError:
                    pass

    def generate_transition_csv(self, filename=None):
        if filename is None:
            filename = f'{self.name}_Transitions.csv'
        pass

    def apply_transition_csv(self, filename):
        pass
        # TODO

    def linked_transitions(self, level):
        adjacent = self.levelsModel.adj[level]
        return {k: t['transition'] for k, t in adjacent.items()}

    def compute_branching_ratios(self, key):
        transitions = self.linked_transitions(key)
        A_coeffs = {}
        for n, t in transitions.items():
            if t.E_upper.name == key:
                A_coeffs[n] = t.A
        totalAs = np.sum(list(A_coeffs.values()))
        ratios = {k: t/totalAs for k, t in A_coeffs.items()}
        return ratios
