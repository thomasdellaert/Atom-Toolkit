import pickle
import re
from itertools import combinations
from typing import List

from config import Q_

import networkx as nx
import numpy as np
import pint
from indexedproperty import indexedproperty

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

        self.J_frac = self.float_to_frac(J)
        self.F_frac = self.float_to_frac(F)
        self.mF_frac = self.float_to_frac(mF)

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

    # region parsing function

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
        s = (float(ss) - 1)/2
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
        sc = (float(scs) - 1)/2
        so = (float(sos) - 1)/2
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
            sc = (float(scs) - 1)/2
        elif lcs.islower():
            sc = 0.0
        else:
            sc = None
        if los.isupper():
            so = (float(sos) - 1)/2
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

class EnergyLevel:
    def __init__(self, term: Term, level: pint.Quantity, lande: float = None, parent=None, atom=None,
                 hfA=Q_(0.0, 'gigahertz'), hfB=Q_(0.0, 'gigahertz'), hfC=Q_(0.0, 'gigahertz')):
        """
        An EnergyLevel represents a single fine-structure manifold in an atom. It contains a number of sublevels,
        which are instances of the HFLevel subclass. An example of a fully instantiated EnergyLevel looks like:
        EnergyLevel
            HFLevel
                ZLevel
                ZLevel
            HFLevel
                ZLevel
                ZLevel
        EnergyLevels serve as the nodes in the internal graph of an Atom object. Its level can be shifted to be
        consistent with a transition that the level participates in

        :param term: a Term object containing the level's quantum numbers
        :param level: the energy of the (center of mass of) the level
        :param lande: the lande-g value of the level
        :param parent: the atom that the level is contained in
        :param atom: the atom that the level is contained in
        :param hfA: the hyperfine A-coefficient
        :param hfB: the hyperfine B-coefficient
        :param hfC: the hyperfine C-coefficient
        """
        self.parent = parent
        self.manifold = self.get_manifold()
        self.atom = atom
        self.term = term
        self._level = level.to('Hz')
        self.name = self.term.name
        self.hfA, self.hfB, self.hfC = hfA, hfB, hfC
        if lande is None:
            try:
                self.lande = self.compute_gJ()
            except NotImplementedError:
                self.lande = 0  # TODO: think about a placeholder value instead?
        else:
            self.lande = lande
        self._sublevels = {}

    def get_manifold(self):
        return self

    def populate_sublevels(self):
        """
        Populates the sublevels dict with the appropriate hyperfine sublevels for the atom that the EnergyLevel is in
        """
        if isinstance(self.parent, Atom):
            for f in np.arange(abs(self.term.J - self.atom.I), self.term.J + self.atom.I + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=f)
                e = HFLevel(t, self.level, lande=self.lande,
                            parent=self, atom=self.atom,
                            hfA=self.hfA, hfB=self.hfB, hfC=self.hfC)
                self[f'F={f}'] = e

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value

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

    def populate_transitions(self, include_zeeman=True):
        if self.atom is None:
            raise RuntimeError('EnergyLevel needs to be contained in an atom to add transitions')
        hf_pairs = list(combinations(list(self.values()), 2))
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
        # TODO: remove the placeholder hfA value
        return EnergyLevel(t, df["Level (cm-1)"][i], lande=df["Lande"][i])

    # region dict-like methods

    def __len__(self):
        """:return: the number of sublevels"""
        return len(self._sublevels)

    def __getitem__(self, key):
        return self._sublevels[key]

    def __setitem__(self, key, level):
        """adds the energylevel to self._sublevels"""
        level.parent = self
        self._sublevels[key] = level

    def __delitem__(self, key):
        """removes the energylevel from self._sublevels"""
        del self._sublevels[key]

    def __iter__(self):
        return iter(self._sublevels)

    def values(self):  # TODO: maybe rename to sublevels
        return self._sublevels.values()

    def keys(self):
        return self._sublevels.keys()

    # endregion

class HFLevel(EnergyLevel):
    def __init__(self, term: Term, level: pint.Quantity, lande: float = None, parent=None, atom=None,
                 hfA=0.0, hfB=0.0, hfC=0.0):
        """
        An HFLevel represents a hyperfine sublevel of an energy level. It lives inside an EnergyLevel,
        and contains Zeeman-sublevel ZLevel objects.
        An HFLevel's level is assigned in the constructor, but in practice, the level defines where it
        is based on its shift relative to its parent level

        :param term: a Term object containing the level's quantum numbers
        :param level: the energy of the (center of mass of) the level
        :param lande: the lande-g value of the level
        :param parent: the atom that the level is contained in
        :param atom: the atom that the level is contained in
        :param hfA: the hyperfine A-coefficient
        :param hfB: the hyperfine B-coefficient
        :param hfC: the hyperfine C-coefficient
        """
        super(HFLevel, self).__init__(term, level, lande, parent, atom, hfA, hfB, hfC)
        self.gF = self.compute_gF()

    def get_manifold(self):
        return self.parent

    def populate_sublevels(self):
        """Populates the sublevels dict with the appropriate Zeeman sublevels"""
        if isinstance(self.parent, EnergyLevel):
            for mf in np.arange(-self.term.F, self.term.F + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=self.term.F, mF=mf)
                e = ZLevel(t, self.level, lande=self.lande,
                           parent=self, atom=self.atom,
                           hfA=self.hfA, hfB=self.hfB, hfC=self.hfC)
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

        return self.manifold.hfA * FM1 + self.manifold.hfB * FE2 + self.manifold.hfC * FM3

    @property
    def shift(self):
        return self.compute_hf_shift()

    @property
    def level(self):
        """When asked, sublevels calculate their position relative to their parent level"""
        return self.parent.level + self.shift

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
        Computes the Lande g-value the hyperfine level
        :return: gF
        """
        F = self.term.F
        J = self.term.J
        I = self.atom.I
        if F != 0:
            return self.lande * (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
        return 0.0

class ZLevel(HFLevel):
    def get_manifold(self):
        return self.parent.parent

    def populate_sublevels(self):
        """A Zeeman sublevel will have no further sublevels."""
        pass

    @property
    def level(self):
        """When asked, sublevels calculate their position relative to their parent level"""
        return self.parent.level + self.gF * self.term.mF * Q_(1.39962449361, 'MHz/G') * self.atom.B

class Transition:
    def __init__(self, E1: EnergyLevel, E2: EnergyLevel, freq=None, A: pint.Quantity = None,
                 name=None, update_mode='upper'):
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
        self.allowed_types = self.transition_allowed(self.E_1, self.E_2)
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
        if freq is not None:
            # TODO: move this stuff to the Atom. The atom should enforce consistency between
            #  the levels and transitions, not the EnergyLevel
            # if self.E_1.level_constrained or self.E_2.level_constrained:
            #     raise RuntimeWarning('This level has been set by another transition')
            if update_mode == 'upper':
                self.E_upper.level = self.E_lower.level + freq.to('Hz')
            elif update_mode == 'lower':
                self.E_lower.level = self.E_upper.level - freq.to('Hz')
            elif update_mode == 'ignore':
                pass
            else:
                raise ValueError('Accepted arguments to update_mode are "upper", "lower", and "ignore"')

    @property
    def freq(self):
        return abs(self.E_1.level - self.E_2.level)

    @property
    def wl(self):
        return self.freq.to('nm')

    def transition_allowed(self, level_0, level_1):
        from transition_strengths import wigner3j, wigner6j
        I = level_0.atom.I
        J0, J1 = level_0.term.J, level_1.term.J
        p0, p1 = level_0.term.parity, level_1.term.parity
        if type(level_0) != type(level_1):
            return 0b000
        if type(level_0) == EnergyLevel:
            return 0b001 * (np.abs(J1 - J0) <= 1.0 and p0 != p1) | \
                   0b010 * (np.abs(J1 - J0) <= 1.0 and p0 == p1) | \
                   0b100 * (np.abs(J1 - J0) <= 2.0 and p0 == p1)
        if type(level_0) == HFLevel:
            F0, F1 = level_0.term.F, level_1.term.F
            init = self.transition_allowed(level_0.manifold, level_1.manifold)
            ret = 0b000
            if init & 1:
                ret |= 0b001 * wigner6j(J0, J1, 1, F1, F0, I) != 0.0
            if (init >> 1) & 1:
                ret |= 0b010 * wigner6j(J0, J1, 1, F1, F0, I) != 0.0
            if (init >> 2) & 1:
                ret |= 0b100 * wigner6j(J0, J1, 2, F1, F0, I) != 0.0
            return ret
        if type(level_0) == ZLevel:
            F0, F1 = level_0.term.F, level_1.term.F
            mF0, mF1 = level_0.term.mF, level_1.term.mF
            init = self.transition_allowed(level_0.parent, level_1.parent)
            ret = 0b000
            if init & 1:
                ret |= 0b001 * sum([wigner3j(F1, 1, F0, -mF1, q, mF0) for q in [-1, 0, 1]]) != 0.0
            if (init >> 1) & 1:
                ret |= 0b010 * sum([wigner3j(F1, 1, F0, -mF1, q, mF0) for q in [-1, 0, 1]]) != 0.0
            if (init >> 2) & 1:
                ret |= 0b100 * sum([wigner3j(F1, 2, F0, -mF1, q, mF0) for q in [-2, -1, 0, 1, 2]]) != 0.0
            return ret
    # TODO: Add appropriate methods. Things like getting the transition type (via clebsch-gordan math), perhaps
    #  determining color, and computing the transition strength / linewidth given the A coefficient

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
        self.B = B
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
            for z_level in list(sublevel.values()):
                self.zModel.add_node(z_level.name, level=z_level)

    def add_transition(self, transition: Transition):
        """
        Add a transition between two EnergyLevels in the atom. If either of the referenced levels aren't in the atom
        yet, they will be added.
        TODO: accept a list of Transitions
        TODO: optionally populate sub-transitions
        :param transition: the Transition to be added
        :return:
        """
        def check_levels_present(transition, model):
            for t in [transition.E_1, transition.E_2]:
                if t not in nx.get_node_attributes(model, 'level').values():
                    self.add_level(t)

        if type(transition.E_1) == EnergyLevel:
            check_levels_present(transition, self.levelsModel)
            self.levelsModel.add_edge(transition.E_1, transition.E_2, transition=transition)
        elif type(transition.E_1) == HFLevel:
            check_levels_present(transition, self.hfModel)
            self.levelsModel.add_edge(transition.E_1.manifold, transition.E_2.manifold, transition=transition)
            self.hfModel.add_edge(transition.E_1, transition.E_2, transition=transition)
        elif type(transition.E_1) == ZLevel:
            check_levels_present(transition, self.zModel)
            self.levelsModel.add_edge(transition.E_1.manifold, transition.E_2.manifold, transition=transition)
            self.hfModel.add_edge(transition.E_1.parent, transition.E_2.parent, transition=transition)
            self.zModel.add_edge(transition.E_1, transition.E_2, transition=transition)

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
    def from_dataframe(cls, df, name, I=0.0, num_levels=None, B=Q_(0.0, 'G')):
        a = Atom(name, I=I, B=B)
        for i in range(num_levels):
            try:
                e = EnergyLevel.from_dataframe(df, i)
                a.add_level(e)
            except KeyError:
                pass
        return a

    @classmethod
    def generate_full(cls, df, name, I=0.0, hf_csv=None, transition_csv=None):
        pass

    # endregion

    def populate_transitions(self, allowed=0b111):
        level_pairs = list(combinations(list(self.levels.values()), 2))
        for pair in level_pairs:
            t = Transition(pair[0], pair[1])
            if t.allowed_types & allowed == 0:
                del t
            else:
                print(t.name, bin(t.allowed_types))
                self.add_transition(t)

    def generate_hf_csv(self, filename=None, blank=False):
        import csv
        if filename is None:
            filename = f'{self.name}_Hyperfine.csv'
        if not blank:
            rows_to_write = [[level.name, level.hfA, level.hfB, level.hfC] for level in list(self.levels.values())]
        else:
            rows_to_write = [[level.name, Q_(0.0, 'GHz'), Q_(0.0, 'GHz'), Q_(0.0, 'GHz')] for level in list(self.levels.values())]
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

    def apply_hf_csv(self, filename):
        import csv
        with open(filename, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                name, hfA, hfB, hfC = row
                self.levels[name].hfA = Q_(hfA)
                self.levels[name].hfB = Q_(hfB)
                self.levels[name].hfC = Q_(hfC)


if __name__ == '__main__':
    from IO import load_NIST_data
    import matplotlib.pyplot as plt

    species = "Yb II"
    I = 0.5
    num_levels = 100
    B = Q_(5.0, 'G')
    df = load_NIST_data(species)

    a = Atom.from_dataframe(df, species, I=I, num_levels=num_levels, B=B)

    a.apply_hf_csv('171Yb_Hyperfine.csv')
    # a = Atom.from_pickle('171Yb.atom')

    for l in list(a.levels.values()):
        print('MAIN:', l.name, l.level.to('THz'), l.hfA)
        for s in list(l.values()):
            print('    SUB:', s.term.term_name, s.shift.to('THz'))

    # a.to_pickle('171Yb')
    a.populate_transitions()
    # cooling = Transition(a.levelsModel.nodes['4f14.6s 2S1/2']['level'], a.levelsModel.nodes['4f14.6p 2P*1/2']['level'])
    #
    # a.add_transition(cooling)

    a.levels['4f13.(2F*).6s2 2F*7/2'].populate_transitions()
    #
    # print(a.levelsModel.edges)
    # print(a.hfModel.edges)
    # print(a.zModel.edges)

    a.generate_hf_csv(filename='171Yb_Hyperfine.csv')
    print('drawing')
    nx.draw(a.levelsModel)
    plt.show()
    print('drawn')
