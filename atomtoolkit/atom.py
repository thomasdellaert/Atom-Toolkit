import itertools
import pickle
import re
import warnings
from typing import List

import networkx as nx
import numpy as np
import pint
from tqdm import tqdm

from . import Q_, ureg
from .atom_helpers import LevelStructure, TransitionStructure
from .wigner import wigner3j, wigner6j

Hz = ureg.hertz
mu_B = 1.39962449361e6  # MHz/G


############################################
#                   Term                   #
############################################

class Term:
    """
    A Term contains all the good quantum numbers of an EnergyLevel, as well as any ancestor terms that can be
    extracted from the configuration. These ancestor terms can (sometimes? often?) be used to convert between
    different couplings, among other things.
    """

    def __init__(self,
                 conf: str, term: str, J: float or str,
                 F: float or str = None, mF: float or str = None,
                 percentage=100.0, quantum_nums=None):
        """
        # TODO: Term symbols with multiple leading percentages, for better calculation of g factors etc
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

        if quantum_nums is None:
            quantum_nums = self.get_quantum_nums()
        self.quantum_nums = quantum_nums
        self.lc, self.sc, self.lo, self.so, self.jc, self.jo, self.l, self.s, self.k = self.quantum_nums

    def __str__(self):
        return self.term_name

    def __repr__(self):
        return f'Term({self.name})'

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
         Looks for the following in self.term:
             ({Jc, Jo})<{J}
         Examples:
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
        """Returns a term as listed in the nist ASD csv dataframe generated by the IO module"""
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


############################################
#               EnergyLevel                #
############################################

class BaseLevel:
    """
    The basic form of an energy level. To be instantiated, an energy level class needs to have:
        - a way to know what atom it's in
        - a way to know what manifold it's in
        - a way to generate any sublevels it may have
        - a defined energy
        - a defined shift from its parent

    Classes that inherit from BaseLevel have dict-like access to their sublevels, meaning that you
    can directly index them using sublevel names
    """

    def __init__(self, term: Term, parent):
        """
        Initialize the level

        :param term: The term symbol that tells the level its quantum numbers
        :param parent: The 'parent' of the level. Can be another level, or an Atom
        """
        self.parent = parent
        self.term = term
        self.atom = self.get_atom()
        self.manifold = self.get_manifold()
        self.name = self.term.name
        self.fixed = False

        self._sublevels = dict()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'{type(self).__name__}(name = {self.name}, level={str(self.level)}, {len(self._sublevels)} sublevels)'

    def get_atom(self):
        """Should return the atom to which the level belongs"""
        raise NotImplementedError()

    def get_manifold(self):
        """Should return the fine structure manifold to which the level belongs"""
        raise NotImplementedError()

    def populate_sublevels(self):
        """Should populate the sublevels of the level"""
        raise NotImplementedError()

    @property
    def level_Hz(self):
        """Should appropriately calculate the energy of the state (in Hz)"""
        raise NotImplementedError()

    @property
    def level(self):
        return self.level_Hz * Hz

    @level.setter
    def level(self, value: pint.Quantity):
        self.level_Hz = value.to(Hz).magnitude

    @property
    def shift_Hz(self):
        """Should return the shift relative to the parent (in Hz)"""
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
    """

    def __init__(self, term: Term, level: pint.Quantity, lande: float = None, parent=None,
                 hfA=Q_(0.0, 'gigahertz'), hfB=Q_(0.0, 'gigahertz'), hfC=Q_(0.0, 'gigahertz')):
        """
        :param term: a Term object containing the level's quantum numbers
        :param level: the energy of the centroid of the level
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
            self.lande = self.compute_gJ()
        else:
            self.lande = lande

    def get_manifold(self):
        """A fine structure level *is* a manifold."""
        return self

    def get_atom(self):
        """A fine structure level is the top level in the hierarchy. Its parent should be an Atom"""
        if self.parent is not None:
            if not isinstance(self.parent, Atom):
                warnings.warn(f'The parent of {self.name} is not an Atom')
            return self.parent
        else:
            return None

    def populate_sublevels(self):
        """
        Populates the sublevels dict with the appropriate hyperfine sublevels, given the atom that the EnergyLevel is in
        """
        if isinstance(self.parent, Atom):
            for f in np.arange(abs(self.term.J - self.atom.I), self.term.J + self.atom.I + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=f, quantum_nums=self.term.quantum_nums)
                e = HFLevel(term=t, parent=self)
                self[f'F={f}'] = e

    @property
    def level_Hz(self):
        return self._level_Hz

    @level_Hz.setter
    def level_Hz(self, value: float):
        """EnergyLevel gets a setter for level_Hz, but everything else calculates a frequency relative to its parent"""
        self._level_Hz = value

    @property
    def shift_Hz(self):
        """The 'shift' of an EnergyLevel is relative to the ground state"""
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
        Computes the Lande g-value of a term. This is only an estimate,
        and many terms with intermediate coupling will be well off from the true value
        :return: gJ
        """
        from .transition_strengths import JJ_to_LS, JK_to_LS, LK_to_LS
        if self.term.coupling == 'LS':
            terms = np.array([[self.term.l], [self.term.s], [1.0]])
        elif self.term.coupling == 'JJ':
            terms = JJ_to_LS(self.term.J, self.term.jc, self.term.jo, self.term.lc, self.term.sc, self.term.lo,
                             self.term.so)
        elif self.term.coupling == 'JK':
            terms = JK_to_LS(self.term.J, self.term.jc, self.term.k, self.term.lc, self.term.sc, self.term.lo,
                             self.term.so)
        elif self.term.coupling == 'LK':
            terms = LK_to_LS(self.term.J, self.term.l, self.term.k, self.term.sc, self.term.so)
        else:
            return 0.0

        J = self.term.J
        ls, ss, percents = terms

        return sum(percents * (1 + 1.0023 * (J * (J + 1) + ss * (ss + 1) - ls * (ls + 1)) / (2 * J * (J + 1))))

    @classmethod
    def from_dataframe(cls, df, i=0):
        """Generate from the i-th row of an atom dataframe. These dataframes should be of the
        format generated by the IO module, and should contain at least the columns "Level (cm-1)"
        and "Lande"
        TODO: Make the dataframe columns adjustable
        :param: df, a pandas dataframe
        :param: i, the row of the dataframe to generate from
        """
        t = Term.from_dataframe(df, i)
        return EnergyLevel(t, df["Level (cm-1)"][i], lande=df["Lande"][i])


class HFLevel(BaseLevel):
    """
    An HFLevel represents a hyperfine sublevel of a fine structure energy level. It lives inside an
    EnergyLevel,and contains Zeeman-sublevel ZLevel objects.
    An HFLevel defines where it is based on its shift relative to its parent level
    """

    def __init__(self, term: Term, parent=None):
        """
        :param term: a Term object containing the level's quantum numbers
        :param parent: the atom that the level is contained in
        """
        super().__init__(term, parent)
        self.gF = self.compute_gF()

    def get_manifold(self):
        """The parent of a hyperfine level is a fine structure level"""
        return self.parent

    def get_atom(self):
        """Inhabits the same atom as its parent"""
        return self.parent.atom

    def populate_sublevels(self):
        """Populates the sublevels dict with the appropriate Zeeman sublevels"""
        if isinstance(self.parent, EnergyLevel):
            for mf in np.arange(-self.term.F, self.term.F + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=self.term.F, mF=mf, quantum_nums=self.term.quantum_nums)
                e = ZLevel(term=t, parent=self)
                self[f'mF={mf}'] = e

    def compute_hf_shift(self):
        """
        Computes the hyperfine shift of a itself given the EnergyLevel's hyperfine coefficients and its F quantum number
        Math from Hoffman Thesis (2014)
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
            FE2 = (3 * IdotJ**2 + 1.5 * IdotJ - I * (I + 1) * J * (J + 1)) / \
                  (2.0 * I * (2.0 * I - 1.0) * J * (2.0 * J - 1.0))

        if J <= 1 or I <= 1:
            FM3 = 0
        else:
            FM3 = (10 * IdotJ**3 + 20 * IdotJ**2 + 2 * IdotJ * (
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
        Computes the Lande g-value for the hyperfine level, given its parent's g-value
        :return: gF
        """
        F = self.term.F
        J = self.term.J
        I = self.atom.I
        if F != 0:
            return self.manifold.lande * (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
        return 0.0


class ZLevel(HFLevel):
    """
    A Zeeman sublevel (currently) is the lowest of the level hierarchy. It defines its position relative
    to a parent hyperfine sublevel, dependent on the atom's magnetic field. It has no sublevels.
    """

    def get_manifold(self):
        return self.parent.manifold

    def populate_sublevels(self):
        """A Zeeman sublevel will have no further sublevels."""
        pass

    @property
    def shift_Hz(self):
        """A zeeman sublevel is shifted from its parent by the magnetic field. """
        # TODO: Breit-Rabi or at least second-order shifts?
        return self.gF * self.term.mF * mu_B * self.atom.B_gauss


############################################
#               Transition                 #
############################################

class BaseTransition:
    """
    A transition contains information about the transition between two levels. When instantiated
    with a set frequency, the Atom will move one of the EnergyLevels in order to make the energy difference
    between them consistent with the transition. This allows you to accommodate isotope shifts/more
    precise spectroscopy, etc.
    """

    def __init__(self, E1: EnergyLevel, E2: EnergyLevel, freq=None, A: pint.Quantity = None, name=None, parent=None):
        """
        :param E1: EnergyLevel 1
        :param E2: EnergyLevel 2
        :param freq: a pint Quantity representing the transition frequency. (can be any quantity that pint can
                     convert to a frequency, such as a wavenumber or a wavelength)
        :param A: The Einstein A coefficient of the transition, corresponding to the natural linewidth
        :param name: the name of the transition
        """

        self.parent = parent
        self._subtransitions = dict()
        self.E_1, self.E_2 = E1, E2
        if self.E_2.level > self.E_1.level:
            self.E_upper = self.E_2
            self.E_lower = self.E_1
        else:
            self.E_upper = self.E_1
            self.E_lower = self.E_2
        self.model_name = (self.E_1.name, self.E_2.name)
        self.name = name
        if self.name is None:
            self.name = f'{self.E_1.name} → {self.E_2.name}'
        self._A = A
        self.A, self.rel_strength = self.compute_linewidth()
        self.allowed_types = self.transition_allowed()
        self.set_freq = freq
        self.populate_subtransitions()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'{type(self).__name__}({self.name}, freq={str(self.freq)}, A={str(self.A)}'

    @property
    def freq_Hz(self):
        """The frequency of a transition is the difference in energy between its levels"""
        return abs(self.E_1.level_Hz - self.E_2.level_Hz)

    @property
    def freq(self):
        return self.freq_Hz * Hz

    @property
    def wl(self):
        return self.freq.to(ureg.nanometer)

    def compute_linewidth(self):
        if self._A is None:
            return None, 1.0
        return float(self._A), 1.0

    def transition_allowed(self):
        """Should return list containing what types of transition are allowed in E1, M1, E2, ... order"""
        raise NotImplementedError

    def add_to_atom(self, atom):
        """Should define how the transition is stored in the Atom"""
        raise NotImplementedError

    def populate_subtransitions(self):
        """Should generate the appropriate transitions between the sublevels of the transition's EnergyLevels"""
        raise NotImplementedError

    def _compute_sublevel_pairs(self, attr: str):
        """Returns pairs of sublevels that are likely to have allowed transitions, given the type of transition"""
        if True not in self.allowed_types:
            return itertools.product(self.E_1.sublevels(), self.E_2.sublevels())
        max_delta = (2 if (self.allowed_types[2] and not self.allowed_types[1]) else 1)
        return ((l1, l2) for l1, l2 in itertools.product(self.E_1.sublevels(), self.E_2.sublevels())
                if abs(l1.term.__getattribute__(attr) - l2.term.__getattribute__(attr)) <= max_delta)

    def __len__(self):
        return len(self._subtransitions)

    def __getitem__(self, item):
        return self._subtransitions[item]

    def __setitem__(self, key, value):
        pass #TODO: How should this behave?

    def __delitem__(self, key):
        del self._subtransitions[key]

    def __iter__(self):
        return iter(self._subtransitions)

    def values(self):
        return self._subtransitions.values()

    def subtransitions(self):
        return self._subtransitions.values()

    def keys(self):
        return self._subtransitions.keys()


class Transition(BaseTransition):
    """
    A Transition connects two EnergyLevel objects
    """

    def transition_allowed(self):
        """Uses selection rules to figure out whether it's an E1, M1, or E2 transition"""
        J0, J1 = self.E_1.term.J, self.E_2.term.J
        p0, p1 = self.E_1.term.parity, self.E_2.term.parity
        if not isinstance(self.E_1, type(self.E_2)):
            return [False, False, False]
        # for now I'm assuming that E2 transitions within a manifold are impossible. Correct me if I'm wrong!
        return ((np.abs(J1 - J0) <= 1.0 and p0 != p1),
                (np.abs(J1 - J0) <= 1.0 and p0 == p1),
                (np.abs(J1 - J0) <= 2.0 and p0 == p1 and not (self.E_1 is self.E_2)))

    def add_to_atom(self, atom):
        """A Transition lives as an edge in the atom's levelsModel graph"""
        atom.levelsModel.add_edge(self.E_1.name, self.E_2.name, transition=self)
        for subtransition in self.subtransitions():
            subtransition.add_to_atom(atom)

    def populate_subtransitions(self):
        """Creates HFTransitions for every allowed pair of hyperfine sublevels in the transition's EnergyLevels"""
        pairs = self._compute_sublevel_pairs('F')
        for pair in pairs:
            t = HFTransition(pair[0], pair[1], parent=self)
            if np.any(np.array(t.allowed_types)):
                self[t.name] = t
            else:
                del t

    def set_frequency(self, freq):
        """Set the frequency of the transition. Useful for adding in things like isotope shifts"""
        if self.set_freq:  # unlock the upper level if the transition is already constrained
            if self.E_upper.fixed and self.E_lower.fixed:
                self.E_upper.fixed = False
        self.set_freq = freq
        self.E_1.atom.enforce(node_or_trans=self.model_name)


class HFTransition(BaseTransition):
    """
    An HFTransition connects two HFLevels. It scales its linewidth and strength relative
    to its parents based on the clebsch-gordan coefficients etc associated with itself
    """

    def compute_linewidth(self):
        """Compute the amplitude/linewidth of the transition relative to others in its multiplet by Clebsch-Gordan math"""
        if self.parent is None or self.parent.A is None:
            return None, 1.0
        I = self.E_1.atom.I
        J1, J2 = self.E_1.term.J, self.E_2.term.J
        F1, F2 = self.E_1.term.F, self.E_2.term.F
        if self.parent.allowed_types[0] or self.parent.allowed_types[1]:
            factor = ((2 * F1 + 1) * (2 * F2 + 1)) * wigner6j(J2, F2, I, F1, J1, 1)**2
        elif self.parent.allowed_types[2]:
            factor = ((2 * F1 + 1) * (2 * F2 + 1)) * wigner6j(J2, F2, I, F1, J1, 2)**2
        else:
            factor = 0.0
        return float(self.parent.A * factor), float(factor)

    def transition_allowed(self):
        """Use the spectator theorem to check whether it's allowed. """
        I = self.E_1.atom.I
        J0, J1 = self.E_1.term.J, self.E_2.term.J
        F0, F1 = self.E_1.term.F, self.E_2.term.F
        if self.parent:
            init = self.parent.allowed_types
        else:
            init = [True, True, True]
        ret = [False, False, False]
        if init[0]:
            ret[0] = wigner6j(J0, J1, 1, F1, F0, I) != 0.0
        if init[1]:
            ret[1] = wigner6j(J0, J1, 1, F1, F0, I) != 0.0
        if init[2]:
            ret[2] = wigner6j(J0, J1, 2, F1, F0, I) != 0.0
        return tuple(ret)

    def add_to_atom(self, atom):
        """An HFTransition lives as an edge in the atom's hfModel graph"""
        atom.hfModel.add_edge(self.E_1.name, self.E_2.name, transition=self)
        for subtransition in self.subtransitions():
            subtransition.add_to_atom(atom)

    def populate_subtransitions(self):
        """Creates ZTransitions for every allowed pair of Zeeman sublevels in the transition's HFLevels"""
        pairs = self._compute_sublevel_pairs('mF')
        for pair in pairs:
            t = ZTransition(pair[0], pair[1], parent=self)
            if np.any(np.array(t.allowed_types)):
                self[t.name] = t
            else:
                del t


class ZTransition(BaseTransition):
    """
     An ZTransition connects two ZLevels. It scales its linewidth and strength relative
     to its parents based on the clebsch-gordan coefficients etc associated with itself
     """

    def compute_linewidth(self):
        """Use the wigner-eckart theorem to figure out relative line strengths"""
        if self.parent is None or self.parent.A is None:
            return None, 1.0
        F1, F2 = self.E_1.term.F, self.E_2.term.F
        mF1, mF2 = self.E_1.term.mF, self.E_2.term.mF
        if self.parent.allowed_types[0] or self.parent.allowed_types[1]:
            factor = sum([wigner3j(F1, 1, F2, -mF1, q, mF2)**2 for q in [-1, 0, 1]])
        elif self.parent.allowed_types[2]:
            factor = sum([wigner3j(F1, 2, F1, -mF1, q, mF2)**2 for q in [-2, -1, 0, 1, 2]])
        else:
            factor = 0.0
        return float(self.parent.A * factor), float(factor)

    def transition_allowed(self):
        """Check whether allowed. Currently only uses a geometric average"""
        # TODO: Maybe add the ability to apply lasers and shit? This sounds like a lot of work
        F0, F1 = self.E_1.term.F, self.E_2.term.F
        mF0, mF1 = self.E_1.term.mF, self.E_2.term.mF
        init = self.parent.allowed_types
        ret = [False, False, False]
        if init[0]:
            ret[0] = sum([wigner3j(F1, 1, F0, -mF1, q, mF0) for q in [-1, 0, 1]]) != 0.0
        if init[1]:
            ret[1] = sum([wigner3j(F1, 1, F0, -mF1, q, mF0) for q in [-1, 0, 1]]) != 0.0
        if init[2]:
            ret[2] = sum([wigner3j(F1, 2, F0, -mF1, q, mF0) for q in [-2, -1, 0, 1, 2]]) != 0.0
        return ret

    def add_to_atom(self, atom):
        """A ZTransition lives as an edge in the atom's zModel"""
        atom.zModel.add_edge(self.E_1.name, self.E_2.name, transition=self)

    def populate_subtransitions(self):
        pass


############################################
#                   Atom                   #
############################################

class Atom:
    """
    An Atom is a container for spectroscopic and atomic physics data. It contains a connected hierarchy of three
    data structures, represented as graphs:
        - The levelsModel, which has EnergyLevel instances as nodes, and Transition instances as edges
        - The hfModel, which has HFLevel instances as nodes, and HFTransition instances as edges
        - The zModel, which has ZLevel instances as nodes, and ZTransition instances as edges
    The Atom is responsible for enforcing physical constraints on these structures. For instance, if the frequency
    of a Transition is changed, the Atom will iterate through its levelsModel graph, making sure that the energy
    differences between levels are consistent with transition frequencies. It also contains parameters that are
    global to all levels, such as the magnetic field 'B' and the nuclear spin quantum number 'I'.

    The Atom's internal graphs are wrapped in a dict-like interface, in the form of the following attributes:
        levels, hflevels, zlevels
        transitions, hftransitions, ztransitions

    The Atom contains some functions for computing physically useful quantities involving multiple energy levels or
    transitions, such as branching ratios and state lifetimes

    Lastly, the Atom contains a lot of ways to give it data, though the preferred method is via a dataframe
    generated by the IO module. This may get move to the atom-builder module in the future
    """

    def __init__(self, name: str, I: float = 0.0, B=Q_(0.0, 'G'),
                 levels: List[EnergyLevel] = None, transitions: List[Transition] = None):
        """
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

        self.levels = LevelStructure(self, self.levelsModel)
        self.hflevels = LevelStructure(self, self.hfModel)
        self.zlevels = LevelStructure(self, self.zModel)

        self.transitions = TransitionStructure(self, self.levelsModel)
        self.hftransitions = TransitionStructure(self, self.hfModel)
        self.ztransitions = TransitionStructure(self, self.zModel)

        if levels is not None:
            for level in levels:
                self.add_level(level)
        if transitions is not None:
            for transition in transitions:
                self.add_transition(transition)

    def __str__(self):
        return self.name

    def __repr__(self):
        if len(self.levels) <= 100:
            l = str(list(self.levels.values()))
        else:
            l = str(list(self.levels[:5])) + "..." + str(list(self.levels[-5:]))

        return f'Atom(name={self.name}, I={self.I}, levels={l})'

    def add_level(self, level: EnergyLevel, key=None):
        """
        Adds a level to the internal graph model. When the level is added, it calculates its sublevels, and the
        sublevels are then also added
        TODO: accept a list of levels?
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

    def add_transition(self, transition: Transition):
        """
        Add a transition between two EnergyLevels in the atom. If either of the referenced levels aren't in the atom
        yet, they will be added.
        TODO: accept a list of Transitions?
        :param transition: the Transition to be added
        :return:
        """

        transition.add_to_atom(self)
        if transition.set_freq is not None:
            self.enforce()

    @property
    def B(self):
        return self.B_gauss * ureg.gauss

    @B.setter
    def B(self, value: pint.Quantity):
        self.B_gauss = value.to(ureg.gauss).magnitude

    def save(self, filename: str):
        """
        Pickles the Atom to a .atom file for loading later
        :param filename: the filename
        """
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
    def load(cls, filename):
        """
        Loads a pickled atom file.
        :param filename:
        :return:
        """
        file = open(filename, "rb")
        p = pickle.load(file)
        file.close()
        if not isinstance(p, Atom):
            raise IOError("The indicated file does not contain an Atom object")
        return p

    @classmethod
    def from_dataframe(cls, df, name, I=0.0, num_levels=None, B=Q_(0.0, 'G'), **kwargs):
        """
        Generate the atom from a dataframe, formatted like the output from the IO module
        :param df: the dataframe to read
        :param name: the name of the atom
        :param I: the nuclear spin quantum number of the atom
        :param num_levels: the number of levels to load from the dataframe
        :param B: the magnetic field
        :param kwargs: None
        :return: an instantiated atom
        """
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

    def populate_transitions(self, allowed=(True, True, True), **kwargs):
        """
        Iterate through every pair of levels in the atom, checking whether a given transition is 'allowed'
        and adding it if it is. Since this involves calculating Clebsch-Gordan coefficients for every possible
        pair of levels, it's slow and scales poorly with atom size. When possible, give a dataframe of transitions.

        :param allowed: a tuple of booleans:  ([E1],[M1],[E2])
        :param kwargs: none
        """
        max_to_try = 20

        allowed_deltas = [0]
        # since the ordering of the list is potentially E1, M1, E2, M2, etc... and the allowed deltas are 1, 1, 2, 2, etc
        for i in range(len(allowed)):
            if allowed[2 * i] or allowed[2 * i + 1]:
                allowed_deltas.append(i + 1)

        # The keys of js are J values, and the values are lists of levels with that J value
        js = {J: [lvl for lvl in self.levels.values() if lvl.term.J == J]
              for J in np.arange(0, max_to_try + 1, 0.5)}
        for delta_j in allowed_deltas:  # range(int(len(allowed)+2/2)):
            set0 = [js[j] for j in list(js.keys())]
            # don't include the last several j values to avoid IndexErrors
            set1 = [js[j + delta_j] for j in list(js.keys())[:int(-(delta_j * 2 + 1))]]
            j_pairs = zip(set0, set1)
            level_pairs = tqdm(list(itertools.chain.from_iterable([itertools.product(j1, j2) for j1, j2 in j_pairs])))
            for pair in level_pairs:
                if self.transitions[(pair[0].name, pair[1].name)] is not None:
                    t = Transition(pair[0], pair[1])
                    if not np.any(np.array(t.allowed_types) & np.array(allowed)):
                        del t
                    else:
                        level_pairs.set_description(f'processing ΔJ={delta_j:3} transition {t.name:93}')
                        self.add_transition(t)

    def populate_internal_transitions(self):
        """
        Iterates over each energy level and populates the internal M1 transitions for each one
        """
        levels = tqdm(list(self.levels.values()))
        for level in levels:
            t = Transition(level, level)
            t.add_to_atom(self)
            levels.set_description(f'adding internal transitions to {level.name:91}')

    def populate_transitions_df(self, df, **kwargs):
        """
        # TODO: move this outside the atom class?
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
                self.add_transition(t)
            except KeyError:
                pass

    def linked_levels(self, level):
        """
        returns a set of levels connected to the current level and the transitions connecting them
        :return: dict(Level: transition)
        """
        adjacent = self.levelsModel.adj[level]
        return {k: t['transition'] for k, t in adjacent.items()}

    def state_lifetime(self, level):
        ts = self.linked_levels(level).values()
        total_A = sum((t for t in ts if t.E_upper is level))
        return 1 / (total_A / (2 * np.pi))

    def compute_branching_ratios(self, key):
        transitions = self.linked_levels(key)
        A_coeffs = {}
        for n, t in transitions.items():
            if t.E_upper.name == key:
                A_coeffs[n] = t.A
        try:
            totalAs = np.sum(list(A_coeffs.values()))
        except TypeError:
            # TODO: make a custom exception for this? Maybe a whole slew of exceptions to throw
            raise TypeError("At least one transition leading from the given level has no valid A coefficient")

        ratios = {k: t / totalAs for k, t in A_coeffs.items()}
        return ratios

    def enforce(self, node_or_trans=None):
        """Enforces consistency of the Atom's internal levelsModel. It does this by traversing all the
        fixed transitions, and moving the EnergyLevels to be consistent with the transition frequencies.
        TODO: this currently only warns on cycles. Eventually, uncertainty math could make cycles that
         are self-consistent within a given error.

         :param node_or_trans: the node or transition that called the enforce function. It is used to restrict
         the enforcement to the subgraph in which the change happened
        """
        # make a subgraph of the full model containing only the fixed edges
        set_edges = [(u, v) for u, v, e in self.levelsModel.edges(data=True) if e['transition'].set_freq is not None]
        set_graph = self.levelsModel.edge_subgraph(set_edges)
        subgraphs = (set_graph.subgraph(c) for c in nx.connected_components(set_graph))
        for sg in subgraphs:
            if node_or_trans:
                if not (node_or_trans in sg or node_or_trans in sg.edges()):
                    continue
            # find the lowest lying level in the subgraph
            nodes = sg.nodes(data='level')
            l, kl = None, None
            for k, n in nodes:
                if l is None:
                    l, kl = n, k
                elif n.level_Hz < l.level_Hz:
                    l, kl = n, k
            # perform a depth first search starting from the lowest energy node in the subgraph,
            # setting subsequent levels one at a time
            search = nx.dfs_edges(sg, source=kl)
            for edge in search:
                t = sg.edges()[edge]['transition']
                el, eu, freq = t.E_lower, t.E_upper, t.set_freq.to(Hz).magnitude
                if el.fixed and eu.fixed and freq != abs(el.level_Hz - eu.level_Hz):
                    warnings.warn('Constraint problem. Perhaps there is a loop of fixed transitions?')
                elif eu.fixed:
                    el.level_Hz = eu.level_Hz - freq
                else:
                    eu.level_Hz = el.level_Hz + freq
                eu.fixed = el.fixed = True
