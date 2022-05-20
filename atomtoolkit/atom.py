from abc import ABC, abstractmethod
import itertools
import pickle
import warnings
from typing import List

import networkx as nx
import numpy as np
import pint
from tqdm import tqdm

from . import Q_, ureg, util, Hz
from .atom_helpers import LevelStructure, TransitionStructure
from .term import Term, MultiTerm
from .wigner import wigner3j, wigner6j

mu_B = 1.39962449361e6  # Hz/G


############################################
#                  Level                   #
############################################

class BaseLevel(ABC):
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

    def __init__(self, term: Term or MultiTerm, parent, alias=None):
        """
        Initialize the level

        :param term: The term symbol that tells the level its quantum numbers
        :param parent: The 'parent' of the level. Can be another level, or an Atom
        """
        self.parent = parent
        self._alias = alias
        if isinstance(term, Term):
            term = MultiTerm(term)
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

    @abstractmethod
    def get_atom(self):  # pragma: no cover
        """Should return the atom to which the level belongs"""
        raise NotImplementedError()

    @abstractmethod
    def get_manifold(self):  # pragma: no cover
        """Should return the fine structure manifold to which the level belongs"""
        raise NotImplementedError()

    @abstractmethod
    def populate_sublevels(self):  # pragma: no cover
        """Should populate the sublevels of the level"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def level_Hz(self):  # pragma: no cover
        """Should appropriately calculate the energy of the state (in Hz)"""
        raise NotImplementedError()

    @property
    def level(self):
        return self.level_Hz * Hz

    @level.setter
    def level(self, value: pint.Quantity):
        self.level_Hz = value.to(Hz).magnitude

    @property
    @abstractmethod
    def shift_Hz(self):  # pragma: no cover
        """Should return the shift relative to the parent (in Hz)"""
        raise NotImplementedError()

    @property
    def shift(self):
        return self.shift_Hz * Hz

    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, al):
        self._alias = al
        self.atom.levels[al] = self

    # region dict-like methods
    def __len__(self):
        self._lazy_compute_sublevels()
        return len(self._sublevels)

    def __getitem__(self, key):
        """Returns the sublevel associated with the key. For atoms with I=0, mJ is usually used instead of mF
        and F is not used, so this is supported"""
        self._lazy_compute_sublevels()
        if self.atom.I == 0 and 'mJ' in key:
            return self._sublevels[f'F={self.term.J_frac}'][key.replace('mJ', 'mF')]
        return self._sublevels[key]

    def __setitem__(self, key, level):
        level.parent = self
        self._sublevels[key] = level

    def __delitem__(self, key):
        del self._sublevels[key]

    def __iter__(self):
        self._lazy_compute_sublevels()
        return iter(self._sublevels)

    def values(self):
        self._lazy_compute_sublevels()
        return self._sublevels.values()

    def sublevels(self):
        self._lazy_compute_sublevels()
        return list(self._sublevels.values())

    def keys(self):
        self._lazy_compute_sublevels()
        return self._sublevels.keys()

    def items(self):
        self._lazy_compute_sublevels()
        return self._sublevels.items()

    # endregion

    def _lazy_compute_sublevels(self):
        """
        Called before the main text of any level that references the sublevels of the atom. If the level hasn't
        populated its sublevels yet, it computes them before returning any data.
        :return:
        """
        # FIXME: Right now this assumes and EnergyLevel, which is inappropriate for Baselevel. Make this more abstract.
        if len(self._sublevels) == 0:
            self.populate_sublevels()
            for sublevel in self.sublevels():
                self.atom.levels.hf_model.add_node(sublevel.name, level=sublevel)
                sublevel.populate_sublevels()
                for z_level in sublevel.sublevels():
                    self.atom.levels.z_model.add_node(z_level.name, level=z_level)


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

    def __init__(self, term: MultiTerm or Term, level: pint.Quantity, lande: float = None, parent=None, alias=None,
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
        super().__init__(term, parent, alias)
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
                t = self.term.make_term_copy(F=f)
                e = HFLevel(term=t, parent=self)
                self[f'F={util.float_to_frac(f)}'] = e

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
        total = 0
        for pct, term in self.term.terms_dict.items():
            if term.coupling == 'LS':
                terms = np.array([[self.term.l], [self.term.s], [1.0]])
            elif term.coupling == 'JJ':
                terms = JJ_to_LS(self.term.J, self.term.jc, self.term.jo, self.term.lc, self.term.sc, self.term.lo,
                                 self.term.so)
            elif term.coupling == 'JK':
                terms = JK_to_LS(self.term.J, self.term.jc, self.term.k, self.term.lc, self.term.sc, self.term.lo,
                                 self.term.so)
            elif term.coupling == 'LK':
                terms = LK_to_LS(self.term.J, self.term.l, self.term.k, self.term.sc, self.term.so)
            else:
                return 0.0

            J = term.J
            ls, ss, percents = terms

            total += pct * sum(percents * (1 + 1.0023 * (J * (J + 1) + ss * (ss + 1) - ls * (ls + 1)) / (2 * J * (J + 1))))
        return total

    @classmethod
    def from_dataframe(cls, df, i=0):
        """Generate from the i-th row of an atom dataframe. These dataframes should be of the
        format generated by the IO module, and should contain at least the columns "Level (cm-1)"
        and "Lande"
        :param: df, a pandas dataframe
        :param: i, the row of the dataframe to generate from
        """
        t = MultiTerm.from_dataframe(df, i)
        return EnergyLevel(t, df["Level (cm-1)"][i], lande=df["Lande"][i])


class HFLevel(BaseLevel):
    """
    An HFLevel represents a hyperfine sublevel of a fine structure energy level. It lives inside an
    EnergyLevel,and contains Zeeman-sublevel ZLevel objects.
    An HFLevel defines where it is based on its shift relative to its parent level
    """

    def __init__(self, term: MultiTerm, parent=None):
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
                t = self.term.make_term_copy(mF=mf)
                e = ZLevel(term=t, parent=self)
                self[f'mF={util.float_to_frac(mf)}'] = e

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

    def __init__(self, term: MultiTerm, parent=None):
        super().__init__(term, parent)
        # FIXME: Originally I wanted this to accept an arbitrary lambda function etc to encompass any
        #  nonlinearity in the Zeeman shift. I couldn't get it to work with Pickle, so I dropped it for now
        self.quadratic_zeeman = 0

    def get_manifold(self):
        return self.parent.manifold

    def populate_sublevels(self):
        """A Zeeman sublevel will have no further sublevels."""
        pass

    @property
    def shift_Hz(self):
        """A zeeman sublevel is shifted from its parent by the magnetic field. """
        return self.gF * self.term.mF * mu_B * self.atom.B_gauss \
            + self.quadratic_zeeman * self.atom.B_gauss ** 2

    # def nonlinear_zeeman(self, B):
    #     """
    #     Can be overridden to add any functional form to the Zeeman shift
    #     example: level.nonlinear_zeeman = lambda B: 155.305 * B ** 2
    #     """
    #     return 0


############################################
#               Transition                 #
############################################

class BaseTransition(ABC):
    """
    A transition contains information about the transition between two levels. When instantiated
    with a set frequency, the Atom will move one of the EnergyLevels in order to make the energy difference
    between them consistent with the transition. This allows you to accommodate isotope shifts/more
    precise spectroscopy, etc.
    """

    def __init__(self, E1: EnergyLevel, E2: EnergyLevel, freq=None, A: pint.Quantity = None, name=None, parent=None, alias=None):
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
        self._alias = alias
        # Energy Levels
        self.E_1, self.E_2 = E1, E2
        if self.E_2.level > self.E_1.level:
            self.E_upper = self.E_2
            self.E_lower = self.E_1
        else:
            self.E_upper = self.E_1
            self.E_lower = self.E_2
        # Check if in an atom
        self.atom = self.E_1.atom
        if self.atom is not None:
            self.add_to_atom(self.atom)
        # Names
        self.model_name = (self.E_1.name, self.E_2.name)
        self.name = name
        if self.name is None:
            self.name = f'{self.E_1.name} → {self.E_2.name}'
        # Physical parameters
        self._A = A
        self.A, self.rel_strength = self.compute_linewidth()
        self.allowed_types = self.transition_allowed()
        self.set_freq = freq
        if alias is not None:
            self.atom.transitions.aliases[alias] = self

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

    @freq.setter
    def freq(self, val):
        self.set_frequency(val)

    @property
    def wl(self):
        return self.freq.to(ureg.nanometer)

    def compute_linewidth(self):
        if self._A is None:
            return None, 1.0
        return self._A.to(Hz), 1.0

    @abstractmethod
    def transition_allowed(self):  # pragma: no cover
        """Should return list containing what types of transition are allowed in E1, M1, E2, ... order"""
        raise NotImplementedError

    @abstractmethod
    def add_to_atom(self, atom):  # pragma: no cover
        """Should define how the transition is stored in the Atom"""
        raise NotImplementedError

    @abstractmethod
    def populate_subtransitions(self):  # pragma: no cover
        """Should generate the appropriate transitions between the sublevels of the transition's EnergyLevels"""
        raise NotImplementedError

    @abstractmethod
    def set_frequency(self, freq):  # pragma: no cover
        raise NotImplementedError

    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, al):
        self._alias = al
        self.atom.transitions.aliases[al] = self

    def _compute_sublevel_pairs(self, attr: str):
        """Returns pairs of sublevels that are likely to have allowed transitions, given the type of transition"""
        if True not in self.allowed_types:
            return itertools.product(self.E_1.sublevels(), self.E_2.sublevels())
        max_delta = (2 if (self.allowed_types[2] and not self.allowed_types[1]) else 1)
        return ((l1, l2) for l1, l2 in itertools.product(self.E_1.sublevels(), self.E_2.sublevels())
                if abs(l1.term.__getattribute__(attr) - l2.term.__getattribute__(attr)) <= max_delta)

    def _lazy_compute_subtransitions(self):
        if len(self._subtransitions) == 0:
            assert len(self.E_1.sublevels()) != 0
            assert len(self.E_2.sublevels()) != 0
            self.populate_subtransitions()
            for transition in self.subtransitions():
                transition.add_to_atom(self.atom)

    def __len__(self):
        self._lazy_compute_subtransitions()
        return len(self._subtransitions)

    def __getitem__(self, item):
        self._lazy_compute_subtransitions()
        try:
            return self._subtransitions[(item[1], item[0])]
        except KeyError:
            return self._subtransitions[item]

    def __setitem__(self, key, value):
        # TODO: make this idiot-proof
        self._subtransitions[key] = value

    def __delitem__(self, key):
        del self._subtransitions[key]

    def __iter__(self):
        self._lazy_compute_subtransitions()
        return iter(self._subtransitions)

    def values(self):
        self._lazy_compute_subtransitions()
        return self._subtransitions.values()

    def subtransitions(self):
        self._lazy_compute_subtransitions()
        return self._subtransitions.values()

    def keys(self):
        self._lazy_compute_subtransitions()
        return self._subtransitions.keys()

    def items(self):
        self._lazy_compute_subtransitions()
        return self._subtransitions.items()


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
        self.atom = atom
        atom.levels.model.add_edge(self.E_lower.name, self.E_upper.name, transition=self)
        if len(self._subtransitions) != 0:
            for subtransition in self.subtransitions():
                subtransition.add_to_atom(atom)

    def populate_subtransitions(self):
        """Creates HFTransitions for every allowed pair of hyperfine sublevels in the transition's EnergyLevels"""
        pairs = self._compute_sublevel_pairs('F')
        for a, b in pairs:
            t = HFTransition(a, b, parent=self)
            if np.any(np.array(t.allowed_types)):
                self[(t.E_lower.term.short_name, t.E_upper.term.short_name)] = t
            else:
                del t

    def set_frequency(self, freq):
        """Set the frequency of the transition. Useful for adding in things like isotope shifts"""
        if self.set_freq:  # unlock the upper level if the transition is already constrained
            if self.E_upper.fixed and self.E_lower.fixed:
                self.E_upper.fixed = False
        self.set_freq = freq
        self.E_1.atom.enforce_consistency(node_or_trans=self.model_name)


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
            factor = ((2 * F1 + 1) * (2 * F2 + 1)) * wigner6j(J2, F2, I, F1, J1, 1) ** 2
        elif self.parent.allowed_types[2]:
            factor = ((2 * F1 + 1) * (2 * F2 + 1)) * wigner6j(J2, F2, I, F1, J1, 2) ** 2
        else:
            factor = 0.0
        return self.parent.A * factor, float(factor)

    def transition_allowed(self):
        """Use the spectator theorem to check whether it's allowed. """
        I = self.E_1.atom.I
        J0, J1 = self.E_1.term.J, self.E_2.term.J
        F0, F1 = self.E_1.term.F, self.E_2.term.F
        if self.parent is not None:
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
        self.atom = atom
        atom.levels.hf_model.add_edge(self.E_lower.name, self.E_lower.name, transition=self)
        if len(self._subtransitions) != 0:
            for subtransition in self.subtransitions():
                subtransition.add_to_atom(atom)

    def populate_subtransitions(self):
        """Creates ZTransitions for every allowed pair of Zeeman sublevels in the transition's HFLevels"""
        pairs = self._compute_sublevel_pairs('mF')
        for a, b in pairs:
            t = ZTransition(a, b, parent=self)
            if np.any(np.array(t.allowed_types)):
                self[(t.E_lower.term.short_name, t.E_upper.term.short_name)] = t
            else:
                del t

    def set_frequency(self, freq):
        diff = freq - self.freq
        self.parent.set_frequency(self.parent.freq + diff)


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
            factor = sum([wigner3j(F1, 1, F2, -mF1, q, mF2) ** 2 for q in [-1, 0, 1]])
        elif self.parent.allowed_types[2]:
            factor = sum([wigner3j(F1, 2, F1, -mF1, q, mF2) ** 2 for q in [-2, -1, 0, 1, 2]])
        else:
            factor = 0.0
        return self.parent.A * factor, float(factor)

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
        self.atom = atom
        atom.levels.z_model.add_edge(self.E_lower.name, self.E_upper.name, transition=self)

    def populate_subtransitions(self):
        pass

    def set_frequency(self, freq):
        diff = freq - self.freq
        self.parent.set_frequency(self.parent.freq + diff)


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
        self.I = util.frac_to_float(I)
        self.B_gauss = B.to(ureg.gauss).magnitude

        self._levelsModel = nx.Graph()
        self._hfModel = nx.Graph()
        self._zModel = nx.Graph()

        self.levels = LevelStructure(self, self._levelsModel, self._hfModel, self._zModel)
        # self.hflevels = LevelStructure(self, self.hfModel)
        # self.zlevels = LevelStructure(self, self.zModel)

        self.transitions = TransitionStructure(self, self._levelsModel, self._hfModel, self._zModel)
        # self.hftransitions = SubtransitionStructure(self, self.hfModel, self.levelsModel)
        # self.ztransitions = SubtransitionStructure(self, self.zModel, self.hfModel)

        if levels is not None:
            for level in levels:
                self.add_level(level)
        if transitions is not None:
            for transition in transitions:
                self.add_transition(transition)

    def __str__(self):
        return self.name

    def __repr__(self):
        if len(self.levels) <= 20:
            l = str(list(self.levels.values()))
        else:
            l = str(list(self.levels[:5])) + "..." + str(list(self.levels[-5:]))

        return f'Atom(name={self.name}, I={self.I}, levels={l})'

    def add_level(self, level: EnergyLevel, key=None, populate_sublevels=False):
        """
        Adds a level to the internal graph model. When the level is added, it calculates its sublevels, and the
        sublevels are then also added
        :param populate_sublevels: whether the atom should populate the sublevels upon addition. If False, the
        sublevels can still be accessed, but are populated lazily on-demand
        :param level: the EnergyLevel to be added
        :param key: A custom name, if desired. Otherwise defaults to the EnergyLevel's name
        :return:
        """
        if key is None:
            key = level.name
        if type(level) != EnergyLevel:
            level = level.manifold
            populate_sublevels = True
        level.atom = self
        level.parent = self
        self._levelsModel.add_node(key, level=level)
        if level.alias is not None:
            self.levels.aliases[level.alias] = level
        if populate_sublevels:
            level.populate_sublevels()
            for sublevel in list(level.values()):
                self._hfModel.add_node(sublevel.name, level=sublevel)
                sublevel.populate_sublevels()
                for z_level in list(sublevel.values()):
                    self._zModel.add_node(z_level.name, level=z_level)

    def add_transition(self, transition: Transition):
        """
        Add a transition between two EnergyLevels in the atom. If either of the referenced levels aren't in the atom
        yet, they will be added.
        :param transition: the Transition to be added
        :return:
        """
        # TODO: This should be able to take a tuple of EnergyLevels, or possibly even indices, and create the
        #  necessary transition on the spot
        transition.add_to_atom(self)
        if transition.alias is not None:
            self.transitions.aliases[transition.alias] = transition
        if transition.set_freq is not None:
            self.enforce_consistency()

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
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """
        Loads a pickled atom file.
        :param filename:
        :return:
        """
        with open(filename, "rb") as file:
            p = pickle.load(file)
            if not isinstance(p, Atom):
                raise IOError("The indicated file does not contain an Atom object")
            return p

    @classmethod
    def from_dataframe(cls, df, name, I=0.0, num_levels=None, B=Q_(0.0, 'G'), **_):
        """
        Generate the atom from a dataframe, formatted like the output from the IO module
        :param df: the dataframe to read
        :param name: the name of the atom
        :param I: the nuclear spin quantum number of the atom
        :param num_levels: the number of levels to load from the dataframe
        :param B: the magnetic field
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

    def populate_transitions(self, allowed=(True, True, True), **_):
        """
        Iterate through every pair of levels in the atom, checking whether a given transition is 'allowed'
        and adding it if it is. Since this involves calculating Clebsch-Gordan coefficients for every possible
        pair of levels, it's slow and scales poorly with atom size. When possible, give a dataframe of transitions.

        :param allowed: a tuple of booleans:  ([E1],[M1],[E2])
        """
        max_to_try = 20
        allowed_deltas = []
        # since the ordering of the list is potentially E1, M1, E2, M2, etc... and the allowed deltas are 1, 1, 2, 2, etc
        for i in range(len(allowed)):
            if allowed[i // 2] or allowed[i // 2 + 1]:
                allowed_deltas.append(i)

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
                if self.transitions[(pair[0].name, pair[1].name)] is None:
                    t = Transition(pair[0], pair[1])
                    if np.any(np.array(t.allowed_types) & np.array(allowed)):
                        level_pairs.set_description(f'processing ΔJ={delta_j:3} transition {t.name:93}')
                        self.add_transition(t)
                    else:
                        del t

    def populate_internal_transitions(self):
        """
        Iterates over each energy level and populates the internal M1 transitions for each one
        """
        levels = tqdm(list(self.levels.values()))
        for level in levels:
            t = Transition(level, level)
            t.add_to_atom(self)
            levels.set_description(f'adding internal transitions to {level.name:91}')

    def populate_transitions_df(self, df, **_):
        """
        # CONSIDER: move this outside the atom class?
        Load transitions into the Atom from a dataframe generated from the IO module's load_transition_data function
        """
        rows = tqdm(list(df.iterrows()))
        for _, row in rows:
            j_l = util.float_to_frac(float(row['j_l']))
            j_u = util.float_to_frac(float(row['j_u']))
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
        returns a dict of levels connected to the current level and the transitions connecting them
        :return: dict(Level: transition)
        """
        # TODO: hyperfine and zeeman should be supported
        adjacent = self._levelsModel.adj[level]
        return {k: t['transition'] for k, t in adjacent.items() if k != level}

    def state_lifetime(self, level):
        ts = self.linked_levels(level).values()
        total_A = sum((t.A for t in ts if t.E_upper is self.levels[level]))
        if total_A == 0:
            return np.inf * ureg.s
        return (1 / (total_A / (2 * np.pi))).to(ureg.s)

    def compute_branching_ratios(self, key):
        transitions = self.linked_levels(key)
        A_coeffs = {}
        for n, t in transitions.items():
            if t.E_upper.name == key:
                A_coeffs[n] = t.A
        try:
            totalAs = np.sum([A.magnitude for A in list(A_coeffs.values())])
        except TypeError:
            # TODO: make a custom exception for this? Maybe a whole slew of exceptions to throw
            raise TypeError(f"At least one transition leading from {key} has no valid A coefficient")

        ratios = {k: t.magnitude / totalAs for k, t in A_coeffs.items()}
        return ratios

    def enforce_consistency(self, node_or_trans=None):
        """Enforces consistency of the Atom's internal levelsModel. It does this by traversing all the
        fixed transitions, and moving the EnergyLevels to be consistent with the transition frequencies.
        TODO: this currently only warns on cycles. Eventually, uncertainty math could make cycles that
         are self-consistent within a given error.

         :param node_or_trans: the node or transition that called the enforce function. It is used to restrict
         the enforcement to the subgraph in which the change happened
        """
        # make a subgraph of the full model containing only the fixed edges
        set_edges = [(u, v) for u, v, e in self._levelsModel.edges(data=True) if e['transition'].set_freq is not None]
        set_graph = self._levelsModel.edge_subgraph(set_edges)
        ccs = (set_graph.subgraph(c) for c in nx.connected_components(set_graph))
        for sg in ccs:
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
                    warnings.warn(f'Constraint problem encountered when checking {edge}. Loop of fixed transitions?')
                elif eu.fixed:
                    el.level_Hz = eu.level_Hz - freq
                else:
                    eu.level_Hz = el.level_Hz + freq
                eu.fixed = el.fixed = True
