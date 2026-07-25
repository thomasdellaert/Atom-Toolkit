from __future__ import annotations
from abc import ABC, abstractmethod
import itertools
import functools
import json
import pickle
import warnings
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
import pint
from tqdm import tqdm

from . import Q_, ureg, util, Hz, G, mu_B, mu_N
from .atom_helpers import LevelStructure, TransitionStructure
from .term import Term, MultiTerm
from .wigner import wigner3j, wigner6j

############################################
#                  Level                   #
############################################


def lazy_compute_sublevels(func: Callable) -> Callable:
    """
    Wraps any function that references the sublevels of the level. If the level hasn't
    populated its sublevels yet, it computes them before returning any data.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(self._sublevels) == 0:
            self._assign_models()
            self.populate_sublevels()
            for sublevel in list(self._sublevels.values()):
                self._submodel.add_node(sublevel.name, level=sublevel)
        return func(self, *args, **kwargs)
    return wrapper


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

    def __init__(self, term: Term or MultiTerm, parent: Atom or BaseLevel, alias: str = None):
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
        self._model = None
        self._submodel = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{type(self).__name__}(name = {self.name}, level={str(self.level)}, {len(self._sublevels)} sublevels)>'

    @abstractmethod
    def get_atom(self) -> Atom:  # pragma: no cover
        """Should return the atom to which the level belongs"""
        raise NotImplementedError()

    @abstractmethod
    def _assign_models(self):
        """
        Should assign the data structures to self._model and self._submodel
        self._model should refer to the LevelStructure that contains the level
        self._submodel should refer to the LevelStructure that contains the level's sublevels

        BaseLevel instances are implemented as nodes in the internal graphs of an Atom object.
        For instance
            an EnergyLevel's associated model is self.atom.levels.model
            an EnergyLevel's associated submodel is self.atom.levels.hfmodel

        """
        raise NotImplementedError()

    @abstractmethod
    def get_manifold(self) -> EnergyLevel:
        """Should return the fine structure manifold to which the level belongs"""
        raise NotImplementedError()

    @abstractmethod
    def populate_sublevels(self):
        """Should populate the sublevels of the level"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def level_Hz(self) -> float:
        """Should appropriately calculate the energy of the state (in Hz)"""
        raise NotImplementedError()

    @level_Hz.setter
    @abstractmethod
    def level_Hz(self, value: float):
        raise NotImplementedError()

    @property
    def level(self) -> pint.Quantity:
        """The absolute energy of the state (as a Quantity)"""
        return self.level_Hz * Hz

    @level.setter
    def level(self, value: pint.Quantity):
        self.level_Hz = value.to(Hz).magnitude

    @property
    @abstractmethod
    def shift_Hz(self) -> float:
        """Should return the shift relative to the parent (in Hz)"""
        raise NotImplementedError()

    @property
    def shift(self) -> pint.Quantity:
        """The level's shift relative to its parent (as a Quantity)"""
        return self.shift_Hz * Hz

    @property
    def alias(self) -> str:
        """
        An alias is a string that acts as a shortened key that can be used to access the level in an Atom
        For instance:
            TmII = Atom(. . .)
            TmII.levels['4f13(2F*<7/2>).6s<1/2> (7/2, 1/2)*4'].alias = 'ground_state'
        allows you to access the level using a simpler key:
            TmII.levels['ground_state']
        """
        return self._alias

    @alias.setter
    def alias(self, al: str):
        self._alias = al
        self.atom.levels.aliases[al] = self

    @property
    def model(self) -> nx.Graph:
        if not self._model:
            self._assign_models()
        return self._model

    @model.setter
    def model(self, val: nx.Graph):
        self._model = val

    @property
    def submodel(self) -> nx.Graph:
        if not self._submodel:
            self._assign_models()
        return self._submodel

    @submodel.setter
    def submodel(self, val: nx.Graph):
        self._submodel = val

    def add_to_atom(self, atom):
        self.atom = atom
        self._assign_models()
        self._model.add_node(self.name, level=self)
        if len(self._sublevels) != 0:
            for sublevel in self.sublevels():
                sublevel.add_to_atom(atom)

    # region dict-like methods
    @lazy_compute_sublevels
    def __len__(self) -> int:
        return len(self._sublevels)

    @lazy_compute_sublevels
    def __getitem__(self, key) -> BaseLevel:
        """Returns the sublevel associated with the key. For atoms with I=0, mJ is usually used instead of mF
        and F is not used, so this is supported"""
        if self.atom.I == 0 and 'mJ' in key:
            return self._sublevels[f'F={self.term.J_frac}'][key.replace('mJ', 'mF')]
        return self._sublevels[key]

    def __setitem__(self, key: str, level: BaseLevel):
        level.parent = self
        self._sublevels[key] = level

    def __delitem__(self, key: str):
        del self._sublevels[key]

    @lazy_compute_sublevels
    def __iter__(self) -> Iterator:
        return iter(self._sublevels)

    @lazy_compute_sublevels
    def values(self) -> ValuesView:
        return self._sublevels.values()

    @lazy_compute_sublevels
    def sublevels(self) -> List:
        return list(self._sublevels.values())

    @lazy_compute_sublevels
    def keys(self) -> KeysView:
        return self._sublevels.keys()

    @lazy_compute_sublevels
    def items(self) -> ItemsView:
        return self._sublevels.items()

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

    def __init__(self, term: MultiTerm or Term, level: pint.Quantity, lande: float = None,
                 parent: Atom = None, alias: str = None,
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
        try:
            if lande is None or np.isnan(lande):
                self.lande = self.compute_gJ()
            else:
                self.lande = lande
        except TypeError:
            self.lande = self.compute_gJ()
        # Nonlinear Zeeman flag and spline LUT (populated by precompute_zeeman)
        self.use_nonlinear_zeeman: bool = False
        self._zeeman_splines: Optional[dict] = None

    def get_manifold(self) -> EnergyLevel:
        """A fine structure level *is* a manifold."""
        return self

    def get_atom(self) -> Atom | None:
        """A fine structure level is the top level in the hierarchy. Its parent should be an Atom"""
        if self.parent is not None:
            if not isinstance(self.parent, Atom):
                warnings.warn(f'The parent of {self.name} is not an Atom')
            return self.parent

    def _assign_models(self):
        self._model = self.atom.levels.model
        self._submodel = self.atom.levels.hf_model

    def populate_sublevels(self):
        """
        Populates the sublevels dict with the appropriate hyperfine sublevels, given the atom that the EnergyLevel is in
        """
        if self.atom is not None:
            for f in np.arange(abs(self.term.J - self.atom.I), self.term.J + self.atom.I + 1):
                t = self.term.make_term_copy(F=f)
                e = HFLevel(term=t, parent=self)
                self[f'F={util.float_to_frac(f)}'] = e

    @property
    def level_Hz(self) -> float:
        return self._level_Hz

    @level_Hz.setter
    def level_Hz(self, value: float):
        """EnergyLevel gets a setter for level_Hz, but everything else calculates a frequency relative to its parent"""
        self._level_Hz = value

    @property
    def shift_Hz(self) -> float:
        """The 'shift' of an EnergyLevel is relative to the ground state"""
        return self.level_Hz

    # region HF coefficients

    @property
    def hfA(self) -> pint.Quantity:
        return self.hfA_Hz * Hz

    @hfA.setter
    def hfA(self, value: pint.Quantity):
        self.hfA_Hz = value.to(Hz).magnitude
        self._invalidate_hf_cache()

    @property
    def hfB(self) -> pint.Quantity:
        return self.hfB_Hz * Hz

    @hfB.setter
    def hfB(self, value: pint.Quantity):
        self.hfB_Hz = value.to(Hz).magnitude
        self._invalidate_hf_cache()

    @property
    def hfC(self) -> pint.Quantity:
        return self.hfC_Hz * Hz

    @hfC.setter
    def hfC(self, value: pint.Quantity):
        self.hfC_Hz = value.to(Hz).magnitude
        self._invalidate_hf_cache()

    def _invalidate_hf_cache(self):
        """Clear the cached HF shift on all already-computed HFLevel sublevels."""
        for hfl in self._sublevels.values():
            hfl._shift_cache = None

    # endregion

    # region Nonlinear Zeeman LUT

    def precompute_zeeman(self, B_min: float, B_max: float, n_points: int = 200) -> None:
        """Precompute a cubic-spline LUT for the nonlinear Zeeman energies.

        After calling this, :meth:`ZLevel.level_hz_nonlinear` (and
        :meth:`ZLevel.level_hz_at` when ``use_nonlinear_zeeman`` is True) will
        evaluate the spline instead of re-diagonalising, making B-sweep loops
        orders of magnitude faster.

        Parameters
        ----------
        B_min, B_max : float
            Field range in gauss.  Queries outside this range fall back to
            direct diagonalisation with a warning.
        n_points : int
            Number of knot points.  200 is typically sub-Hz accurate over
            a few hundred gauss; increase for very wide ranges or steep
            nonlinearity.
        """
        try:
            from scipy.interpolate import CubicSpline
        except ImportError as exc:
            raise ImportError(
                "scipy is required for precompute_zeeman. "
                "Install it with: pip install scipy"
            ) from exc
        from .zeeman import diagonalize_hfz_batch, _hf_shift_hz, _mF_blocks

        B_array = np.linspace(B_min, B_max, n_points)
        batch = diagonalize_hfz_batch(self, B_array)

        J = float(self.term.J)
        I = float(self.atom.I)

        self._zeeman_splines = {}
        for mF, eigenvalues_array in batch.items():
            # Determine F ordering by ascending B=0 HF energy (adiabatic labels)
            F_list = sorted(
                [float(f) for f in np.arange(abs(J - I), J + I + 1) if f >= abs(mF) - 1e-9],
                key=lambda f: _hf_shift_hz(J, I, f, self.hfA_Hz, self.hfB_Hz, self.hfC_Hz),
            )
            for idx, F in enumerate(F_list):
                spline = CubicSpline(B_array, eigenvalues_array[:, idx])
                self._zeeman_splines[(F, mF)] = (spline, (B_min, B_max))

    def clear_zeeman_cache(self) -> None:
        """Discard the precomputed spline LUT, reverting to live diagonalisation."""
        self._zeeman_splines = None

    # endregion

    def compute_gJ(self) -> float:
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
                warnings.warn(
                    f"Unknown coupling scheme {term.coupling!r} for level {self.name!r}; "
                    f"contribution to gJ set to 0."
                )
                continue

            J = term.J
            ls, ss, percents = terms

            total += pct/100.0 * sum(percents * (1 + 1.0023 * (J * (J + 1) + ss * (ss + 1) - ls * (ls + 1))
                                     / (2 * J * (J + 1))))
        return total

    @classmethod
    def from_dataframe(cls, df, i=0) -> EnergyLevel:
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

    def __init__(self, term: MultiTerm, parent: EnergyLevel = None):
        """
        :param term: a Term object containing the level's quantum numbers
        :param parent: the atom that the level is contained in
        """
        super().__init__(term, parent)
        self._shift_cache: Optional[float] = None
        self.gF = self.compute_gF()

    def get_manifold(self) -> EnergyLevel:
        """The parent of a hyperfine level is a fine structure level"""
        return self.parent

    def get_atom(self) -> Atom:
        """Inhabits the same atom as its parent"""
        return self.parent.atom

    def _assign_models(self):
        self._model = self.atom.levels.hf_model
        self._submodel = self.atom.levels.z_model

    def populate_sublevels(self):
        """Populates the sublevels dict with the appropriate Zeeman sublevels"""
        if isinstance(self.parent, EnergyLevel):
            for mf in np.arange(-self.term.F, self.term.F + 1):
                t = self.term.make_term_copy(mF=mf)
                e = ZLevel(term=t, parent=self)
                self[f'mF={util.float_to_frac(mf)}'] = e

    def compute_hf_shift(self) -> float:
        """
        Computes the hyperfine shift of itself given the EnergyLevel's hyperfine coefficients and its F quantum number
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
    def shift_Hz(self) -> float:
        if self._shift_cache is None:
            self._shift_cache = self.compute_hf_shift()
        return self._shift_cache

    @property
    def level_Hz(self) -> float:
        return self.parent.level_Hz + self.shift_Hz

    @property
    def level(self) -> pint.Quantity:
        """When asked, sublevels calculate their position relative to their parent level"""
        return (self.parent.level_Hz + self.shift_Hz) * Hz

    @level.setter
    def level(self, value: pint.Quantity):
        """
        When the level of a sublevel is changed (for example, by a transition being added), the sublevel tells its
        parent to move by the same amount instead of moving itself
        :param value: the new level of the hyperfine level
        """
        shift = value - self.level
        self.parent.level += shift

    def compute_gF(self) -> float:
        """
        Computes the Lande g-value for the hyperfine level.

        Includes both the electronic (gJ) and nuclear (gI) contributions:

            gF = gJ · [F(F+1)+J(J+1)-I(I+1)] / [2F(F+1)]
               - gI · (μN/μB) · [F(F+1)+I(I+1)-J(J+1)] / [2F(F+1)]

        The nuclear term is ~1/1836 × gI and is zero when atom.gI is unset (0.0).

        :return: gF (dimensionless; multiply by μB to get the moment in energy/field units)
        """
        F = self.term.F
        J = self.term.J
        I = self.atom.I
        if F == 0:
            return 0.0
        denom = 2 * F * (F + 1)
        electronic = self.manifold.lande * (F * (F + 1) + J * (J + 1) - I * (I + 1)) / denom
        gI = getattr(self.atom, 'gI', 0.0)
        nuclear = gI * (mu_N / mu_B).magnitude * (F * (F + 1) + I * (I + 1) - J * (J + 1)) / denom
        return electronic - nuclear


class ZLevel(HFLevel):
    """
    A Zeeman sublevel (currently) is the lowest of the level hierarchy. It defines its position relative
    to a parent hyperfine sublevel, dependent on the atom's magnetic field. It has no sublevels.
    """
    # CONSIDER: Should this inherit from BaseLevel instead?
    def __init__(self, term: MultiTerm, parent: HFLevel = None):
        super().__init__(term, parent)
        # FIXME: Originally I wanted this to accept an arbitrary lambda function etc to encompass any
        #  nonlinearity in the Zeeman shift. I couldn't get it to work with Pickle, so I dropped it for now
        self.quadratic_zeeman = 0.0

    def get_manifold(self) -> EnergyLevel:
        return self.parent.manifold

    def populate_sublevels(self):
        """A Zeeman sublevel will have no further sublevels."""
        return

    def _assign_models(self):
        self._model = self.atom.levels.z_model
        self._submodel = None

    def _lazy_compute_sublevels(self):
        return

    def shift_hz_at(self, B_gauss: float) -> float:
        """Perturbative Zeeman shift in Hz: gF·mF·μB·B + quadratic_zeeman·B²."""
        return (self.gF * self.term.mF * mu_B.to(Hz / G).magnitude * B_gauss
                + self.quadratic_zeeman * B_gauss ** 2)

    def level_hz_at(self, B_gauss: float) -> float:
        """Absolute energy in Hz at a given field strength.

        Dispatches to :meth:`level_hz_nonlinear` when the parent manifold has
        ``use_nonlinear_zeeman = True``, otherwise uses the perturbative formula.
        """
        if self.manifold.use_nonlinear_zeeman:
            if self.quadratic_zeeman != 0.0:
                warnings.warn(
                    f"{self.name}: quadratic_zeeman is set but use_nonlinear_zeeman=True "
                    f"— the exact diagonalisation supersedes it. "
                    f"Remove the quadratic_zeeman assignment to silence this warning.",
                    stacklevel=2,
                )
            return self.level_hz_nonlinear(B_gauss)
        return self.parent.level_Hz + self.shift_hz_at(B_gauss)

    def level_hz_nonlinear(self, B_gauss: float) -> float:
        """Exact nonlinear Zeeman energy via H_HF + H_Z diagonalisation (Hz).

        Uses the spline LUT if :meth:`EnergyLevel.precompute_zeeman` has been
        called on the parent manifold; otherwise falls back to cached
        matrix diagonalisation.

        Requires ``atom.gI`` to be set for the nuclear Zeeman contribution.
        """
        from .zeeman import level_hz_nonlinear as _nlz
        return _nlz(self, B_gauss)

    def shift_hz_nonlinear(self, B_gauss: float) -> float:
        """Nonlinear Zeeman shift only (absolute energy minus EnergyLevel centroid)."""
        return self.level_hz_nonlinear(B_gauss) - self.manifold.level_Hz

    def zeeman_sensitivity_hz_per_gauss(self, B_gauss: float = 0.0, delta: float = 1e-3) -> float:
        """First-order Zeeman sensitivity dE/dB in Hz/G at a given field.

        For linear Zeeman levels (``use_nonlinear_zeeman=False``), returns the exact
        first-order coefficient gF·mF·μB.

        For nonlinear Zeeman levels, evaluates a central finite-difference derivative
        of the exact H_HF + H_Z eigenenergy, making it straightforward to scan for
        magnetically-insensitive ('clock') states where the sensitivity crosses zero.

        Parameters
        ----------
        B_gauss : float
            Magnetic field in gauss at which to evaluate the sensitivity.
            Defaults to B = 0.
        delta : float
            Finite-difference step in gauss (default 1e-3 G).

        Returns
        -------
        float : dE/dB in Hz/G.
        """
        if self.manifold.use_nonlinear_zeeman:
            if B_gauss > delta:
                return (self.level_hz_at(B_gauss + delta) - self.level_hz_at(B_gauss - delta)) / (2 * delta)
            else:
                return (self.level_hz_at(B_gauss + delta) - self.level_hz_at(B_gauss)) / delta
        return self.gF * self.term.mF * float(mu_B.to(Hz / G).magnitude)

    @property
    def shift_Hz(self) -> float:
        """Zeeman shift at B=0 (always zero; use shift_hz_at(B) for finite fields)."""
        return 0.0


############################################
#               Transition                 #
############################################


class AllowedTypes(NamedTuple):
    """Which multipole orders are allowed for a transition: (E1, M1, E2)."""
    E1: bool = False
    M1: bool = False
    E2: bool = False


def lazy_compute_subtransitions(func):
    """
    Wraps any function that references the subtransitions of the transition. If the transition hasn't
    populated its subtransitions yet, it computes them before returning any data.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(self._subtransitions) == 0:
            assert len(self.E_1.sublevels()) != 0
            assert len(self.E_2.sublevels()) != 0
            self.populate_subtransitions()
            for transition in list(self._subtransitions.values()):
                transition.add_to_atom(self.atom)
        return func(self, *args, **kwargs)
    return wrapper


class BaseTransition(ABC):
    """
    A transition contains information about the transition between two levels. When instantiated
    with a set frequency, the Atom will move one of the EnergyLevels in order to make the energy difference
    between them consistent with the transition. This allows you to accommodate isotope shifts/more
    precise spectroscopy, etc.
    """

    def __init__(self, E1: BaseLevel, E2: BaseLevel,
                 freq: pint.Quantity = None, A: pint.Quantity = None,
                 name: str = None, parent: BaseTransition = None, alias: str = None):
        """
        :param E1: EnergyLevel 1
        :param E2: EnergyLevel 2
        :param freq: a pint Quantity representing the transition frequency. (can be any quantity that pint can
                     convert to a frequency, such as a wavenumber or a wavelength)
        :param A: The Einstein A coefficient of the transition, corresponding to the natural linewidth
        :param name: the name of the transition
        :param parent: the parent of the transition, the thing that contains it
        :param alias: an alias for the transition; registered when the transition is added to an atom
        """

        self.parent = parent
        self._subtransitions = dict()
        self._alias = alias
        self._model = None
        self._submodel = None
        # Energy Levels
        self.E_1, self.E_2 = E1, E2
        if self.E_2.level > self.E_1.level:
            self.E_upper = self.E_2
            self.E_lower = self.E_1
        else:
            self.E_upper = self.E_1
            self.E_lower = self.E_2
        # atom reference — set properly via add_to_atom; kept here for lookup only
        self.atom = self.E_1.atom
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

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{type(self).__name__}({self.name}, freq={str(self.freq)}, A={str(self.A)}>'

    @property
    def freq_Hz(self) -> float:
        """The frequency of a transition is the difference in energy between its levels"""
        return abs(self.E_1.level_Hz - self.E_2.level_Hz)

    @property
    def freq(self) -> pint.Quantity:
        return self.freq_Hz * Hz

    @freq.setter
    def freq(self, val: pint.Quantity):
        self.set_frequency(val)

    @property
    def wl(self) -> pint.Quantity:
        return self.freq.to(ureg.nanometer)

    @property
    def omega(self) -> pint.Quantity:
        return self.freq * 2 * np.pi

    @property
    def total_A(self) -> Optional[pint.Quantity]:
        """Total spontaneous emission rate from the upper level (Hz).

        For gross-level Transitions this equals ``self.A``.  For HFTransitions
        and ZTransitions it climbs the parent chain to the root Transition,
        returning the gross natural linewidth Γ that governs saturation and
        scattering.
        """
        t = self
        while t.parent is not None:
            t = t.parent
        return t.A

    @property
    def I_sat(self) -> pint.Quantity:
        """Saturation intensity for this transition (W/m²).

        For a gross-level Transition (``rel_strength = 1``) this is the standard
        cycling-transition saturation intensity:

            I_sat = π h ν³ Γ / (3 c²)

        For a subtransition (HFTransition, ZTransition) with angular-momentum
        coupling factor ``S = rel_strength < 1``, the saturation intensity is
        higher because the sublevel pair couples to the field with a reduced
        dipole matrix element d_sub = d_cycling · √S:

            I_sat_sub = π h ν³ Γ / (3 c² S) = I_sat_cycling / S

        Here Γ = ``total_A`` is the gross Einstein A coefficient (total natural
        linewidth), not the partial linewidth ``self.A`` of the subtransition.

        Raises
        ------
        ValueError : if the gross-level A coefficient is missing, or if
            ``rel_strength == 0`` (forbidden transition — I_sat is undefined).
        """
        A_total = self.total_A
        if A_total is None:
            raise ValueError(
                f"Transition {self.name!r} has no A coefficient — cannot compute I_sat."
            )
        if self.rel_strength == 0.0:
            raise ValueError(
                f"Transition {self.name!r} is forbidden (rel_strength=0) — I_sat is undefined."
            )
        h = ureg.planck_constant
        c = ureg.speed_of_light
        return (np.pi * h * self.freq ** 3 * A_total / (3 * c ** 2 * self.rel_strength)).to('W/m**2')

    def saturation_parameter(self, intensity: pint.Quantity) -> float:
        """On-resonance saturation parameter s = I / I_sat (dimensionless).

        For a subtransition, ``I_sat`` is the sublevel-specific saturation
        intensity (larger than the cycling-transition value by ``1/rel_strength``),
        so *s* already accounts for the reduced coupling strength.

        Parameters
        ----------
        intensity : pint Quantity
            Laser intensity (any unit convertible to W/m²).

        Returns
        -------
        float : s = I / I_sat.
        """
        return float((intensity / self.I_sat).to('').magnitude)

    def rabi_frequency(self, intensity: pint.Quantity) -> pint.Quantity:
        """On-resonance Rabi frequency Ω/(2π) for this transition (Hz).

        Derived from Ω = Γ √(s/2) where Γ = ``total_A`` is the gross natural
        linewidth and *s* = ``saturation_parameter(intensity)`` is the
        sublevel-specific saturation parameter.  For a gross-level Transition
        (``rel_strength=1``) this is the cycling Rabi frequency.  For a
        subtransition it is reduced by √(rel_strength), reflecting the weaker
        coupling.  The angular Rabi frequency for Hamiltonians is 2π × this
        value.

        Parameters
        ----------
        intensity : pint Quantity
            Laser intensity (any unit convertible to W/m²).

        Returns
        -------
        pint Quantity : Ω/(2π) in Hz.
        """
        A_total = self.total_A
        if A_total is None:
            raise ValueError(f"Transition {self.name!r} has no A coefficient.")
        s = self.saturation_parameter(intensity)
        return (A_total * np.sqrt(s / 2)).to(Hz)

    def scattering_rate(self, intensity: pint.Quantity,
                        detuning: pint.Quantity = None) -> pint.Quantity:
        """Photon scattering rate for a two-level model (photons/s).

        Uses the optical Bloch equation steady-state result:

            R_sc = (Γ/2) · s / (1 + s + (2Δ/Γ)²)

        where Γ = ``total_A`` is the gross natural linewidth, *s* =
        ``saturation_parameter(intensity)`` is the sublevel-specific saturation
        parameter, and Δ is the laser detuning.  At resonance the rate saturates
        to Γ/2 as I → ∞ regardless of which sublevel pair is driven; at large
        detuning it tends to (Γ/4) · s · (Γ/Δ)².

        Parameters
        ----------
        intensity : pint Quantity
            Laser intensity (any unit convertible to W/m²).
        detuning : pint Quantity, optional
            Laser detuning from resonance (any unit convertible to Hz).
            Defaults to on-resonance (Δ = 0).

        Returns
        -------
        pint Quantity : scattering rate in Hz (photons/s).
        """
        A_total = self.total_A
        if A_total is None:
            raise ValueError(f"Transition {self.name!r} has no A coefficient.")
        s = self.saturation_parameter(intensity)
        Gamma = A_total.to(Hz)
        x2 = 0.0 if detuning is None else float((2 * detuning.to(Hz) / Gamma).magnitude) ** 2
        return (Gamma / 2 * s / (1 + s + x2)).to(Hz)

    def compute_linewidth(self) -> Tuple[pint.Quantity or None, float]:
        """
        computes the linewidth of the transition, including angular momentum factors
        :return: A coefficient, angular momentum factor
        """
        if self._A is None:
            return None, 1.0
        return self._A.to(Hz), 1.0

    @abstractmethod
    def transition_allowed(self) -> AllowedTypes:
        """Should return AllowedTypes indicating which multipole orders are allowed (E1, M1, E2)."""
        raise NotImplementedError

    @abstractmethod
    def _assign_models(self):
        raise NotImplementedError

    @abstractmethod
    def populate_subtransitions(self):
        """Should generate the appropriate transitions between the sublevels of the transition's EnergyLevels"""
        raise NotImplementedError

    @abstractmethod
    def set_frequency(self, freq: pint.Quantity):
        raise NotImplementedError

    @property
    def alias(self) -> str:
        return self._alias

    @alias.setter
    def alias(self, al: str):
        self._alias = al
        self.atom.transitions.aliases[al] = self

    def add_to_atom(self, atom: Atom):  # pragma: no cover
        """Add the transition to the atom and the appropriate graphs within the atom"""
        self.atom = atom
        self._assign_models()
        self._model.add_edge(self.E_lower.name, self.E_upper.name, transition=self)
        if len(self._subtransitions) != 0:
            for subtransition in self._subtransitions:
                subtransition.add_to_atom(atom)

    def _compute_sublevel_pairs(self, attr: str) -> Iterable[Tuple[BaseLevel, BaseLevel]]:
        """
        Returns pairs of sublevels that are likely to have allowed transitions, given the type of transition
        :param attr: the Term attribute to use for selection rules. Can be anything in {L, J, F, mF, mJ}
        :return: An iterable of pairs of sublevels with possible allowed transitions
        """
        if True not in self.allowed_types:
            return itertools.product(self.E_1.sublevels(), self.E_2.sublevels())
        max_delta = (2 if self.allowed_types.E2 else 1)
        return ((l1, l2) for l1, l2 in itertools.product(self.E_1.sublevels(), self.E_2.sublevels())
                if abs(l1.term.__getattribute__(attr) - l2.term.__getattribute__(attr)) <= max_delta)

    # region: dict-like-methods

    @lazy_compute_subtransitions
    def __len__(self) -> int:
        return len(self._subtransitions)

    @lazy_compute_subtransitions
    def __getitem__(self, item: Tuple[str, str]) -> BaseTransition:
        try:
            return self._subtransitions[(item[1], item[0])]
        except KeyError:
            return self._subtransitions[item]

    def __setitem__(self, key: Tuple[str, str], value: BaseTransition):
        self._subtransitions[key] = value

    def __delitem__(self, key: Tuple[str, str]):
        del self._subtransitions[key]

    @lazy_compute_subtransitions
    def __iter__(self) -> Iterator:
        return iter(self._subtransitions)

    @lazy_compute_subtransitions
    def values(self) -> ValuesView:
        return self._subtransitions.values()

    @lazy_compute_subtransitions
    def subtransitions(self) -> List:
        return list(self._subtransitions.values())

    @lazy_compute_subtransitions
    def keys(self) -> KeysView:
        return self._subtransitions.keys()

    @lazy_compute_subtransitions
    def items(self) -> ItemsView:
        return self._subtransitions.items()

    # endregion


class Transition(BaseTransition):
    """
    A Transition connects two EnergyLevel objects
    """

    def transition_allowed(self) -> AllowedTypes:
        """Uses selection rules to figure out whether it's an E1, M1, or E2 transition"""
        J0, J1 = self.E_1.term.J, self.E_2.term.J
        p0, p1 = self.E_1.term.parity, self.E_2.term.parity
        if not isinstance(self.E_1, type(self.E_2)):
            return AllowedTypes()
        # for now, I'm assuming that E2 transitions within a manifold are impossible or at least strongly suppressed.
        # Correct me if I'm wrong!
        #
        # A photon of multipole order k carries k units of angular momentum, so J0, J1 and k
        # must satisfy the full triangle condition |J1 - J0| <= k <= J1 + J0 -- note the lower
        # bound as well as the upper one. Dropping the lower bound would wrongly allow
        # J=0 -> J=0 for every multipole (e.g. the alkaline-earth 1S0 -> 3P*0 clock
        # transitions, which are strictly forbidden and only become weakly allowed in odd
        # isotopes through hyperfine mixing), and J=0 -> J=1 for E2.
        return AllowedTypes(
            E1=(np.abs(J1 - J0) <= 1.0 <= J1 + J0 and p0 != p1),
            M1=(np.abs(J1 - J0) <= 1.0 <= J1 + J0 and p0 == p1),
            E2=(np.abs(J1 - J0) <= 2.0 <= J1 + J0 and p0 == p1 and not (self.E_1 is self.E_2)),
        )

    def _assign_models(self):
        """A Transition lives as an edge in the atom's levelsModel graph"""
        self._model = self.atom.transitions.model
        self._submodel = self.atom.transitions.hf_model

    def populate_subtransitions(self):
        """Creates HFTransitions for every allowed pair of hyperfine sublevels in the transition's EnergyLevels"""
        pairs = self._compute_sublevel_pairs('F')
        for a, b in pairs:
            t = HFTransition(a, b, parent=self)
            if any(t.allowed_types):
                self[(t.E_lower.term.short_name, t.E_upper.term.short_name)] = t
            else:
                del t

    def set_frequency(self, freq: pint.Quantity):
        """Set the frequency of the transition. Useful for adding in things like isotope shifts"""
        if self.set_freq:  # unlock the upper level if the transition is already constrained
            if self.E_upper.fixed and self.E_lower.fixed:
                self.E_upper.fixed = False
        self.set_freq = freq
        self.E_1.atom.enforce_consistency(node_or_trans=self.model_name)


class HFTransition(BaseTransition):
    """
    An HFTransition connects two HFLevels. It scales its linewidth and strength relative
    to its parents based on the clebsch-gordan coefficients etc. associated with itself
    """

    def compute_linewidth(self) -> Tuple[pint.Quantity or None, float]:
        """Compute the amplitude/linewidth of the transition relative to others in its multiplet by Clebsch-Gordan
        math """
        if self.parent is None or self.parent.A is None:
            return None, 1.0
        I = self.E_1.atom.I
        J1, J2 = self.E_1.term.J, self.E_2.term.J
        F1, F2 = self.E_1.term.F, self.E_2.term.F
        if self.parent.allowed_types.E1 or self.parent.allowed_types.M1:
            factor = ((2 * F1 + 1) * (2 * F2 + 1)) * wigner6j(J2, F2, I, F1, J1, 1) ** 2
        elif self.parent.allowed_types.E2:
            factor = ((2 * F1 + 1) * (2 * F2 + 1)) * wigner6j(J2, F2, I, F1, J1, 2) ** 2
        else:
            factor = 0.0
        return self.parent.A * factor, float(factor)

    def transition_allowed(self) -> AllowedTypes:
        """Use the spectator theorem to check whether it's allowed. """
        I = self.E_1.atom.I
        J0, J1 = self.E_1.term.J, self.E_2.term.J
        F0, F1 = self.E_1.term.F, self.E_2.term.F
        init = self.parent.allowed_types if self.parent is not None else AllowedTypes(True, True, True)
        w1 = wigner6j(J0, J1, 1, F1, F0, I) != 0.0
        w2 = wigner6j(J0, J1, 2, F1, F0, I) != 0.0
        return AllowedTypes(
            E1=w1 and init.E1,
            M1=w1 and init.M1,
            E2=w2 and init.E2,
        )

    def _assign_models(self):
        """An HFTransition lives as an edge in the atom's hfModel graph"""
        self._model = self.atom.transitions.hf_model
        self._submodel = self.atom.transitions.z_model

    def populate_subtransitions(self):
        """Creates ZTransitions for every allowed pair of Zeeman sublevels in the transition's HFLevels"""
        pairs = self._compute_sublevel_pairs('mF')
        for a, b in pairs:
            t = ZTransition(a, b, parent=self)
            if any(t.allowed_types):
                self[(t.E_lower.term.short_name, t.E_upper.term.short_name)] = t
            else:
                del t

    def set_frequency(self, freq: pint.Quantity):
        diff = freq - self.freq
        self.parent.set_frequency(self.parent.freq + diff)


class ZTransition(BaseTransition):
    """
    An ZTransition connects two ZLevels. It scales its linewidth and strength relative
    to its parents based on the clebsch-gordan coefficients etc. associated with itself
    """

    def compute_linewidth(self) -> Tuple[pint.Quantity or None, float]:
        """Use the wigner-eckart theorem to figure out relative line strengths"""
        if self.parent is None or self.parent.A is None:
            return None, 1.0
        F1, F2 = self.E_1.term.F, self.E_2.term.F
        mF1, mF2 = self.E_1.term.mF, self.E_2.term.mF
        if self.parent.allowed_types.E1 or self.parent.allowed_types.M1:
            factor = sum([wigner3j(F1, 1, F2, -mF1, q, mF2) ** 2 for q in [-1, 0, 1]])
        elif self.parent.allowed_types.E2:
            factor = sum([wigner3j(F1, 2, F2, -mF1, q, mF2) ** 2 for q in [-2, -1, 0, 1, 2]])
        else:
            factor = 0.0
        return self.parent.A * factor, float(factor)

    def transition_allowed(self) -> AllowedTypes:
        """Check whether allowed. Currently only uses a geometric average"""
        F0, F1 = self.E_1.term.F, self.E_2.term.F
        mF0, mF1 = self.E_1.term.mF, self.E_2.term.mF
        init = self.parent.allowed_types
        w1 = sum([wigner3j(F1, 1, F0, -mF1, q, mF0) for q in [-1, 0, 1]]) != 0.0
        w2 = sum([wigner3j(F1, 2, F0, -mF1, q, mF0) for q in [-2, -1, 0, 1, 2]]) != 0.0
        return AllowedTypes(
            E1=w1 and init.E1,
            M1=w1 and init.M1,
            E2=w2 and init.E2,
        )

    def _assign_models(self):
        """A ZTransition lives as an edge in the atom's zModel"""
        self._model = self.atom.transitions.z_model
        self._submodel = None

    def populate_subtransitions(self):
        """A ZTransition is the lowest-level transition"""
        pass

    def set_frequency(self, freq: pint.Quantity):
        diff = freq - self.freq
        self.parent.set_frequency(self.parent.freq + diff)


############################################
#                   Atom                   #
############################################

_ATOM_JSON_SCHEMA = 3   # bump when the to_dict format changes


def _migrate_atom_dict(d: dict, from_schema: int) -> dict:
    """
    Apply forward-compatibility migrations to a dict loaded from a JSON file
    whose ``_schema`` version is older than the current :data:`_ATOM_JSON_SCHEMA`.

    Add a new entry to ``_migrations`` whenever a breaking change is made to the
    ``to_dict`` format, and increment ``_ATOM_JSON_SCHEMA``.  Each migration
    function receives the dict at the *previous* schema version and must return
    it at the *next* schema version.
    """
    _migrations: dict = {
        # schema 1 stored B_gauss; schema 2 removes it (B moved to Environment)
        1: lambda d: {k: v for k, v in d.items() if k != 'B_gauss'},
        # schema 3 adds gI (nuclear g-factor) and per-level use_nonlinear_zeeman
        2: lambda d: {**d, 'gI': d.get('gI', 0.0)},
    }
    for version in range(from_schema, _ATOM_JSON_SCHEMA):
        d = _migrations[version](d)
    return d


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

    def __init__(self, name: str, I: float = 0.0,
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
        self.gI: float = 0.0  # nuclear g-factor (dimensionless, in nuclear magneton units)

        self._levelsModel = nx.Graph()
        self._hfModel = nx.Graph()
        self._zModel = nx.Graph()

        self.levels = LevelStructure(self, self._levelsModel, self._hfModel, self._zModel)
        self.transitions = TransitionStructure(self, self._levelsModel, self._hfModel, self._zModel)

        if levels is not None:
            for level in levels:
                self.add_level(level)
        if transitions is not None:
            for transition in transitions:
                self.add_transition(transition)

    def __repr__(self):
        return f'<Atom(name={self.name}, I={self.I}, {len(self.levels)} levels)>'

    def __str__(self):
        return self.name

    @property
    def hflevels(self) -> Dict[str, 'HFLevel']:
        """
        Flat dict of all HFLevel objects that have been lazily computed so far.

        Keys are the full level names (e.g. ``'4f14.6s 2S1/2 F=0'``).
        HF sublevels are only present for EnergyLevels whose sublevels have
        already been accessed; iterate ``atom.levels.values()`` and access
        ``level.sublevels()`` to force-populate all of them first.
        """
        return dict(nx.get_node_attributes(self._hfModel, 'level'))

    @property
    def zlevels(self) -> Dict[str, 'ZLevel']:
        """
        Flat dict of all ZLevel objects that have been lazily computed so far.

        Keys are the full level names (e.g. ``'4f14.6s 2S1/2 F=0 mF=0'``).
        """
        return dict(nx.get_node_attributes(self._zModel, 'level'))

    @property
    def hftransitions(self) -> Dict[Tuple[str, str], 'HFTransition']:
        """
        Flat dict of all HFTransition objects that have been lazily computed so far.
        Keys are ``(lower_name, upper_name)`` tuples of HFLevel names.
        """
        return dict(nx.get_edge_attributes(self._hfModel, 'transition'))

    @property
    def ztransitions(self) -> Dict[Tuple[str, str], 'ZTransition']:
        """
        Flat dict of all ZTransition objects that have been lazily computed so far.
        Keys are ``(lower_name, upper_name)`` tuples of ZLevel names.
        """
        return dict(nx.get_edge_attributes(self._zModel, 'transition'))

    def add_level(self, level: EnergyLevel, populate_sublevels: bool = False, alias=None):
        """
        Adds a level to the internal graph model. When the level is added, it calculates its sublevels, and the
        sublevels are then also added
        :param populate_sublevels: whether the atom should populate the sublevels upon addition. If False, the
        sublevels can still be accessed, but are populated lazily on-demand
        :param level: the EnergyLevel to be added
        :param alias: an alias for accessing the EnergyLevel
        :return: the level that was added (in case you'd like an easy reference to it)
        """
        # TODO: On-the-spot EnergyLevel creation?
        level.add_to_atom(self)
        if alias is not None:
            level.alias = alias
        if level.alias is not None:
            self.levels.aliases[level.alias] = level
        if populate_sublevels:
            level.populate_sublevels()
            for sublevel in list(level.values()):
                sublevel.add_to_atom(self)
        return level

    def add_transition(self, transition: Transition or Tuple, **kwargs):
        """
        Add a transition between two EnergyLevels in the atom. If either of the referenced levels aren't in the atom
        yet, they will be added.
        :param transition: the Transition to be added
        :return: the transition that was added (in case you'd like an easy reference to it)
        """
        # CONSIDER: Should you be able to add non-top-level transitions directly to an atom?
        if isinstance(transition, tuple):
            if transition in self.transitions.keys():
                warnings.warn(f'transition {transition} is already in {self.name}')
                transition = self.transitions[transition]
            elif isinstance(transition[0], str):
                transition = Transition(self.levels[transition[0]], self.levels[transition[1]], **kwargs)
            elif isinstance(transition[0], BaseLevel):
                transition = Transition(transition[0], transition[1], **kwargs)

        transition.add_to_atom(self)
        if transition.alias is not None:
            self.transitions.aliases[transition.alias] = transition
        if transition.set_freq is not None:
            self.enforce_consistency(node_or_trans=transition)

        return transition

    def save(self, filename: str):
        """
        Pickles the Atom to a ``.atom`` file.

        .. deprecated::
            Prefer :meth:`save_json` for anything that should survive changes to
            the Atom class.  Pickle files break silently whenever internal class
            attributes are renamed, added, or removed.

        :param filename: the filename (without extension, or with ``.atom``)
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
    def load(cls, filename: str) -> Atom:
        """
        Load an Atom from a file.

        Dispatches automatically based on extension:

        - ``.json`` → :meth:`load_json` (recommended)
        - ``.atom`` → pickle (legacy)

        :param filename: path to the file
        :return: a fully-initialised Atom
        """
        if filename.endswith('.json'):
            return cls.load_json(filename)
        with open(filename, "rb") as file:
            p = pickle.load(file)
            if not isinstance(p, Atom):
                raise IOError("The indicated file does not contain an Atom object")
            return p

    # ------------------------------------------------------------------
    # JSON serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Export the atom to a plain-Python dict that is directly JSON-serialisable.

        Only *source-of-truth* data is stored — EnergyLevel energies and quantum
        numbers, hyperfine coefficients, and Transition A coefficients.  Derived
        objects (HFLevel, ZLevel, HFTransition, ZTransition) are excluded and will
        be regenerated lazily as usual when the atom is reloaded.

        The ``"_schema"`` key records the format version so that future changes can
        be migrated automatically via :func:`_migrate_atom_dict`.
        """
        levels = []
        for lvl in self.levels.values():
            # Serialise the MultiTerm as an ordered list of Term dicts
            if isinstance(lvl.term, MultiTerm):
                term_list = [
                    {"conf": t.conf, "term": t.term,
                     "J": t.J_frac, "percentage": t.percentage}
                    for t in lvl.term.terms
                ]
            else:
                term_list = [
                    {"conf": lvl.term.conf, "term": lvl.term.term,
                     "J": lvl.term.J_frac, "percentage": lvl.term.percentage}
                ]

            # Collect non-default quadratic Zeeman values from already-populated
            # ZLevels without triggering lazy computation on anything new.
            quad_z = {}
            for hf_key, hfl in lvl._sublevels.items():
                for z_key, zl in hfl._sublevels.items():
                    if zl.quadratic_zeeman != 0.0:
                        quad_z[f"{hf_key} {z_key}"] = zl.quadratic_zeeman

            levels.append({
                "terms":                 term_list,
                "level_Hz":              lvl.level_Hz,
                "hfA_Hz":                lvl.hfA_Hz,
                "hfB_Hz":                lvl.hfB_Hz,
                "hfC_Hz":                lvl.hfC_Hz,
                "lande":                 lvl.lande,
                "alias":                 lvl._alias,
                "quadratic_zeeman":      quad_z,
                "use_nonlinear_zeeman":  lvl.use_nonlinear_zeeman,
            })

        transitions = []
        for (lower_name, upper_name), t in self.transitions.items():
            if lower_name == upper_name:   # skip internal M1 self-transitions
                continue
            transitions.append({
                "lower":  lower_name,
                "upper":  upper_name,
                "A_Hz":   t.A.to('Hz').magnitude if t.A is not None else None,
                "alias":  t._alias,
            })

        return {
            "_schema":     _ATOM_JSON_SCHEMA,
            "name":        self.name,
            "I":           self.I,
            "gI":          self.gI,
            "levels":      levels,
            "transitions": transitions,
        }

    def save_json(self, filename: str):
        """
        Save the atom to a human-readable JSON file.

        Prefer this over :meth:`save` (pickle) for anything that should survive
        code changes.  The file can be reloaded with :meth:`load_json` or
        :meth:`load` (extension is detected automatically).

        :param filename: output path; ``.json`` is appended if absent
        """
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> Atom:
        """
        Reconstruct an Atom from a plain-Python dict produced by :meth:`to_dict`.

        Migration hooks (``_migrate_atom_dict``) are applied automatically when
        the stored schema version is older than the current one.

        :param d: the dict to reconstruct from
        :return: a fully-initialised Atom
        """
        schema = d.get('_schema', 1)
        d = _migrate_atom_dict(d, schema)

        a = cls(d['name'], I=d['I'])
        a.gI = d.get('gI', 0.0)

        # --- levels -------------------------------------------------------
        aliases_to_set = {}   # level_name → alias  (applied after add_level)
        for lvl_data in d['levels']:
            terms = [
                Term(t['conf'], t['term'], t['J'], percentage=t['percentage'])
                for t in lvl_data['terms']
            ]
            mt = MultiTerm(*terms) if len(terms) > 1 else terms[0]

            e = EnergyLevel(
                mt,
                level=lvl_data['level_Hz'] * Hz,
                lande=lvl_data.get('lande'),
                hfA=lvl_data['hfA_Hz'] * Hz,
                hfB=lvl_data['hfB_Hz'] * Hz,
                hfC=lvl_data['hfC_Hz'] * Hz,
            )
            e.use_nonlinear_zeeman = lvl_data.get('use_nonlinear_zeeman', False)
            a.add_level(e)

            if lvl_data.get('alias'):
                aliases_to_set[e.name] = lvl_data['alias']

        for level_name, alias in aliases_to_set.items():
            a.levels[level_name].alias = alias

        # --- transitions --------------------------------------------------
        tr_aliases = {}   # (lower, upper) → alias
        for tr_data in d['transitions']:
            try:
                lower = a.levels[tr_data['lower']]
                upper = a.levels[tr_data['upper']]
            except KeyError:
                warnings.warn(
                    f"Skipping transition ({tr_data['lower']!r}, "
                    f"{tr_data['upper']!r}): level not found in atom"
                )
                continue
            A = Q_(tr_data['A_Hz'], 'Hz') if tr_data.get('A_Hz') is not None else None
            t = a.add_transition(Transition(lower, upper, A=A))
            if tr_data.get('alias'):
                tr_aliases[(tr_data['lower'], tr_data['upper'])] = tr_data['alias']

        a.populate_internal_transitions()

        for (ln, un), alias in tr_aliases.items():
            a.transitions[(ln, un)].alias = alias

        # --- quadratic Zeeman (force-populate the affected sublevels) -----
        for lvl_data in d['levels']:
            if not lvl_data.get('quadratic_zeeman'):
                continue
            level_name = a.levels[  # derive name from first term + J
                f"{lvl_data['terms'][0]['conf']} "
                f"{lvl_data['terms'][0]['term']}{lvl_data['terms'][0]['J']}"
            ].name
            for sublevel_key, qz_val in lvl_data['quadratic_zeeman'].items():
                hf_key, z_key = sublevel_key.rsplit(' ', maxsplit=1)
                a.levels[level_name][hf_key][z_key].quadratic_zeeman = qz_val

        return a

    @classmethod
    def load_json(cls, filename: str) -> Atom:
        """
        Load an atom from a JSON file produced by :meth:`save_json`.

        :param filename: path to the ``.json`` file
        :return: a fully-initialised Atom
        """
        with open(filename, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str,
                       I: float = 0.0, num_levels: int = None, **_) -> Atom:
        """
        Generate the atom from a dataframe, formatted like the output from the IO module
        :param df: the dataframe to read
        :param name: the name of the atom
        :param I: the nuclear spin quantum number of the atom
        :param num_levels: the number of levels to load from the dataframe
        :return: an instantiated atom
        """
        if num_levels is None:
            num_levels = len(df)
        a = Atom(name, I=I)
        rows = tqdm(range(num_levels))
        skipped = []
        for i in rows:
            try:
                e = EnergyLevel.from_dataframe(df, i)
                if e.term.J is None:
                    # NIST lumps some high-l Rydberg multiplets together with no J at all
                    # (Mg I '3s.7i I', for instance). Without J there are no selection
                    # rules, sublevels or g-factors to be had, so the level is unusable.
                    raise ValueError('no J value given')
                rows.set_description(f'adding level {e.name:109}')
                a.add_level(e)
            except KeyError:
                pass
            except (ValueError, TypeError, IndexError) as exc:
                # A term symbol the parser doesn't handle (deeply nested JJ/JK couplings
                # in heavy elements, say) shouldn't take the whole species down with it.
                # Skip the level and report it, so the rest of the atom is still usable.
                try:
                    label = f'{df["Configuration"][i]} {df["Term"][i]} J={df["J"][i]}'
                except KeyError:  # pragma: no cover - malformed dataframe
                    label = f'row {i}'
                skipped.append(f'{label} ({type(exc).__name__}: {exc})')
        if skipped:
            warnings.warn(
                f'{name}: skipped {len(skipped)} unusable level(s):\n  ' + '\n  '.join(skipped),
                stacklevel=2,
            )
        return a

    def populate_transitions(self, allowed: Tuple[bool, ...] = (True, False, False), **_):
        """
        Iterate through every pair of levels in the atom, checking whether a given transition is 'allowed'
        and adding it if it is. Since this involves calculating Clebsch-Gordan coefficients for every possible
        pair of levels, it's slow and scales poorly with atom size. When possible, give a dataframe of transitions.

        :param allowed: a tuple of booleans:  ([E1],[M1],[E2])
        """
        max_to_try = 20
        allowed_deltas = []
        # since the ordering of the list is potentially E1, M1, E2, M2, etc. and the allowed deltas are 1, 1, 2, 2, etc.
        for i in range(len(allowed)):
            if allowed[i // 2] or allowed[i // 2 + 1]:
                allowed_deltas.append(i)

        # The keys of js are J values, and the values are lists of levels with that J value
        js = {J: [lvl for lvl in self.levels.values() if lvl.term.J == J]
              for J in np.arange(0, max_to_try + 1, 0.5)}
        for delta_j in allowed_deltas:
            set0 = [js[j] for j in list(js.keys())]
            # don't include the last several j values to avoid IndexErrors
            set1 = [js[j + delta_j] for j in list(js.keys())[:int(-(delta_j * 2 + 1))]]
            j_pairs = zip(set0, set1)
            level_pairs = tqdm(list(itertools.chain.from_iterable([itertools.product(j1, j2) for j1, j2 in j_pairs])))
            for pair in level_pairs:
                if self.transitions[(pair[0].name, pair[1].name)] is None:
                    t = Transition(pair[0], pair[1])
                    if any(ta and wa for ta, wa in zip(t.allowed_types, allowed)):
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

    def populate_transitions_df(self, df: pd.DataFrame, **_):
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

    def linked_levels(self, key: str) -> Dict[BaseLevel: BaseTransition]:
        """
        returns a dict of levels connected to the current level and the transitions connecting them
        :return: dict(Level: transition)
        """
        level = self.levels[key]
        if not isinstance(level, EnergyLevel):
            for k, trans in self.linked_levels(level.parent.name).items():
                trans.subtransitions()  # triggers lazy registration in the graph
        adjacent = level.model.adj[key]
        return {self.levels[k]: t['transition'] for k, t in adjacent.items() if k != key}

    def state_lifetime(self, key: str) -> pint.Quantity:
        """
        Compute the lifetime of the level based on the transitions leading from it
        :param key: the key of the level to compute
        :return: the state lifetime in seconds
        """
        ts = self.linked_levels(key).values()
        total_A = sum((t.A for t in ts if t.E_upper is self.levels[key]))
        if total_A == 0:
            return np.inf * ureg.s
        return (1 / (total_A / (2 * np.pi))).to(ureg.s)

    def compute_branching_ratios(self, key: str) -> Dict[str: float]:
        """
        Compute the branching ratios from a level to other levels.
        Requires that all levels to be considered have assigned A coefficients.

        Note: this only considers direct (one-step) decays.  For cascade branching
        (e.g. excited → metastable → ground) see the TODO below.

        TODO: add compute_cascade_branching_ratios() that follows multi-step decay
        chains and returns effective ground-state population fractions.

        :param key: the key of the level to compute
        :return: a dict of level names and the branching ratios to those levels
        """
        transitions = self.linked_levels(key)
        A_coeffs = {}
        for n, t in transitions.items():
            if t.E_upper.name == key:
                A_coeffs[n] = t.A
        try:
            totalAs = np.sum([A.magnitude for A in list(A_coeffs.values())])
        except TypeError:
            # CONSIDER: make a custom exception for this? Maybe a whole slew of exceptions to throw
            raise ValueError(f"At least one transition leading from {key} has no valid A coefficient")

        ratios = {k.name: t.magnitude / totalAs for k, t in A_coeffs.items()}
        return ratios

    def transitions_near(self, center: pint.Quantity, tolerance: pint.Quantity) -> List[Transition]:
        """Find all transitions within *tolerance* of *center* wavelength or frequency.

        TODO: Not yet implemented.

        Parameters
        ----------
        center : pint Quantity
            Centre wavelength (nm, m, …) or frequency (Hz, THz, …) to search near.
        tolerance : pint Quantity
            Half-width of the search window in the same unit family.

        Returns
        -------
        list of Transition sorted by proximity to *center*.
        """
        raise NotImplementedError("transitions_near is not yet implemented.")

    def find_clock_transitions(
            self,
            B_range: Tuple[float, float] = (0.0, 100.0),
            sensitivity_threshold: float = 1e3,
    ) -> List[Tuple['ZLevel', 'ZLevel', float]]:
        """Find Zeeman sublevel pairs with near-zero first-order field sensitivity.

        A 'clock' pair (|a⟩, |b⟩) satisfies |d(E_a - E_b)/dB| < sensitivity_threshold
        at some field B inside *B_range*.  Requires ``use_nonlinear_zeeman=True`` on
        the relevant manifolds to capture nonlinear turning points.

        TODO: Not yet implemented.

        Parameters
        ----------
        B_range : (float, float)
            Search range in gauss (default 0–100 G).
        sensitivity_threshold : float
            Maximum differential sensitivity in Hz/G to consider clock-like.

        Returns
        -------
        list of (lower_ZLevel, upper_ZLevel, B_zero_crossing_gauss) tuples.
        """
        raise NotImplementedError("find_clock_transitions is not yet implemented.")

    def ac_stark_shift(self, level: EnergyLevel, laser_freq: pint.Quantity,
                       intensity: pint.Quantity) -> pint.Quantity:
        """Scalar AC Stark shift on *level* from an off-resonant laser (Hz).

        Computes ΔE = Σ_j ħΩ_j² / (2(ω - ω_j)) summed over all connected levels j,
        using the A coefficients and transition matrix elements stored in the atom.
        Requires A coefficients on all connected transitions.

        TODO: Not yet implemented.  Also missing: tensor and vector light shifts,
        magic-wavelength finder (where the differential shift between two levels
        vanishes), and dynamic polarizability as a function of laser wavelength.

        Parameters
        ----------
        level : EnergyLevel
            The level whose energy shift is requested.
        laser_freq : pint Quantity
            Laser frequency (Hz, THz, nm, …).
        intensity : pint Quantity
            Laser intensity (W/m², mW/cm², …).

        Returns
        -------
        pint Quantity : AC Stark shift in Hz.
        """
        raise NotImplementedError("ac_stark_shift is not yet implemented.")

    def doppler_cooling_limit(self, transition_key: Tuple[str, str]) -> pint.Quantity:
        """Doppler cooling limit temperature T_D = ħΓ/(2k_B) for a transition (K).

        TODO: Not yet implemented.  Related quantities also missing:
        maximum scattering force F_max = ħkΓ/2, radiation pressure acceleration,
        and capture velocity for a given beam geometry.

        Parameters
        ----------
        transition_key : (lower_name, upper_name)

        Returns
        -------
        pint Quantity : Doppler limit temperature in K.
        """
        raise NotImplementedError("doppler_cooling_limit is not yet implemented.")

    def enforce_consistency(self, node_or_trans: BaseLevel or BaseTransition = None):
        """Enforces consistency of the Atom's internal levelsModel. It does this by traversing all the
        fixed transitions, and moving the EnergyLevels to be consistent with the transition frequencies.
        CONSIDER: this currently only warns on cycles. Eventually, uncertainty math could make cycles that
         are self-consistent within a given error.

         :param node_or_trans: the node or transition that called the enforce function. It is used to restrict
         the enforcement to the subgraph in which the change happened
        """
        # make a subgraph of the full model containing only the fixed edges
        set_edges = [(u, v) for u, v, e in self._levelsModel.edges(data=True) if e['transition'].set_freq is not None]
        set_graph = self._levelsModel.edge_subgraph(set_edges)
        connected_components = (set_graph.subgraph(c) for c in nx.connected_components(set_graph))
        for subgraph in connected_components:
            if node_or_trans:
                # check that the node/transition that triggered the enforcement call is contained in the subgraph
                if not (node_or_trans in subgraph or node_or_trans in subgraph.edges()):
                    continue
            # find the lowest lying level in the subgraph
            nodes = subgraph.nodes(data='level')
            lowest, key_lowest = None, None
            for key, node in nodes:
                if lowest is None:
                    lowest, key_lowest = node, key
                elif node.level_Hz < lowest.level_Hz:
                    lowest, key_lowest = node, key
            # perform a depth first search starting from the lowest energy node in the subgraph,
            # setting subsequent levels one at a time
            search = nx.dfs_edges(subgraph, source=key_lowest)
            for edge in search:
                t = subgraph.edges()[edge]['transition']
                e_l, e_u, freq = t.E_lower, t.E_upper, t.set_freq.to(Hz).magnitude
                if e_l.fixed and e_u.fixed and freq != abs(e_l.level_Hz - e_u.level_Hz):
                    warnings.warn(f'Constraint problem encountered when checking {edge}. Loop of fixed transitions?')
                elif e_u.fixed:
                    e_l.level_Hz = e_u.level_Hz - freq
                else:
                    e_u.level_Hz = e_l.level_Hz + freq
                e_u.fixed = e_l.fixed = True
