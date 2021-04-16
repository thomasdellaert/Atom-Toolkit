import pandas as pd
import numpy as np
import networkx as nx
import pickle
import pint
import pint_pandas
from indexedproperty import indexedproperty
# import json
import re
from itertools import combinations
from typing import List

ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)
pint_pandas.PintType.ureg = ureg
Q_ = ureg.Quantity
pd.options.display.max_rows = 999

c = pint.Context('spectroscopy')
c.add_transformation('[wavenumber]', '[frequency]',
                     lambda ureg, x: x * ureg.speed_of_light)
c.add_transformation('[frequency]', '[wavenumber]',
                     lambda ureg, x: x / ureg.speed_of_light)
ureg.enable_contexts('spectroscopy')


def load_NIST_data(species, term_ordered=True):
    df = pd.read_csv(
        'https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0' +
        '&spectrum=' + species.replace(' ', '+') +
        '&submit=Retrieve+Data' +
        '&units=0' +
        '&format=2' +
        '&output=0' +
        '&page_size=15' +
        '&multiplet_ordered=' + ('on' if term_ordered else '0') +
        '&conf_out=on' +
        '&term_out=on' +
        '&level_out=on' +
        '&unc_out=0' +
        '&j_out=on' +
        '&lande_out=on' +
        '&perc_out=on' +
        '&biblio=0' +
        '&temp=',
        index_col=False)
    # === strip the data of extraneous symbols ===
    df_clean = df.applymap(lambda x: x.strip(' ="?'))
    # === coerce types ===
    df_clean['Configuration'] = df_clean['Configuration'].astype('str')
    df_clean['Term'] = df_clean['Term'].astype('str')
    df_clean['Term'] = df_clean['Term'].apply(lambda x: re.sub(r'[a-z] ', '', x))
    df_clean['J'] = df_clean['J'].astype('str')
    df_clean['Level (cm-1)'] = pd.to_numeric(df_clean['Level (cm-1)'], errors='coerce')
    #    keep only the initial number of the leading percentage for now, replacing NaN with 100% I guess
    df_clean['Leading percentages'] = df_clean['Leading percentages'].apply(lambda x: re.sub(r' ?:.*', '', x))
    df_clean['Leading percentages'] = pd.to_numeric(df_clean['Leading percentages'], errors='coerce')
    df_clean['Leading percentages'] = df_clean['Leading percentages'].fillna(value=100.0)
    if 'Lande' not in df_clean.columns:
        df_clean['Lande'] = None
    df_clean['Lande'] = pd.to_numeric(df_clean['Lande'], errors='coerce')
    # drop rows that don't have a defined level
    df_clean = df_clean.dropna(subset=['Level (cm-1)'])
    # convert levels to pint Quantities
    df_clean['Level (cm-1)'] = df_clean['Level (cm-1)'].astype('pint[cm**-1]')
    df_clean['Level (Hz)'] = df_clean['Level (cm-1)'].pint.to('Hz')

    df_clean = df_clean[df_clean.J.str.contains("---") == False]  # happens at ionization thresholds
    df_clean = df_clean[df_clean.J.str.contains(",") == False]  # happens when J is unknown
    # reset the indices, since we may have dropped some rows
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean

class Term:
    def __init__(self,
                 conf: str, term: str, J: float or str,
                 F: float or str = None, mF: float or str = None,
                 percentage=100.0):
        """
        TODO: docstring
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
                 hfA=0.0, hfB=0.0, hfC=0.0):
        """
        TODO: docstring
        :param term:
        :param level:
        :param lande:
        :param parent:
        :param atom:
        :param hfA:
        :param hfB:
        :param hfC:
        """
        self.parent = parent
        self.atom = atom
        self.term = term
        self._level = level.to('Hz')
        self.shift = self._level
        self.name = self.term.name
        self.hfA, self.hfB, self.hfC = hfA, hfB, hfC
        self.level_constrained = False
        if lande is None:
            try:
                self.lande = self.compute_gJ()
            except NotImplementedError:
                self.lande = 0  # TODO: think about a placeholder value instead?
        else:
            self.lande = lande
        self._sublevels = {}
        self.populate_sublevels()

    def populate_sublevels(self):
        """
        Populates the sublevels dict with the appropriate hyperfine sublevels for the atom that the EnergyLevel is in
        """
        if isinstance(self.parent, Atom):
            for f in np.arange(abs(self.term.J - self.parent.I), self.term.J + self.parent.I + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=f)
                shift = self.compute_hf_shift(f)
                e = HFLevel(t, self.level + shift, lande=self.lande,
                            parent=self, atom=self.atom,
                            hfA=self.hfA, hfB=self.hfB, hfC=self.hfC)
                self[f'F={f}'] = e

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = value

    def compute_hf_shift(self, F):
        """
        Computes the hyperfine shift of a level given the EnergyLevel's hyperfine coefficients and an F level
        :param F: the F-level to be calculated
        :return: the shift of the level
        """
        J = self.term.J
        I = self.atom.I

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

        return self.hfA * FM1 + self.hfB * FE2 + self.hfC * FM3

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

    def populate_HF_transitions(self, include_zeeman=True):
        if self.atom is None:
            raise AttributeError('EnergyLevel needs to be contained in an atom to add transitions')
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

    def values(self):
        return self._sublevels.values()

    def keys(self):
        return self._sublevels.keys()

class HFLevel(EnergyLevel):
    def __init__(self, term: Term, level: pint.Quantity, lande: float = None, parent=None, atom=None,
                 hfA=0.0, hfB=0.0, hfC=0.0):
        """
        TODO: docstring
        :param term:
        :param level:
        :param lande:
        :param parent:
        :param atom:
        :param hfA:
        :param hfB:
        :param hfC:
        """
        super(HFLevel, self).__init__(term, level, lande, parent, atom, hfA, hfB, hfC)
        self.gF = self.compute_gF()
        self.shift = self._level - self.parent.level

    def populate_sublevels(self):
        """Populates the sublevels dict with the appropriate Zeeman sublevels"""
        if isinstance(self.parent, EnergyLevel):
            for mf in np.arange(-self.term.F, self.term.F + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=self.term.F, mF=mf)
                e = ZLevel(t, self.level, lande=self.lande,
                           parent=self, atom=self.atom,
                           hfA=self.hfA, hfB=self.hfB, hfC=self.hfC)
                self[f'mF={mf}'] = e

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
    def populate_sublevels(self):
        """A Zeeman sublevel will have no further sublevels."""
        pass

class Transition:
    def __init__(self, E1: EnergyLevel, E2: EnergyLevel, freq=None, A: pint.Quantity = None,
                 name=None, update_mode='upper', atom=None):
        """
        TODO: docstring
        :param E1:
        :param E2:
        :param freq:
        :param A:
        :param name:
        :param update_mode:
        :param atom:
        """
        self.E_1 = E1
        self.E_2 = E2
        self.A = A
        if self.E_2.level > self.E_1.level:
            self.E_upper = self.E_2
            self.E_lower = self.E_1
        else:
            self.E_upper = self.E_1
            self.E_lower = self.E_2
        self.name = name
        if self.name is None:
            self.name = f'{self.E_1.name} -> {self.E_2.name}'
        self.atom = atom
        self.freq = abs(self.E_1.level - self.E_2.level)
        self.wl = self.freq.to('nm')
        if freq is not None:
            if self.E_1.level_constrained or self.E_2.level_constrained:
                raise ValueError('This level has been set by another transition')
                # TODO: make this a warning
            self.E_upper.level_constrained = True
            self.E_lower.level_constrained = True
            if update_mode == 'upper':
                self.E_upper.level = self.E_lower.level + freq.to('Hz')
            elif update_mode == 'lower':
                self.E_lower.level = self.E_upper.level - freq.to('Hz')
            elif update_mode == 'ignore':
                pass
            else:
                raise ValueError('Accepted arguments to update_mode are "upper", "lower", and "ignore"')
    # TODO: Add appropriate methods. Things like getting the transition type (via clebsch-gordan math), perhaps
    #  determining color, and computing the transition strength / linewidth given the A coefficient

# noinspection PyPropertyDefinition
class Atom:
    def __init__(self, name: str, I: float = 0.0,
                 levels: List[EnergyLevel] = None, transitions: List[Transition] = None):
        """
        TODO: docstring
        :param name:
        :param I:
        :param levels:
        :param transitions:
        """
        self.name = name
        self.I = Term.frac_to_float(I)
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
        level.parent = self
        level.atom = self
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
        :param transition: the Transition to be added
        :return:
        """
        print(isinstance(transition.E_1, HFLevel))

        if type(transition.E_1) == EnergyLevel:
            print(f'adding transition between {transition.E_1.name} and {transition.E_2.name}')
            self.levelsModel.add_edge(transition.E_1, transition.E_2, transition=transition)
        elif type(transition.E_1) == HFLevel:
            print(f'adding hyperfine transition between {transition.E_1.name} and {transition.E_2.name}')
            self.levelsModel.add_edge(transition.E_1.parent, transition.E_2.parent, transition=transition)
            self.hfModel.add_edge(transition.E_1, transition.E_2, transition=transition)
        elif type(transition.E_1) == ZLevel:
            print(f'adding zeeman hyperfine transition between {transition.E_1.name} and {transition.E_2.name}')
            self.levelsModel.add_edge(transition.E_1.parent.parent, transition.E_2.parent.parent, transition=transition)
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

    # endregion

    # TODO: possibly a provision for B-fields? This would have to propagate down the whole tree, annoyingly


if __name__ == '__main__':
    def energy_level_from_df(df, i):
        t = Term(df["Configuration"][i], df["Term"][i], df["J"][i], percentage=df["Leading percentages"])
        e = EnergyLevel(t, df["Level (cm-1)"][i], lande=df["Lande"][i], hfA=10 * ureg('gigahertz'))
        return e

    species = "Yb II"
    I = 0.5
    num_levels = 80
    df = load_NIST_data(species)

    a = Atom(species, I=I)
    for i in range(num_levels):
        try:
            e = energy_level_from_df(df, i)
            a.add_level(e)
        except KeyError:
            pass

    # a = Atom.from_pickle('171Yb.atom')

    for l in list(a.levels.values()):
        print('MAIN:', l.name, l.level.to('THz'))
        for s in list(l.values()):
            print('    SUB:', s.term.term_name, s.shift.to('THz'))

    # a.to_pickle('171Yb')
    cooling = Transition(a.levelsModel.nodes['4f14.6s 2S1/2']['level'], a.levelsModel.nodes['4f14.6p 2P*1/2']['level'])

    a.add_transition(cooling)

    print('=== Before ===')

    print(a.levelsModel.edges)
    print(a.hfModel.edges)
    print(a.zModel.edges)

    a.levels['4f13.(2F*).6s2 2F*7/2'].populate_HF_transitions()

    print(a.levelsModel.edges)
    print(a.hfModel.edges)
    print(a.zModel.edges)
    #
    # import matplotlib.pyplot as plt
    # nx.draw(a.zModel)
    # plt.show()