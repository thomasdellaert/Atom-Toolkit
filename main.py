import pandas as pd
import numpy as np
# import networkx as nx
import pint
import pint_pandas
# import json
import re

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
        # find the following forms: 2F, 3P*, and extract the relevant substrings
        [(ss, ls)] = re.findall(r'(\d+)([A-Z])', self.term)
        s = (float(ss) - 1)/2
        l = self.let_to_l(ls)
        return l, s

    def parse_JK_term(self):
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
        if let == '':
            return None
        if len(let) > 1:
            raise ValueError('Argument must be a single character in "SPDFGHIKLMNOQRTUVWXYZ" or an empty string')
        return 'SPDFGHIKLMNOQRTUVWXYZ'.index(let.upper())

    @staticmethod
    def l_to_let(l: int) -> str:
        return 'SPDFGHIKLMNOQRTUVWXYZ'[l]

    @staticmethod
    def frac_to_float(frac: float or str) -> float or None:
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
        self.term = term
        self._level = None
        self.level = level.to('Hz')
        self.parent = parent
        self.atom = atom
        self.name = self.term.name
        self.hfA, self.hfB, self.hfC = hfA, hfB, hfC
        self.level_constrained = False
        if lande is None:
            self.lande = self.compute_gJ()
        else:
            self.lande = lande
        self._sublevels = {}
        self.populate_sublevels()

    def populate_sublevels(self):
        if isinstance(self.parent, Atom):
            for f in np.arange(abs(self.term.J - self.parent.I), self.term.J + self.parent.I + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=f)
                shift = self.compute_hf_shift(f)
                e = EnergyLevel(t, self.level + shift, lande=self.lande,
                                parent=self, atom=self.atom,
                                hfA=self.hfA, hfB=self.hfB, hfC=self.hfC)
                self[f'F={f}'] = e

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        try:
            shift = value - self.level
        except pint.errors.DimensionalityError:
            self._level = value
            return
        self._level = value
        for sublevel in list(self.values()):
            sublevel.level += shift

    def compute_hf_shift(self, F):
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
        if self.term.coupling != 'LS':
            raise ValueError("Unable to compute g_J for non-LS-coupled terms")
        J = self.term.J
        L = self.term.l
        S = self.term.s
        return 1 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

    def update_level(self, value):
        # a method for when the level of a child level changes, necessitating an update
        shift = value - self.level
        self._level = self.level + shift

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

    def keys(self):
        return self._sublevels.keys()

class HFLevel(EnergyLevel):
    def __init__(self, term: Term, level: pint.Quantity, lande: float = None, parent=None, atom=None,
                 hfA=0.0, hfB=0.0, hfC=0.0):
        super(HFLevel, self).__init__(term, level, lande, parent, atom, hfA, hfB, hfC)
        self.gF = self.compute_gF()

    def populate_sublevels(self):
        if isinstance(self.parent, EnergyLevel):
            for mf in np.arange(-self.term.F, self.term.F + 1):
                t = Term(self.term.conf, self.term.term, self.term.J, F=self.term.F, mF=mf)
                e = EnergyLevel(t, self.level, lande=self.lande,
                                parent=self, atom=self.atom,
                                hfA=self.hfA, hfB=self.hfB, hfC=self.hfC)
                self[f'mF={mf}'] = e

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        try:
            shift = value - self.level
        except pint.errors.DimensionalityError:
            self._level = value
            return
        self._level = value
        self.parent.update_level(self.parent.level + shift)
        for sublevel in list(self.values()):
            sublevel.level += shift

    def compute_gF(self):
        F = self.term.F
        J = self.term.J
        I = self.atom.I
        if F != 0:
            return self.lande * (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1))
        return 0

class ZLevel(HFLevel):
    def populate_sublevels(self):
        pass

class Transition:
    def __init__(self, E1: EnergyLevel, E2: EnergyLevel, freq=None, A: float = None,
                 name=None, update_mode='upper', atom=None):
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


class Atom:
    def __init__(self, name: str, I: float = 0.0,
                 levels: List[EnergyLevel] = None, transitions: List[Transition] = None):
        self.name = name
        self.I = Term.frac_to_float(I)
        self.levels = _LevelDict(self)
        self.transitions = _TransitionDict(self)
        if levels is not None:
            for level in levels:
                self.levels.append(level)
        if transitions is not None:
            for transition in transitions:
                self.transitions.append(transition)

    def __str__(self):
        return f'{self.name} I={Term.float_to_frac(self.I)}'

    def __repr__(self):
        return f"Atom({self.name}, I={self.I}, levels={self.levels})"

    def to_JSON(self):
        # TODO: self.to_JSON
        pass

    @classmethod
    def from_JSON(cls):
        # TODO: cls.from_JSON
        pass

    def to_pickle(self):
        # TODO: self.to_pickle
        pass

    @classmethod
    def from_pickle(cls):
        # TODO: cls.from_pickle
        pass

    # TODO: possibly a provision for B-fields? This would have to propagate down the whole tree, annoyingly

class _TransitionDict:
    def __init__(self, atom: Atom):
        self._transitions = {}
        self.atom = atom

    def __len__(self):
        return len(self._transitions)

    def __getitem__(self, key):
        return self._transitions[key]

    def __setitem__(self, key, transition):
        transition.atom = self
        self._transitions[key] = transition

    def __delitem__(self, key):
        del self._transitions[key]

    def __iter__(self):
        return iter(self._transitions)

    def append(self, transition):
        transition.atom = self.atom
        self._transitions[transition.name] = transition

    def values(self):
        return self._transitions.values()

    def keys(self):
        return self._transitions.keys()


class _LevelDict:
    def __init__(self, atom: Atom):
        self._levels = {}
        self.atom = atom

    def __len__(self):
        return len(self._levels)

    def __getitem__(self, key):
        return self._levels[key]

    def __setitem__(self, key, level):
        level.parent = self
        level.atom = self
        level.populate_sublevels()
        self._levels[key] = level

    def __delitem__(self, key):
        del self._levels[key]

    def __iter__(self):
        return iter(self._levels)

    def append(self, level):
        level.parent = self.atom
        level.atom = self.atom
        level.populate_sublevels()
        self._levels[level.name] = level

    def values(self):
        return self._levels.values()

    def keys(self):
        return self._levels.keys()


if __name__ == '__main__':
    def energy_level_from_df(df, i):
        t = Term(df["Configuration"][i], df["Term"][i], df["J"][i], percentage=df["Leading percentages"])
        e = EnergyLevel(t, df["Level (cm-1)"][i], lande=df["Lande"][i], hfA=0.1 * ureg('megahertz'))
        return e

    species = "Yb II"
    I = 0.5
    num_levels = 30
    df = load_NIST_data(species)

    a = Atom(species, I=I)
    for i in range(num_levels):
        try:
            e = energy_level_from_df(df, i)
            a.levels.append(e)
        except KeyError:
            pass

    for l in list(a.levels.values()):
        print('MAIN:', l.name, l.level.to('THz'))
        for s in list(l.values()):
            print('    SUB:', s.term.term_name, s.level.to('THz'))
    #         for z in list(s.values()):
    #             print('        Zee:', z.term.term_name)
