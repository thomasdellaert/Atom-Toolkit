import collections.abc

import numpy as np

from . import util


class Term:
    """
    A Term contains all the good quantum numbers of an EnergyLevel, as well as any ancestor terms that can be
    extracted from the configuration. These ancestor terms can (sometimes? often?) be used to convert between
    different couplings, among other things.
    """

    # TODO: Right now, terms are defined one level deep. In principle, they could be defined recursively
    #  from the configuration with fewer assumptions

    __slots__ = ['conf', 'term', 'percentage', 'parity',
                 'J', 'F', 'mF',
                 'J_frac', 'F_frac', 'mF_frac',
                 'term_name', 'short_name', 'name',
                 'coupling', 'quantum_nums',
                 'lc', 'sc', 'lo', 'so', 'jc', 'jo', 'l', 's', 'k']

    def __init__(self,
                 conf: str, term: str, J: float or str,
                 F: float or str = None, mF: float or str = None,
                 percentage=100.0, quantum_nums=None):
        """
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
        self.parity = (-1 if '*' in self.term else 1)

        self.J = util.frac_to_float(J)
        self.F = util.frac_to_float(F)
        self.mF = util.frac_to_float(mF)
        self.J_frac, self.F_frac, self.mF_frac = util.float_to_frac(self.J), util.float_to_frac(self.F), util.float_to_frac(
            self.mF)

        self.term_name = f'{self.term}{self.J_frac}'
        self.short_name = f'{self.term}{self.J_frac}'
        if self.F is not None:
            self.term_name += f' F={self.F_frac}'
            self.short_name = f'F={self.F_frac}'
        if self.mF is not None:
            self.term_name += f' mF={self.mF_frac}'
            self.short_name = f'mF={self.mF_frac}'

        self.name = f'{self.conf} {self.term_name}'

        self.coupling = util.get_term_coupling(self.term)

        if quantum_nums is None:
            quantum_nums = util.get_quantum_nums(self.conf, self.term)
        self.quantum_nums = quantum_nums
        self.lc, self.sc, self.lo, self.so, self.jc, self.jo, self.l, self.s, self.k = self.quantum_nums

    def __str__(self):
        return self.term_name #CONSIDER: Is this what I want?

    def __repr__(self):
        return f'Term({self.name})'

    def __eq__(self, other):
        # CONSIDER: rn this matches multiterms. Is this the behavior I want?
        return self.quantum_nums == other.quantum_nums and \
               self.conf == other.conf and \
               self.term == other.term

    def make_term_copy(self, F=None, mF=None):
        """
        make a copy of the original term, potentially adding F and/or mF for producing subterms
        :param F:
        :param mF:
        :return:
        """
        if self.F is not None:
            return Term(self.conf, self.term, self.J, F=self.F, mF=mF, quantum_nums=self.quantum_nums, percentage=self.percentage)
        return Term(self.conf, self.term, self.J, F=F, mF=mF, quantum_nums=self.quantum_nums, percentage=self.percentage)

    @classmethod
    def from_dataframe(cls, df, i=0):
        """Returns a term as listed in the nist ASD csv dataframe generated by the IO module"""
        return Term(df["Configuration"][i], df["Term"][i], df["J"][i], percentage=df["Leading percentages"])


class MultiTerm(collections.abc.Sequence):

    def __init__(self, *terms: Term):
        """
        A MultiTerm is a container for multiple terms, useful for situations in which just the leading term won't do. It
        passes through any attribute requests to the leading term, so that it doesn't have to be engaged with unless needed
        :param terms: the terms that the MultiTerm contains
        """
        self.terms = sorted(terms, key=lambda t: t.percentage, reverse=True)
        self.terms_dict = {t.percentage: t for t in self.terms}

    def __getattribute__(self, item):
        """
        When you need to access a term-like attribute, this ensures that the MultiTerm defaults to the leading term.
        For example:
            MT.mF == MT.terms[0].mF
        """
        if item in Term.__slots__:
            return self.terms[0].__getattribute__(item)
        return object.__getattribute__(self, item)

    @property
    def full_name(self):
        return f'{self.name} ({self.percentage}%)'

    def __getitem__(self, item: int):
        return self.terms[item]

    def __len__(self):
        return self.terms.__len__()

    def __eq__(self, other):
        if isinstance(other, Term):
            return self[0] == other
        return np.all([self[i] == other[i] for i in range(len(self))])

    def make_term_copy(self, F=None, mF=None):
        """
        make a copy of the original term, potentially adding F and/or mF for producing subterms
        :param F:
        :param mF:
        :return:
        """
        return MultiTerm(*[t.make_term_copy(F, mF) for t in self.terms])

    @classmethod
    def from_dataframe(cls, df, i=0):
        """Returns a term as listed in the nist ASD csv dataframe generated by the IO module"""
        confs = [c for c in df.columns if 'Configuration' in c]
        terms = [c for c in df.columns if 'Term' in c]
        pcts = [c for c in df.columns if 'Percentage' in c]
        term_objs = []
        for j in range(len(confs)):
            if type(df[confs[j]][i]) == str:
                term_objs.append(Term(df[confs[j]][i], df[terms[j]][i], df['J'][i], percentage=df[pcts[j]][i]))
        return MultiTerm(*term_objs)