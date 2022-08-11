import re

from .transition_strengths import JJ_to_LS, JK_to_LS, LK_to_LS


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


def l_to_let(l: int or float) -> str:
    """
    :param l: a positive integer
    :return: the corresponding L-value in spectroscopic notation
    """
    if type(l) is float:
        if not l.is_integer():
            raise (ValueError("Requires an integer or an integer-valued float"))
    if type(l) is str:
        raise TypeError()
    return 'SPDFGHIKLMNOQRTUVWXYZ'[int(l)]


def frac_to_float(frac: float or str) -> float or None:
    """
        :param frac: a string formatted as "1/2", "5/2", "3", etc
        :return: the corresponding float
        """
    if frac is (None or ''):
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


def term_to_LS(conf, term, j):
    coupling = get_term_coupling(term)
    lc, sc, lo, so, jc, jo, l, s, k = get_quantum_nums(conf, term)
    if coupling == 'LS':
        return [(term, 100.00)]
    elif coupling == 'JK':
        return readable_JK_to_LS(j, jc, k, lc, sc, lo, so)
    elif coupling == 'JJ':
        return readable_JJ_to_LS(j, jc, jo, lc, sc, lo, so)
    elif coupling == 'LK':
        return readable_LK_to_LS(j, l, k, sc, so)


def _sets_to_list(sets):
    # converts the default output of the basis conversion methods below (an unlabeled array of floats) into
    # a dict that's human-readable
    if len(sets) == 0:
        raise ValueError("Invalid input state")
    return [(str(int(2 * s + 1)) + l_to_let(l), round(100 * ampl ** 2, 2)) for l, s, ampl in sets if ampl != 0]


def readable_JJ_to_LS(J, Jc, Jo, lc, sc, lo, so):
    sets = list(zip(*[list(a) for a in JJ_to_LS(J, Jc, Jo, lc, sc, lo, so)]))
    return _sets_to_list(sets)


def readable_JK_to_LS(J, Jc, K, lc, sc, lo, so):
    sets = list(zip(*[list(a) for a in JK_to_LS(J, Jc, K, lc, sc, lo, so)]))
    return _sets_to_list(sets)


def readable_LK_to_LS(J, L, K, sc, so):
    sets = list(zip(*[list(a) for a in LK_to_LS(J, L, K, sc, so)]))
    return _sets_to_list(sets)

############################################
#                 Parsing                  #
############################################

def get_term_coupling(term):
    """
            :return: A string corresponding to the coupling detected in the term symbol
            """
    if '[' in term:
        if term[0] in 'SPDFGHIKLMNOQRTUVWXYZ':
            return 'LK'
        else:
            return 'JK'
    elif '(' in term:
        return 'JJ'
    elif len(term) > 1:
        return 'LS'
    else:
        return "unknown"


def get_quantum_nums(conf, term):
    """
    Calls the appropriate term parsing method in order to (hopefully) extract all usable information
    :return: a tuple of useful quantum numbers
    """
    coupling = get_term_coupling(term)
    lc = sc = lo = so = jc = jo = l = s = k = None
    if coupling == 'LS':
        l, s = parse_LS_term(term)
    elif coupling == 'JK':
        lc, sc, lo, so, jc, k = parse_JK_term(conf, term)
    elif coupling == 'JJ':
        lc, sc, lo, so, jc, jo = parse_JJ_term(conf, term)
    elif coupling == 'LK':
        lc, sc, lo, so, l, k = parse_LK_term(conf, term)
    return lc, sc, lo, so, jc, jo, l, s, k


def parse_LS_term(term):
    """
    Parses an LS-coupled term. Looks for the following forms in self.term:
        {2S+1}{L}
    Examples:
        2F, 3P*, 1S, 6L
    :return: L, S
    """
    # find the following forms: 2F, 3P*, and extract the relevant substrings
    [(ss, ls)] = re.findall(r'(\d+)([A-Z])', term)
    s = (float(ss) - 1) / 2
    l = let_to_l(ls)
    return l, s


def parse_JK_term(conf, term):
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
    relevant_parts = re.findall(r'(\d+)([A-Z])\*?(?:<(.+?)>)?', conf)
    if len(relevant_parts) == 2:
        [(scs, lcs, jcs), (_, los, _)] = relevant_parts
    else:
        [(scs, lcs, jcs)] = relevant_parts
        los = conf[-1]
    # find the following forms: 2[3/2], 3[4]*, and extract the relevant substrings
    [(sos, ks)] = re.findall(r'(\d+)\[(.+?)]', term)

    jc = frac_to_float(jcs)
    k = frac_to_float(ks)
    sc = (float(scs) - 1) / 2
    so = (float(sos) - 1) / 2
    lc = let_to_l(lcs)
    lo = let_to_l(los)

    return lc, sc, lo, so, jc, k


def parse_JJ_term(conf, term):
    """
     Parses a JJ-coupled term.
     Looks for the following in self.term:
         ({Jc, Jo})<{J}
     Examples:
         (2, 1/2)<5/2>

     :return: Lc, Sc, Lo, So, Jc, Jo
     """
    # find the following forms: 3D<2>, 7p<3/2>, (8,5/2)*<21/2>, and extract the relevant substrings
    relevant_parts = re.findall(r'(?:(\d+)([A-Za-z])|\(.+?\))\*?<(.+?)>', conf)
    if len(relevant_parts) == 0:  # sometimes the ancestor terms are in the term, not in the config
        relevant_parts = re.findall(r'(?:(\d+)([A-Za-z])|\(.+?\))\*?<(.+?)>', term)
    [(scs, lcs, jcs), (sos, los, jos)] = relevant_parts

    jc = frac_to_float(jcs)
    jo = frac_to_float(jos)
    lc = let_to_l(lcs)
    lo = let_to_l(los)
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


def parse_LK_term(conf, term):
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
    relevant_parts = re.findall(r'\((\d+)([A-Z])\*?\)', conf)
    if len(relevant_parts) == 2:
        [(scs, lcs), (_, los)] = relevant_parts
    else:
        [(scs, lcs)] = relevant_parts
        los = conf[-1]
    # find the following forms: D 2[3/2], P* 3[4]*, and extract the relevant substrings
    [(ls, sos, ks)] = re.findall(r'([A-Z])\*? ?(\d+)\[(.+?)]', term)

    k = frac_to_float(ks)
    sc = (float(scs) - 1) / 2
    so = (float(sos) - 1) / 2
    l = let_to_l(ls)
    lc = let_to_l(lcs)
    lo = let_to_l(los)

    return lc, sc, lo, so, l, k
