import csv
import re
import urllib.error
import warnings

import pandas as pd

from . import *
from .atom import Atom


# When printing or displaying DataFrames, pint_pandas really likes to throw UnitStrippedWarnings,
# so we'll turn them off if the module is imported
warnings.filterwarnings("ignore", category=pint.UnitStrippedWarning)


def _parse_NIST_df(df: pd.DataFrame, save=False, species: str = None) -> pd.DataFrame:
    """
    Clean and parse a raw NIST ASD levels dataframe (from the API or a local CSV).
    Both sources produce the same Excel-quoted CSV format, so the pipeline is identical.
    """
    # === strip the data of extraneous symbols ===
    df_clean = df.applymap(lambda k: k.strip(' ="?') if isinstance(k, str) else k)

    # === coerce types ===
    df_clean['Configuration'] = df_clean['Configuration'].astype('str')
    df_clean['Term'] = df_clean['Term'].astype('str')
    df_clean['Term'] = df_clean['Term'].apply(lambda k: re.sub(r'[a-z] ', '', k))
    df_clean['J'] = df_clean['J'].astype('str')
    df_clean['Level (cm-1)'] = pd.to_numeric(df_clean['Level (cm-1)'], errors='coerce')

    if 'Lande' not in df_clean.columns:
        df_clean['Lande'] = None
    df_clean['Lande'] = pd.to_numeric(df_clean['Lande'], errors='coerce')

    # === Parsing NIST's leading percentages column is extremely complicated ===

    #    Not all NIST ASD tables even include leading percentages, so insert the column if not present
    if 'Leading percentages' not in df_clean.columns:
        df_clean['Leading percentages'] = '100.0'
    #    Terms with insufficient first percentage have everything stored in the leading percentages column
    #    so find the terms with only '*' or '' and get the term from the leading percentage
    #       35  3[5/2]*       :    30                                 (2F*<7/2>)(3F) 3[3/2]*
    #       36  (7/2,1)       :    25  (2F*<7/2>).5d.6p.(1P*<1>)
    reg = r'\ *\d+\ +([^\ ]+)\ *.*\ :\ *\d+\ +.+?(?:\ +.+?)?\ *$'
    df_clean.loc[df_clean["Term"] == '', 'Term'] = df_clean['Leading percentages'].str.extract(reg, expand=False)
    df_clean.loc[df_clean["Term"] == '*', 'Term'] = df_clean['Leading percentages'].str.extract(reg, expand=False)
    #    extract the relevant data available in the config leading percentage. Example cases:
    #       81                :    11  (2F*<5/2>).5d2.(1D)            1[7/2]*
    #       35  3[5/2]*       :    30                                 (2F*<7/2>)(3F) 3[3/2]*
    df_clean[['Percentage', 'Percentage_2', 'Configuration_2', 'Term_2']] = \
        df_clean['Leading percentages'].str.extract(r'\ *(\d+)\ +.+?\ *.*:\ *(\d+)\ +(.+?)(?:\ +(.+?))?\ *$')
    #    sometimes, nist likes to give alternate assignments using the word "or", which doesn't make it into the
    #    API CSV. For now, just drop these alternate assignments and replace with None
    #    TODO: Allow term objects to have alternate coupling schemes. Almost always, normal coupling conversion
    #     yields the same result, so this is low priority and could even be auto-detected
    df_clean.loc[pd.to_numeric(df_clean['Percentage'], errors='coerce') +
                 pd.to_numeric(df_clean['Percentage_2'], errors='coerce') > 100.0, 'Configuration_2'] = 'None'
    df_clean.loc[pd.to_numeric(df_clean['Percentage'], errors='coerce') +
                 pd.to_numeric(df_clean['Percentage_2'], errors='coerce') > 100.0, 'Term_2'] = ''
    df_clean.loc[pd.to_numeric(df_clean['Percentage'], errors='coerce') +
                 pd.to_numeric(df_clean['Percentage_2'], errors='coerce') > 100.0, 'Percentage_2'] = ''
    #    NIST doesn't see fit to give the jj-coupled terms in their second percentages,
    #    so we extract the j values from the configuration and store them in Term_2_int
    #       54                :    19                                 (2F*<5/2>)(3F*<2>)
    df_clean.loc[df_clean["Term_2"].isnull(), 'Term_2_int'] = df_clean['Configuration_2'].str.findall(r'\<(.+?)\>')
    #    take the j values from Term_2_int and create a JJ term symbol from them
    try:
        df_clean.loc[~df_clean["Term_2_int"].isnull(), 'Term_2'] = \
            df_clean["Term_2_int"].map(lambda k: '({}, {})'.format(*k), na_action='ignore')
    except IndexError:
        # FIXME sometimes some values in Term_2_int aren't JJ coupled (so far only seen in Na I, above ionization).
        #   I do not have time for this right now, and if any Rydberg people wanna get mad at me later they can fix it
        pass
    df_clean = df_clean.drop("Term_2_int", axis=1)
    df_clean['Term'] = df_clean['Term'].fillna('')
    df_clean['Term_2'] = df_clean['Term_2'].fillna('')

    def reconstitute_conf(c0, c1):
        """
        grab any info in c0 that has been removed from c1 and put it back where it belongs. For instance:
        4f13.(2F*<7/2>).5d2.(3P) + (2F*<7/2>)(3F) ==> 4f13.(2F*<7/2>).5d2.(3F)
        """
        if type(c1) != str:
            return
        if re.match(r'\(.+?\)\(.+?\)', c1):
            return re.sub(r'\((.+?)\)', '({})', c0).format(*re.findall(r'\((.+?)\)', c1))
        elif re.match(r'\(.+?\).+\(.+?\)', c1):
            return c0.split('(')[0]+c1
        else:
            return c1

    df_clean['Configuration_2'] = df_clean.apply(
        lambda k: reconstitute_conf(k['Configuration'], k['Configuration_2']), axis=1)
    df_clean['Percentage'] = pd.to_numeric(df_clean['Percentage'], errors='coerce')
    df_clean['Percentage_2'] = pd.to_numeric(df_clean['Percentage_2'], errors='coerce')
    df_clean['Percentage'] = df_clean['Percentage'].fillna(value=100.0)
    df_clean = df_clean.drop('Leading percentages', axis=1)

    # drop rows that don't have a defined level
    df_clean = df_clean.dropna(subset=['Level (cm-1)'])

    # convert levels to pint Quantities
    df_clean['Level (cm-1)'] = df_clean['Level (cm-1)'].astype('pint[cm**-1]')
    df_clean['Level (Hz)'] = df_clean['Level (cm-1)'].pint.to('Hz')

    # remove any terms above ionization, and the ionization row
    df_clean = df_clean.loc[:(df_clean['Term'] == 'Limit').idxmax() - 1]
    # split up levels with multiple J
    df_clean.J = df_clean.J.str.replace(" or", ",")
    df_clean.J = df_clean.J.str.split(",")
    df_clean = df_clean.explode('J')
    # reset the indices, since we may have dropped some rows
    df_clean.reset_index(drop=True, inplace=True)

    df_clean = df_clean[['Level (cm-1)', 'Level (Hz)', 'J', 'Lande',
                         'Configuration', 'Term', 'Percentage',
                         'Configuration_2', 'Term_2', 'Percentage_2']]

    if save:
        if isinstance(save, str):
            if save[-4:] != '.csv':
                save += '.csv'
        else:
            save = f'{species}_level_data.csv'
        df_clean.to_csv(save)

    return df_clean
    #  TODO: uncertainty


def _fetch_NIST_data(species: str, term_ordered: bool = False) -> pd.DataFrame:
    """Fetch raw NIST ASD level data from the web API and return the unparsed dataframe."""
    url = (
        'https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0'
        + '&spectrum=' + species.replace(' ', '+')
        + '&submit=Retrieve+Data'
        + '&units=0'
        + '&format=3'
        + '&output=0'
        + '&multiplet_ordered=' + ('on' if term_ordered else '0')
        + '&conf_out=on'
        + '&term_out=on'
        + '&level_out=on'
        + '&unc_out=on'
        + '&j_out=on'
        + '&lande_out=on'
        + '&perc_out=on'
        + '&temp='
    )
    return pd.read_csv(url, index_col=False)


_NIST_CSV_INSTRUCTIONS = """\
The NIST ASD API returned a {status} error for '{species}'.
To load data from a local CSV instead:
  1. Go to https://physics.nist.gov/PhysRefData/ASD/levels_form.html
  2. Enter the species (e.g. "Yb II"), select all output fields, and set Format = CSV
  3. Save the downloaded file, then call:
       IO.load_NIST_data_from_csv('path/to/file.csv')
"""


def load_NIST_data(species: str, term_ordered: bool = False, save=False) -> pd.DataFrame:
    """
    Load energy level data for a species from the NIST ASD web API.

    :param species: Species in astronomical notation, e.g. ``'Yb II'``, ``'Ca II'``
    :param term_ordered: If True, order levels by term rather than energy
    :param save: Save the cleaned dataframe to CSV. Pass a filename string or True
        (which autogenerates ``<species>_level_data.csv``)
    :return: Cleaned dataframe — see :func:`load_NIST_data_from_csv` for column details

    .. note::
        If the NIST ASD API is unavailable (e.g. returns 403), download the CSV manually
        from https://physics.nist.gov/PhysRefData/ASD/levels_form.html and use
        :func:`load_NIST_data_from_csv` instead.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            df = _fetch_NIST_data(species, term_ordered)
            return _parse_NIST_df(df, save=save, species=species)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                _NIST_CSV_INSTRUCTIONS.format(status=exc.code, species=species)
            ) from exc
        except Exception as exc:
            # Catch other network failures (timeouts, SSL errors, etc.)
            if any(kw in str(exc).lower() for kw in ('403', 'forbidden', 'connection', 'timeout', 'ssl')):
                raise RuntimeError(
                    _NIST_CSV_INSTRUCTIONS.format(status=type(exc).__name__, species=species)
                ) from exc
            raise


def load_NIST_data_from_csv(path: str, save=False) -> pd.DataFrame:
    """
    Load NIST ASD energy level data from a locally saved CSV file.

    The CSV must be in the format produced by the NIST ASD Levels form
    (Format = CSV).  Download from:
    https://physics.nist.gov/PhysRefData/ASD/levels_form.html

    :param path: Path to the CSV file
    :param save: Save the cleaned dataframe to a CSV. Pass a filename string or True
    :return: Cleaned dataframe with columns:
        ``Level (cm-1)``, ``Level (Hz)``, ``J``, ``Lande``,
        ``Configuration``, ``Term``, ``Percentage``,
        ``Configuration_2``, ``Term_2``, ``Percentage_2``
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        df = pd.read_csv(path, index_col=False)
        return _parse_NIST_df(df, save=save)


def load_transition_data(filename: str, columns: dict = None, **kwargs):
    """

    :param filename: the name of the CSV to be imported
    :param columns: a dict containing at least the following as keys, which defines the columns to use:
        "conf_l", "conf_u", "term_l", "term_u", "j_l", "j_u"
        optional columns include:
        "A" OR "A_coeff", "frequency" OR "freq", "alias", "nickname", "ref_A", "ref_freq"
    :return:
    """
    df_file = pd.read_csv(filename, **kwargs)
    if ("freq" not in columns) and ("frequency" in columns):
        columns["freq"] = columns["frequency"]
    if ("A" not in columns) and ("A_coeff" in columns):
        columns["A"] = columns["A_coeff"]
    df = pd.DataFrame(data={k: df_file[columns[k]] for k in ["conf_l", "term_l", "j_l",
                                                             "conf_u", "term_u", "j_u"]})
    if "freq" in columns:
        df["freq"] = df_file[columns["freq"]]
    else:
        df["freq"] = None
    if "A" in columns:
        df["A"] = df_file[columns["A"]].astype('pint[MHz]')
    else:
        df["A"] = None
    return df


def load_level_data(filename: str, columns: dict = None, **kwargs):
    """
    :param filename: the name of the CSV to be imported
    :param columns: a dict containing at least the following as keys, which defines the columns to use:
        conf
        term
        j
        level (cm-1)
        lande
        leading percentages
    :return:
    """
    if columns is None:
        columns = {i: i for i in ['configuration', 'term', 'j', 'level (cm-1)', 'lande', 'leading percentages']}

    df_file = pd.read_csv(filename, **kwargs)
    df_file.rename(str.lower, axis='columns')
    df = pd.DataFrame(data={k: df_file[columns[k]] for k in columns})
    try:
        df['Level (cm-1)'] = df['Level (cm-1)'].astype(float)
    except ValueError:
        df['Level (cm-1)'] = df['Level (cm-1)'].str.replace(r'[^\d\.]+', '', regex=True)
        df['Level (cm-1)'] = df['Level (cm-1)'].astype(float)
    df['Level (cm-1)'] = df['Level (cm-1)'].astype('pint[cm**-1]')
    df['Level (Hz)'] = df['Level (cm-1)'].pint.to('Hz')

    # df['Level (Hz)'] = df['Level (cm-1)'].pint.to('Hz')

    return df


def generate_transition_csv(atom, filename=None):
    """
    Export all transitions in the atom to a CSV file.

    Columns: ``lower``, ``upper``, ``A (MHz)``, ``freq (THz)``

    :param atom: the Atom to export transitions from
    :param filename: output filename; defaults to ``{atom.name}_Transitions.csv``
    """
    if filename is None:
        filename = f'{atom.name}_Transitions.csv'
    rows = []
    for (lower_name, upper_name), t in atom.transitions.items():
        try:
            A_val = t.A.to('MHz').magnitude if t.A is not None else ''
        except Exception:
            A_val = ''
        try:
            freq_val = t.freq.to('THz').magnitude if t.freq is not None else ''
        except Exception:
            freq_val = ''
        rows.append([lower_name, upper_name, A_val, freq_val])
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lower', 'upper', 'A (MHz)', 'freq (THz)'])
        writer.writerows(rows)


def apply_transition_csv(atom, filename):
    """
    Apply transition data from a CSV file to an Atom.  The CSV must be
    formatted like the output of :func:`generate_transition_csv` with columns
    ``lower``, ``upper``, ``A (MHz)``, and ``freq (THz)``.

    Missing or empty values for A or freq are silently skipped.

    :param atom: the Atom to update
    :param filename: path to the CSV file
    """
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = atom.transitions[(row['lower'], row['upper'])]
            except (KeyError, TypeError):
                continue
            if t is None:
                continue
            try:
                if row.get('A (MHz)', '').strip():
                    t.A = Q_(float(row['A (MHz)']), 'MHz')
            except (ValueError, AttributeError):
                pass
            try:
                if row.get('freq (THz)', '').strip():
                    t.freq = Q_(float(row['freq (THz)']), 'THz')
            except (ValueError, AttributeError):
                pass


def generate_hf_csv(atom, filename=None, blank=False, def_A=Q_(0.0, 'GHz')):
    """
    Output the hyperfine coefficients of every energy level in the atom
    :param atom: the atom from which to generate the csv
    :param filename: the filename to output to
    :param blank: whether to output a blank template file
    :param def_A: the default A value to write for levels without an A value
    :return:
    """
    if filename is None:
        filename = f'{atom.name}_Hyperfine.csv'
    if not blank:
        rows_to_write = [
            [level.name, (level.hfA if level.hfA != Q_(0.0, 'GHz') else def_A), level.hfB, level.hfC]
            for level in list(atom.levels.values())]
    else:
        rows_to_write = [
            [level.name, Q_(0.0, 'GHz'), Q_(0.0, 'GHz'), Q_(0.0, 'GHz')]
            for level in list(atom.levels.values())]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_write)


def apply_hf_csv(atom, filename):
    """
        Apply the hyperfine data csv. CSV should be formatted like the output of generate_hf_csv
        :param atom: the atom to apply the csv to
        :param filename: the file to load
        :return:
        """
    import csv
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                name, hfA, hfB, hfC = row
                atom.levels[name].hfA = Q_(hfA)
                atom.levels[name].hfB = Q_(hfB)
                atom.levels[name].hfC = Q_(hfC)
            except KeyError:
                pass


# TODO: apply_levels_csv and apply_hf_csv kind of do the same thing. Really I only need two functions:
#  one that iterates over levels, and one that iterates over transitions/pairs of levels

def generate_full_from_dataframe(df, name, I=0.0, **kwargs):
    """
    Generate the atom from a dataframe, formatted like the output from the IO module.
    Also populate the internal transitions of the atom.
    Also apply any csv's that are passed to it to the atom.
    :param df: the dataframe
    :param name: the atom's name
    :param I: the nuclear spin quantum number of the atom
    :param kwargs:
        'transitions_csv', 'transitions_df', 'hf_csv', 'subtransitions'
    :return: an instantiated atom
    """
    a = Atom.from_dataframe(df, name, I, **kwargs)
    if 'hf_csv' in kwargs:
        try:
            apply_hf_csv(a, kwargs['hf_csv'])
        except (FileNotFoundError, TypeError):
            pass
    if 'transitions_csv' in kwargs:
        apply_transition_csv(a, kwargs['transitions_csv'])
    elif 'transitions_df' in kwargs:
        if kwargs['transitions_df'] is not None:
            a.populate_transitions_df(kwargs['transitions_df'], **kwargs)
        else:
            a.populate_transitions(**kwargs)
    else:
        a.populate_transitions(**kwargs)
    a.populate_internal_transitions()
    return a
