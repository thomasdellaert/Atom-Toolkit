import csv
import re
import warnings

import pandas as pd
from tqdm import tqdm

from . import *
from .atom import Atom

warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_NIST_data(species, term_ordered=False, save=False):
    pbar = tqdm(total=7)
    pbar.update(1)
    pbar.set_description('loading data')
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

    pbar.update(1)
    pbar.set_description('data loaded')
    # === strip the data of extraneous symbols ===
    df_clean = df.applymap(lambda x: x.strip(' ="?'))
    pbar.update(1)
    pbar.set_description('data stripped')
    # === coerce types ===
    df_clean['Configuration'] = df_clean['Configuration'].astype('str')
    df_clean['Term'] = df_clean['Term'].astype('str')
    df_clean['Term'] = df_clean['Term'].apply(lambda x: re.sub(r'[a-z] ', '', x))
    df_clean['J'] = df_clean['J'].astype('str')
    df_clean['Level (cm-1)'] = pd.to_numeric(df_clean['Level (cm-1)'], errors='coerce')
    #  keep only the initial number of the leading percentage for now, replacing NaN with 100% I guess
    if 'Leading percentages' not in df_clean.columns:
        df_clean['Leading percentages'] = '100'
    df_clean['Leading percentages'] = df_clean['Leading percentages'].apply(lambda x: re.sub(r' ?:.*', '', x))
    df_clean['Leading percentages'] = pd.to_numeric(df_clean['Leading percentages'], errors='coerce')
    df_clean['Leading percentages'] = df_clean['Leading percentages'].fillna(value=100.0)
    if 'Lande' not in df_clean.columns:
        df_clean['Lande'] = None
    df_clean['Lande'] = pd.to_numeric(df_clean['Lande'], errors='coerce')
    pbar.update(1)
    pbar.set_description('data typed')
    # drop rows that don't have a defined level
    df_clean = df_clean.dropna(subset=['Level (cm-1)'])
    pbar.update(1)
    pbar.set_description('empty levels dropped')
    # convert levels to pint Quantities
    df_clean['Level (cm-1)'] = df_clean['Level (cm-1)'].astype('pint[cm**-1]')
    df_clean['Level (Hz)'] = df_clean['Level (cm-1)'].pint.to('Hz')
    pbar.update(1)
    pbar.set_description('pint quantities created')

    df_clean = df_clean.loc[
               :(df_clean['Term'] == 'Limit').idxmax() - 1]  # remove any terms above ionization, and the ionization row
    df_clean = df_clean[df_clean.J.str.contains(",") == False]  # happens when J is unknown
    # reset the indices, since we may have dropped some rows
    df_clean.reset_index(drop=True, inplace=True)

    pbar.update(1)
    pbar.set_description('data finalized')

    pbar.close()

    if save:
        if type(save) == str:
            if save[-4:] != '.csv':
                save += '.csv'
        else:
            save = f'{species}_level_data.csv'
        df_clean.to_csv(save)

    return df_clean
    #  TODO: uncertainty


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
        df["A"] = df_file[columns["A"]]
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


def generate_transition_csv(atom, filename):
    if filename is None:
        filename = f'{atom.name}_Transitions.csv'
    raise NotImplementedError
    # TODO: generate_transition_csv


def apply_transition_csv(atom, filename):
    raise NotImplementedError
    # TODO: apply_transition_csv


def generate_hf_csv(atom, filename=None, blank=False, def_A=Q_(0.0, 'gigahertz')):
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
        except FileNotFoundError:
            pass
    if 'transitions_csv' in kwargs:
        apply_transition_csv(a, kwargs['transitions_csv'])
    elif 'transitions_df' in kwargs:
        if kwargs['transitions_df'] is not None:
            a.populate_transitions_df(kwargs['transitions_df'], **kwargs)
        else:
            a.populate_transitions(**kwargs)
    else:
        a.populate_transitions(allowed=(True, False, False), **kwargs)
    a.populate_internal_transitions()
    return a
