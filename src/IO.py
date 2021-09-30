import re
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .config import *

import pandas as pd
from tqdm import tqdm

def load_NIST_data(species, term_ordered=False):
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
    #    keep only the initial number of the leading percentage for now, replacing NaN with 100% I guess
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

    df_clean = df_clean.loc[:(df_clean['Term'] == 'Limit').idxmax()-1] #remove any terms above ionization, and the ionization row
    df_clean = df_clean[df_clean.J.str.contains(",") == False]  # happens when J is unknown
    # reset the indices, since we may have dropped some rows
    df_clean.reset_index(drop=True, inplace=True)

    pbar.update(1)
    pbar.set_description('data finalized')

    pbar.close()
    return df_clean
    #  TODO: uncertainty

def load_transition_data(filename:str, columns:dict = None, **kwargs):
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
    df = pd.DataFrame(data={k:df_file[columns[k]] for k in ["conf_l", "term_l", "j_l",
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

if __name__ == "__main__":
    # df = load_transition_data("resources/Yb_II_Oscillator_Strengths.csv", columns={
    #     "conf_l": "LConfiguration",
    #     "conf_u": "UConfiguration",
    #     "term_l": "LTerm",
    #     "term_u": "UTerm",
    #     "j_l": "LJ",
    #     "j_u": "UJ",
    #     "A": "A atlas"
    # })
    df = load_NIST_data("Yb II")