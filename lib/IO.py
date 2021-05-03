import re

import pandas as pd


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
    #  TODO: uncertainty
