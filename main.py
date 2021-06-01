from IO import load_NIST_data, load_transition_data
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from networkx import draw

from config import *
from atom import Atom

if __name__ == '__main__':

    speciesdict = {
        '171Yb': {'species': 'Yb II', 'I': 0.5},
        '173Yb': {'species': 'Yb II', 'I': 2.5},
        '138Ba': {'species': 'Ba II', 'I': 0.0},
        '133Ba': {'species': 'Ba II', 'I': 0.5},
        '201Hg': {'species': 'Hg I', 'I': 1.5},
        '9Be':   {'species': 'Be II', 'I': 1.5}
    }

    # whether to load from pickle
    pickleq = False
    # Name of the atom
    species = '171Yb'
    # Number of levels to generate
    num_levels = 80
    # Magnetic field
    B = Q_(5.0, 'G')

    if pickleq:
        a = Atom.from_pickle(f'atoms/{species}.atom')
    else:
        df = load_NIST_data(speciesdict[species]['species'])
        trans_df = load_transition_data("resources/Yb_II_Oscillator_Strengths.csv", columns={
        "conf_l": "LConfiguration", "conf_u": "UConfiguration",
        "term_l": "LTerm", "term_u": "UTerm", "j_l": "LJ", "j_u": "UJ",
        "A": "A atlas"}).dropna(subset=['A'])
        a = Atom.generate_full_from_dataframe(df, species, speciesdict[species]['I'],
                                              num_levels=num_levels,
                                              B=B,
                                              hf_csv=f'resources/{species}_Hyperfine.csv',
                                              transitions_df=trans_df,
                                              allowed=0b001)
        a.to_pickle(f'atoms/{species}.atom')
        a.generate_hf_csv(filename=f'resources/{species}_Hyperfine.csv')

    # for l in list(a.levels.values()):
    #     print('MAIN:', l.name, l.level.to('THz'), l.hfA)
    #     for s in list(l.values()):
    #         print('    SUB:', s.term.term_name, s.shift)

    print(a.compute_branching_ratios('4f14.6p 2P*1/2'))
    print(sorted(list(a.compute_branching_ratios('4f14.6p 2P*1/2').values()), reverse=True))

    posdict = {l.name:(l.term.J, l.level_Hz) for l in a.levels.values()}

    draw(a.levelsModel, pos=posdict, with_labels=True, font_size=8, node_size=100)
    plt.show()

    #
    # draw(a.hfModel, with_labels=True, font_size=8, node_size=100)
    # plt.show()
    #
    # draw(a.zModel, with_labels=True, font_size=8, node_size=100)
    # plt.show()
