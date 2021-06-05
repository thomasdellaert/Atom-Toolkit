from IO import load_NIST_data, load_transition_data
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import networkx as nx

from config import *
from atom import Atom

if __name__ == '__main__':
    def draw_levels(atom, plot_type='norm', **kwargs):
        posdict = {l.name:(l.term.J, l.level_Hz) for l in atom.levels.values()}
        if plot_type == 'norm':
            nx.draw(atom.levelsModel, pos=posdict, node_shape="_", with_labels=True, font_size=8, **kwargs)
        elif plot_type == 'hf':
            nx.draw(a.hfModel, pos=posdict, node_shape="_", with_labels=True, font_size=8, **kwargs)
        elif plot_type == 'z':
            nx.draw(a.zModel, pos=posdict, node_shape="_", with_labels=True, font_size=8, **kwargs)
        plt.show()

    def load_atom(species, num_levels=None, save=False, B=Q_(0.0, 'G'), pickleq=False):
        if pickleq:
            a = Atom.from_pickle(f'atoms/{species}.atom')
        else:
            df = load_NIST_data(speciesdict[species]['species'])
            if 'trans_path' in speciesdict[species]:
                trans_df = load_transition_data(speciesdict[species]['trans_path'], columns={
                "conf_l": "LConfiguration", "conf_u": "UConfiguration",
                "term_l": "LTerm", "term_u": "UTerm", "j_l": "LJ", "j_u": "UJ",
                "A": "A DREAM"}).dropna(subset=['A'])
            else:
                trans_df = None
            a = Atom.generate_full_from_dataframe(df, species, speciesdict[species]['I'],
                                                  num_levels=num_levels,
                                                  B=B,
                                                  hf_csv=f'resources/{species}_Hyperfine.csv',
                                                  transitions_df=trans_df,
                                                  allowed=0b001)
            if save:
                a.to_pickle(f'atoms/{species}.atom')
                a.generate_hf_csv(filename=f'resources/{species}_Hyperfine.csv')
        return a

    speciesdict = {
        '171Yb': {'species': 'Yb II', 'I': 0.5, 'trans_path': "resources/Yb_II_Oscillator_Strengths.csv"},
        '173Yb': {'species': 'Yb II', 'I': 2.5, 'trans_path': "resources/Yb_II_Oscillator_Strengths.csv"},
        '138Ba': {'species': 'Ba II', 'I': 0.0},
        '133Ba': {'species': 'Ba II', 'I': 0.5},
        '201Hg': {'species': 'Hg I', 'I': 1.5},
        '9Be':   {'species': 'Be II', 'I': 1.5}
    }

    # whether to load from pickle
    pickleq = True
    #whether to save the ion
    save = False
    # Name of the atom
    species = '171Yb'
    # Number of levels to generate
    num_levels = 200
    # Magnetic field
    B = Q_(5.0, 'G')

    a = load_atom(species, pickleq=pickleq, save=save, num_levels=num_levels, B=B)

    # for l in list(a.levels.values()):
    #     print('MAIN:', l.name, l.level.to('THz'), l.hfA)
    #     for s in list(l.values()):
    #         print('    SUB:', s.term.term_name, s.shift)

    # print(sorted(list(a.compute_branching_ratios('4f14.6p 2P*1/2').values()), reverse=True))
    # print(a.compute_branching_ratios('4f14.6p 2P*1/2'))

    draw_levels(a, edge_color=(0.0, 0.0, 0.0, 0.1), node_size=400)
