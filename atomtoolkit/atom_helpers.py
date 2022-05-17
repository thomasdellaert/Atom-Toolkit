import networkx as nx


class LevelStructure:
    def __init__(self, atom, model: nx.Graph, hfmodel: nx.Graph, zmodel: nx.Graph):
        self.atom = atom
        self.model = model
        self.hf_model = hfmodel
        self.z_model = zmodel

    def __repr__(self):
        return f'LevelStructure containing {len(self)} levels of type {type(list(self.values())[0]).__name__}'

    def __getitem__(self, key):
        if 'mF=' in key: # match [conf] [term] F=[f] mF=[mF]
            main, hf, z = key.rsplit(' ', maxsplit=2)
            return nx.get_node_attributes(self.model, 'level')[main][hf][z]
        if 'mJ=' in key:
            main, z = key.rsplit(' mJ=', maxsplit=1)
            l =  nx.get_node_attributes(self.model, 'level')[main]
            return l[f'F={l.term.J_frac}'][f'mF={z}']
        elif 'F=' in key: # match [conf] [term] F=[f]
            main, hf = key.rsplit(' ', maxsplit=1)
            return nx.get_node_attributes(self.model, 'level')[main][hf]
        else: # match [conf] [term]
            return nx.get_node_attributes(self.model, 'level')[key]

    def __setitem__(self, key, value):
        self.atom.add_level(value, key=key)

    def __delitem__(self, key):
        self.model.remove_node(key)

    def __len__(self):
        return len(nx.get_node_attributes(self.model, 'level'))

    def __iter__(self):
        return nx.get_node_attributes(self.model, 'level').__iter__()

    def keys(self):
        return nx.get_node_attributes(self.model, 'level').keys()

    def values(self):
        return nx.get_node_attributes(self.model, 'level').values()

    def levels(self):
        return self.values()

    def append(self, value):
        self.atom.add_level(value)

    def list_names(self):
        return list(self.keys())


class TransitionStructure:
    def __init__(self, atom, model: nx.Graph, hfmodel: nx.Graph, zmodel: nx.Graph):
        self.atom = atom
        self.model = model
        self.hf_model = hfmodel
        self.z_model = zmodel

    def __repr__(self):
        return f'TransitionStructure containing {len(self)} transitions of type {type(list(self.values())[0]).__name__}'

    def __getitem__(self, key):
        lvl0 = self.atom.levels[key[0]]
        lvl1 = self.atom.levels[key[1]]
        if lvl0.level > lvl1.level:
            k_lower, k_upper = key[1], key[0]
        else:
            k_lower, k_upper = key[0], key[1]
        if 'mF=' in key[0]: # match [conf] [term] F=[f] mF=[mF]
            main_0, hf_0, z_0 = k_lower.rsplit(' ', maxsplit=2)
            main_1, hf_1, z_1 = k_upper.rsplit(' ', maxsplit=2)
            return nx.get_edge_attributes(self.model, 'transition')[(main_0, main_1)][(hf_0, hf_1)][(z_0, z_1)]
        if 'mJ=' in key[0]:
            main_0, z_0 = k_lower.rsplit(' mJ=', maxsplit=1)
            main_1, z_1 = k_upper.rsplit(' mJ=', maxsplit=1)
            l_0 =  nx.get_node_attributes(self.model, 'level')[main_0]
            l_1 =  nx.get_node_attributes(self.model, 'level')[main_1]
            return nx.get_edge_attributes(self.model, 'transition')[(main_0, main_1)][
                (f'F={l_0.term.J_frac}', f'F={l_1.term.J_frac}')][
                (f'mF={z_0}', f'mF={z_1}')]
        elif 'F=' in key[0]: # match [conf] [term] F=[f]
            main_0, hf_0 = k_lower.rsplit(' ', maxsplit=1)
            main_1, hf_1 = k_upper.rsplit(' ', maxsplit=1)
            return nx.get_edge_attributes(self.model, 'transition')[(main_0, main_1)][(hf_0, hf_1)]
        else: # match [conf] [term]
            return nx.get_edge_attributes(self.model, 'transition')[(k_lower, k_upper)]

    def __setitem__(self, value):
        self.atom.add_transition(value)

    def __delitem__(self, level1, level2):
        self.model.remove_edge(level1, level2)
        # TODO: make this also remove children? Think about how to implement this. probably this should be done by
        #  overriding __del__ in BaseLevel and BaseTransition

    def __len__(self):
        return len(nx.get_edge_attributes(self.model, 'transition'))

    def __iter__(self):
        return nx.get_edge_attributes(self.model, 'transition').__iter__()

    def keys(self):
        return nx.get_edge_attributes(self.model, 'transition').keys()

    def values(self):
        return nx.get_edge_attributes(self.model, 'transition').values()

    def append(self, value):
        self.atom.add_transition(value)

    def list_names(self, hide_self_transitions=True):
        if hide_self_transitions:
            return [key for key in self.keys() if key[0] != key[1]]
        return list(self.keys())

# CONSIDER: search_levels? search_transitions?
