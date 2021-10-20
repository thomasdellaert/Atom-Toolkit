import networkx as nx

class ModelFrontEnd:
    def __init__(self, atom, model: nx.Graph):
        self.atom = atom
        self.model = atom.levelsModel

class LevelStructure(ModelFrontEnd):

    def __getitem__(self, key):
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

class TransitionStructure(ModelFrontEnd):

    def __getitem__(self, key):
        try:
            try:
                return nx.get_edge_attributes(self.model, 'transition')[(key[1], key[0])]
            except KeyError:
                return nx.get_edge_attributes(self.model, 'transition')[key]
        except KeyError:
            return None

    def __setitem__(self, value):
        self.atom.add_transition(value)

    def __delitem__(self, level1, level2):
        self.model.remove_edge(level1, level2)
        # TODO: make this also remove children? Think about how to implement this

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