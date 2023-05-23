from __future__ import annotations
from typing import *

import networkx as nx

if TYPE_CHECKING:
    from atomtoolkit.atom import Atom, BaseLevel, BaseTransition, EnergyLevel, Transition


class LevelStructure:
    """
    A LevelStructure provides a dict-like interface to the *nodes* of an Atom's internal networkx Graph.
    It supports access to levels by various equivalent keys, as well as by user-assigned aliases
    """
    def __init__(self, atom: Atom, model: nx.Graph, hf_model: nx.Graph, z_model: nx.Graph):
        self.atom = atom
        self.model = model
        self.hf_model = hf_model
        self.z_model = z_model
        self.aliases = dict()

    def __repr__(self):
        return f'LevelStructure containing {len(self)} levels of type {type(list(self.values())[0]).__name__}'

    def __getitem__(self, key: Union[int, slice, str]) -> BaseLevel or List[BaseLevel]:
        """
        Parses the key to return the associated level. Supports various kinds of access. Access by:
            - integer:
                returns the nth-lowest-lying level
            - slice:
                returns the levels indexed by the slice
            - alias
            - [configuration] [term] F=[F] mF=[mF]
            - [configuration] [term] mJ=[mF]
                when I=0, F = I + J = J, so mF==mJ. Often, the F nomenclature is then ignored
            - [configuration] [term] F=[F]
            - [configuration] [term]

        :param key: the key to get
        :return: the level associated with the key
        """
        if isinstance(key, int) or isinstance(key, slice):
            return sorted(nx.get_node_attributes(self.model, 'level').values(), key=lambda x: x.level_Hz)[key]
        if key in self.aliases:
            return self.aliases[key]
        if 'mF=' in key:  # match [conf] [term] F=[f] mF=[mF]
            main, hf, z = key.rsplit(' ', maxsplit=2)
            return nx.get_node_attributes(self.model, 'level')[main][hf][z]
        elif 'mJ=' in key:  # match [conf] [term] mJ=[mJ] --- only applies when I=0
            main, z = key.rsplit(' mJ=', maxsplit=1)
            l = nx.get_node_attributes(self.model, 'level')[main]
            return l[f'F={l.term.J_frac}'][f'mF={z}']
        elif 'F=' in key:  # match [conf] [term] F=[F]
            main, hf = key.rsplit(' ', maxsplit=1)
            return nx.get_node_attributes(self.model, 'level')[main][hf]
        else:  # match [conf] [term]
            return nx.get_node_attributes(self.model, 'level')[key]

    def __setitem__(self, key: str, value: EnergyLevel):
        self.atom.add_level(value, alias=key)

    def __delitem__(self, key: str):
        self.model.remove_node(key)

    def __len__(self) -> int:
        return len(nx.get_node_attributes(self.model, 'level'))

    def __iter__(self) -> Iterator:
        return nx.get_node_attributes(self.model, 'level').__iter__()

    def keys(self) -> KeysView:
        return nx.get_node_attributes(self.model, 'level').keys()

    def values(self) -> ValuesView:
        return nx.get_node_attributes(self.model, 'level').values()

    def levels(self) -> ValuesView:
        return self.values()

    def append(self, value: EnergyLevel):
        self.atom.add_level(value)

    def list_names(self) -> List[str]:
        return list(self.keys())


class TransitionStructure:
    """
    A TransitionStructure provides a dict-like interface to the *edges* of an Atom's internal networkx Graph.
    It supports access to transitions by various equivalent keys, as well as by user-assigned aliases
    """
    def __init__(self, atom: Atom, model: nx.Graph, hf_model: nx.Graph, z_model: nx.Graph):
        self.atom = atom
        self.model = model
        self.hf_model = hf_model
        self.z_model = z_model
        self.aliases = dict()

    def __repr__(self):
        return f'TransitionStructure containing {len(self)} transitions of type {type(list(self.values())[0]).__name__}'

    def __getitem__(self, key: Tuple[str, str] or Tuple[int, int]) -> BaseTransition:
        """
        Parses the key to return the associated transition. Supports various kinds of access. Access by:
            - Tuple[int, int]:
                 (n, m) returns the transition between the nth-lowest-lying level and the mth-lowest-lying level
            - alias
            - ([conf] [term] F=[F] mF=[mF], [conf] [term] F=[F] mF=[mF])
            - ([conf] [term] mJ=[mJ], [conf] [term] mJ=[mj])
            - ([conf] [term] F=[F], [conf] [term] F=[F])
            - ([conf] [term], [conf] [term])

        :param key: the key to get
        :return: the transition associated with the key
        """
        if key in self.aliases.keys():
            return self.aliases[key]
        # Regardless of the order the keys are passed, internally they will always be processed as (lower, upper)
        if self.atom.levels[key[0]].level > self.atom.levels[key[1]].level:
            k_lower, k_upper = key[1], key[0]
        else:
            k_lower, k_upper = key[0], key[1]
        if 'mF=' in key[0]:  # match [conf] [term] F=[f] mF=[mF]
            main_0, hf_0, z_0 = k_lower.rsplit(' ', maxsplit=2)
            main_1, hf_1, z_1 = k_upper.rsplit(' ', maxsplit=2)
            return nx.get_edge_attributes(self.model, 'transition')[(main_0, main_1)][(hf_0, hf_1)][(z_0, z_1)]
        elif 'mJ=' in key[0]:
            main_0, z_0 = k_lower.rsplit(' mJ=', maxsplit=1)
            main_1, z_1 = k_upper.rsplit(' mJ=', maxsplit=1)
            l_0 = nx.get_node_attributes(self.model, 'level')[main_0]
            l_1 = nx.get_node_attributes(self.model, 'level')[main_1]
            return nx.get_edge_attributes(self.model, 'transition')[(main_0, main_1)][
                (f'F={l_0.term.J_frac}', f'F={l_1.term.J_frac}')][
                (f'mF={z_0}', f'mF={z_1}')]
        elif 'F=' in key[0]:  # match [conf] [term] F=[f]
            main_0, hf_0 = k_lower.rsplit(' ', maxsplit=1)
            main_1, hf_1 = k_upper.rsplit(' ', maxsplit=1)
            return nx.get_edge_attributes(self.model, 'transition')[(main_0, main_1)][(hf_0, hf_1)]
        else:  # match [conf] [term]
            return nx.get_edge_attributes(self.model, 'transition')[(k_lower, k_upper)]

    def __setitem__(self, value: Transition):
        self.atom.add_transition(value)

    def __delitem__(self, level1: str, level2: str):
        self.model.remove_edge(level1, level2)

    def __len__(self) -> int:
        return len(nx.get_edge_attributes(self.model, 'transition'))

    def __iter__(self) -> Iterator:
        return nx.get_edge_attributes(self.model, 'transition').__iter__()

    def keys(self) -> KeysView:
        return nx.get_edge_attributes(self.model, 'transition').keys()

    def values(self) -> ValuesView:
        return nx.get_edge_attributes(self.model, 'transition').values()

    def append(self, value: Transition):
        self.atom.add_transition(value)

    def list_names(self, hide_self_transitions: bool = True):
        """
        lists the names of all the transitions in the structure
        :param hide_self_transitions: if True, transitions between a level and itself will be ignored
            (for instance, hyperfine transitions within a level)
        :return: a list of level names
        """
        if hide_self_transitions:
            return [key for key in self.keys() if key[0] != key[1]]
        return list(self.keys())

# CONSIDER: search_levels? search_transitions?
