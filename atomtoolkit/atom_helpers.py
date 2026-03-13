from __future__ import annotations
from typing import *

import networkx as nx

if TYPE_CHECKING:
    from atomtoolkit.atom import Atom, BaseLevel, BaseTransition, EnergyLevel, Transition


def _parse_level_key(key: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse a structured level key into its three components.

    Keys follow the pattern ``{main} [F={f}] [mF={mf}]`` where ``{main}`` is
    the NIST configuration + term string.  The HF and Zeeman suffixes are
    delimited by a literal space followed by ``F=`` or ``mF=`` — neither of
    which can appear in a NIST configuration string.

    Returns ``(main, hf_part, z_part)`` where missing parts are ``None``.
    ``hf_part`` is the full ``'F=...'`` token; ``z_part`` is ``'mF=...'``.

    Examples::

        '4f14.6s 2S1/2'            → ('4f14.6s 2S1/2', None,    None)
        '4f14.6s 2S1/2 F=0'        → ('4f14.6s 2S1/2', 'F=0',   None)
        '4f14.6s 2S1/2 F=0 mF=0'  → ('4f14.6s 2S1/2', 'F=0',  'mF=0')
    """
    z_part: Optional[str] = None
    hf_part: Optional[str] = None

    if ' mF=' in key:
        key, mf_val = key.rsplit(' mF=', 1)
        z_part = f'mF={mf_val}'
    if ' F=' in key:
        key, hf_val = key.rsplit(' F=', 1)
        hf_part = f'F={hf_val}'

    return key, hf_part, z_part


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
        levels = nx.get_node_attributes(self.model, 'level')
        # mJ= shorthand is only valid for I=0 atoms (mJ aliases mF)
        if ' mJ=' in key:
            main, mj_val = key.rsplit(' mJ=', 1)
            level = levels[main]
            return level[f'F={level.term.J_frac}'][f'mF={mj_val}']
        main, hf_part, z_part = _parse_level_key(key)
        try:
            level = levels[main]
        except KeyError:
            raise KeyError(
                f"Level {main!r} not found. "
                f"(Parsed from key {key!r}: main={main!r}, hf={hf_part!r}, z={z_part!r})"
            ) from None
        if hf_part is not None:
            level = level[hf_part]
        if z_part is not None:
            level = level[z_part]
        return level

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
        edges = nx.get_edge_attributes(self.model, 'transition')
        # mJ= shorthand: handle both keys together
        if ' mJ=' in key[0]:
            main_0, mj_0 = k_lower.rsplit(' mJ=', 1)
            main_1, mj_1 = k_upper.rsplit(' mJ=', 1)
            levels = nx.get_node_attributes(self.model, 'level')
            l_0, l_1 = levels[main_0], levels[main_1]
            return edges[(main_0, main_1)][
                (f'F={l_0.term.J_frac}', f'F={l_1.term.J_frac}')][
                (f'mF={mj_0}', f'mF={mj_1}')]
        main_0, hf_0, z_0 = _parse_level_key(k_lower)
        main_1, hf_1, z_1 = _parse_level_key(k_upper)
        try:
            t = edges[(main_0, main_1)]
        except KeyError:
            return None
        if hf_0 is not None:
            t = t[(hf_0, hf_1)]
        if z_0 is not None:
            t = t[(z_0, z_1)]
        return t

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

    def items(self) -> ItemsView:
        return nx.get_edge_attributes(self.model, 'transition').items()

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
