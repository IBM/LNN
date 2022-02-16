##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import random
import itertools
from typing import Union, Tuple, List, TypeVar

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from . import _utils
from .constants import Fact


TRUE = Fact.TRUE
FALSE = Fact.FALSE


def truth_table(n: int, states=None) -> List[Tuple[Fact, ...]]:
    if states is None:
        states = [FALSE, TRUE]
    return list(itertools.product(states, repeat=n))


def truth_table_dict(*args: str, states=None):
    if states is None:
        states = [FALSE, TRUE]
    for instance in itertools.product(states, repeat=len(args)):
        yield dict(zip(args, instance))


def fact_to_bool(*fact: Fact) -> Union[Fact, bool, Tuple[bool, ...]]:
    if len(fact) > 1:
        return tuple(map(fact_to_bool, fact))
    if fact[0] is TRUE:
        return True
    elif fact[0] is FALSE:
        return False
    else:
        return fact[0]


def bool_to_fact(*truth: bool) -> Union[Fact, Tuple[Fact, ...]]:
    if len(truth) > 1:
        return tuple(map(bool_to_fact, truth))
    return TRUE if truth[0] else FALSE


def predicate_truth_table(*args: str, arity: int, model, states=None):
    """
    predicate_truth_table('p', 'q', 'r', model=model)

    randomises the truth table into a predicate by str(int) rows

    **Returns**

    model
    """
    if states is None:
        states = [FALSE, TRUE]
    from lnn import Predicate  # noqa: F401
    n = len(args)
    TT = np.array(truth_table(n, states))
    _range = list(range(len(TT)))
    for idx, arg in enumerate(args):
        model[arg] = Predicate(arity=arity)
        random.shuffle(_range)
        for i in _range:
            grounding = f'{i}' if arity == 1 else (f'{i}',)*arity
            truth = TT[i, idx].item()
            model[arg].add_facts({grounding: truth})
    return model


def plot_graph(self, **kwds) -> None:
    options = {
        'with_labels': True,
        'arrows': True,
        'edge_color': '#d0e2ff',
        'node_size': 1,
        'font_size': 9,
    }
    options.update(kwds)
    pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
    nx.draw(self.graph, pos, **options)
    plt.show()


def plot_loss(total_loss, losses) -> None:
    loss, cummulative_loss = total_loss
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Model Loss')
    axs[0].plot(np.array(loss))
    for ax in axs.flat:
        ax.set(xlabel='Epochs', ylabel='Loss')
    axs[0].legend(['Total Loss'])
    axs[1].plot(np.array(cummulative_loss))
    axs[1].legend([str.capitalize(i) for i in losses])
    plt.show()


Model = TypeVar('Model')


def plot_params(self: Model) -> None:
    legend = []
    for node in self.nodes:
        if hasattr(self[node], 'parameter_history'):
            for param, data in self[node].parameter_history.items():
                if isinstance(data[0], list):
                    operands = list(self[node].operands)
                    legend_idxs = [
                        f'_{operands[i]}' for i in list(range(len(data[0])))]
                else:
                    legend_idxs = ['']
                [legend.append(
                    f'{node} {_utils.param_symbols[param]}{i}')
                    for i in legend_idxs]
                plt.plot(data)
    plt.xlabel('Epochs')
    plt.legend(legend)
    plt.title(f'{self.name} Parameters')
    plt.show()
