##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from typing import Union, Tuple, List, TypeVar, Iterable

from . import _utils
from .constants import Fact
from lnn.symbolic.logic.leaf_formula import Predicate

import re
import random
import itertools
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

TRUE = Fact.TRUE
FALSE = Fact.FALSE


def split_string_into_groundings(state: str) -> Tuple[str]:
    r"""

    Parameters
    ----------
    :param state: The state is given as a string value representing the groundings; i.e "('T','T')"

    Examples
    --------
    P,Q = Predicate("P","Q")
    x,y = Variables("x","y")
    PQ  = Or(P(x), Q(y))
    PQ.upward()
    PQ.state(("T","T")) returns Fact.TRUE

    Returns
    -------
    Groundings, given as strings, as a tuple of strings ; i.e "('T','T')" -> ("T","T")
    """
    pattern = r'[\'"()]'
    grounding_strings = re.sub(pattern, "", state)
    partial_groundings = grounding_strings.split(",")
    partial_groundings = [pg.strip() for pg in partial_groundings]
    return tuple(partial_groundings)


def get_binary_truth_table(formulae: dict) -> dict:
    r"""

    Parameters
    ----------
    :param formulae: Formulae is a dictionary with str:formula format ;i.e {"P or Q": <lnn.symbolic.logic.n_ary_neuron >}

    Returns
    -------
    Truth table in a dictionary format

    """
    table = dict()
    groundings = set()
    for f in formulae.values():
        groundings.update(f.groundings)
    groundings = alphanumeric_sort(groundings)
    table[""] = [g for g in groundings]
    for name, formula in formulae.items():
        table[name] = [formula.state(g).name for g in groundings]
    return table


def alphanumeric_sort(iterable: Iterable):
    def get_int(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [get_int(c) for c in re.split("([0-9]+)", key)]

    return sorted(iterable, key=alphanum_key)


def get_ternary_table(formulae: dict, unique_groundings: List[str] = None) -> dict:
    keys = list(list(formulae.values())[0].state().keys())
    formula = list(formulae.keys())[0]
    keys = [split_string_into_groundings(key) for key in keys]
    operator = list(formulae.values())[0]
    if unique_groundings is None:
        # unique_groundings = sorted(list(set(itertools.chain.from_iterable(keys))))
        unique_groundings = ["F", "U", "T"]
    grounding_map = dict(zip(range(len(unique_groundings)), unique_groundings))
    number_of_unique_groundings = len(unique_groundings)
    grounding_grid = np.zeros(
        (number_of_unique_groundings, number_of_unique_groundings), dtype="int,int"
    )
    truth_grid = np.array(
        [[""] * number_of_unique_groundings] * number_of_unique_groundings, dtype="<U10"
    )
    for i in range(grounding_grid.shape[0]):
        for j in range(grounding_grid.shape[1]):
            row, column = grounding_map[i], grounding_map[j]
            truth_grid[i, j] = operator.state((row, column)).name
    row_title = np.atleast_2d(unique_groundings).T
    column_data = np.atleast_2d([formula] + unique_groundings)
    truth_grid = np.hstack((row_title, truth_grid))
    truth_grid = np.vstack((column_data, truth_grid))
    return truth_grid


def get_n_ary_truth_table(formulae: dict, unique_groundings: List[str] = None) -> dict:
    r"""

    Parameters
    ----------
    :param formulae: Formulae is a dictionary with str:formula format ;i.e {"P or Q": <lnn.symbolic.logic.n_ary_neuron >}
    :param unique_groundings: Unique groundings that fill either the rows or columns i.e ["F", "U", "T"]

    Returns
    -------
    Truth table in a dictionary format

    """
    keys = list(list(formulae.values())[0].state().keys())
    formula = list(formulae.keys())[0]
    keys = [split_string_into_groundings(key) for key in keys]
    if unique_groundings is None:
        unique_groundings = sorted(list(set(itertools.chain.from_iterable(keys))))
    truth = {f"{formula}": unique_groundings}
    for unique_grounding in unique_groundings:
        table_columns = list(
            filter(
                lambda partial_grounding: partial_grounding[0].startswith(
                    f"{unique_grounding}"
                ),
                keys,
            )
        )
        for _, formula in formulae.items():
            truth[unique_grounding] = [
                formula.state(table_column).name for table_column in table_columns
            ]
    return truth


def pretty_truth_table(formulae: dict, unique_groundings: List[str] = None) -> None:
    r"""

    Parameters
    ----------
    :param formulae: Formulae is a dictionary with str:formula format ;i.e {"P or Q": <lnn.symbolic.logic.n_ary_neuron >}
    :param arity: The number of variables specified
    :param unique_groundings: unique_groundings: Unique groundings that fill either the rows or columns i.e ["F", "U", "T"]

    Returns
    -------
    A pretty truth table
    """
    keys = list(list(formulae.values())[0].state().keys())
    if len(keys[0]) <= 2:
        table = get_binary_truth_table(formulae)
        print(tabulate(table, headers="keys", tablefmt="fancy_grid"))
    else:
        table = get_ternary_table(formulae, unique_groundings)
        print(tabulate(table, tablefmt="fancy_grid"))


def generate_truth_table(P: Predicate, Q: Predicate, states=None) -> None:
    r"""

    Parameters
    ----------
    P: Predicate P
    Q: Predicate Q
    states: Data that you would like to each of the predicates

    Returns
    -------
    Nothing, causes side effect on P and Q
    """
    if states is None:
        states = [Fact.FALSE, Fact.UNKNOWN, Fact.TRUE]
    idx = [f"{i}" for i in range(len(states))]
    data = dict(zip(idx, states))
    P.add_data(data)
    Q.add_data(data)


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
    predicate_truth_table("p", "q", "r", model=model)

    randomises the truth table into a predicate by str(int) rows

    Returns
    -------
    model : Model
    """
    if states is None:
        states = [FALSE, TRUE]
    from lnn import Predicate  # noqa: F401

    n = len(args)
    TT = np.array(truth_table(n, states))
    _range = list(range(len(TT)))
    for idx, arg in enumerate(args):
        model[arg] = Predicate(arg, arity=arity)
        random.shuffle(_range)
        for i in _range:
            grounding = f"{i}" if arity == 1 else (f"{i}",) * arity
            truth = TT[i, idx].item()
            model[arg].add_data({grounding: truth})
    return model


def plot_loss(total_loss, losses):
    loss, cummulative_loss = total_loss
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Model Loss")
    axs[0].plot(np.array(loss))
    for ax in axs.flat:
        ax.set(xlabel="Epochs", ylabel="Loss")
    axs[0].legend(["Total Loss"])
    axs[1].plot(np.array(cummulative_loss))
    axs[1].legend([loss.value.capitalize() for loss in losses])
    plt.show()


Model = TypeVar("Model")


def plot_params(self: Model):
    legend = []
    for node in self.nodes.values():
        if hasattr(node, "parameter_history"):
            for param, data in node.parameter_history.items():
                if isinstance(data[0], list):
                    operands = list(node.operands)
                    legend_idxs = [f"_{operands[i]}" for i in list(range(len(data[0])))]
                else:
                    legend_idxs = [""]
                [
                    legend.append(f"{node.name} {_utils.param_symbols[param]}{i}")
                    for i in legend_idxs
                ]
                plt.plot(data)
    plt.xlabel("Epochs")
    plt.legend(legend)
    plt.title(f"{self.name} Parameters")
    plt.show()


def return1(args: Union[List, Tuple]):
    if len(args) == 1:
        return args[0]
    return args
