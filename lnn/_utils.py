##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import time
import logging
from typing import Union, TypeVar, Tuple, List, Iterable

from . import _exceptions
from .constants import World, Fact, _Fact

import torch
import numpy as np


def fact_to_bounds(
    fact: Union[Fact, World],
    propositional: bool,
    dims: list = None,
    requires_grad: bool = False,
    init_empty: bool = False,
) -> torch.Tensor:
    """Create classical bounds tensor

    bounds tensor repeats over dims according to (batch, groundings)
    None facts assumes `Unknown`
        tuple facts required in bounds form `(Lower, Upper)`
        All facts are converted to bounds, see
            [F2 - table 3](https://arxiv.org/pdf/2006.13155.pdf) for
            additional description of how truths and bounds are mapped

    Parameters
    ------------
    fact: Fact
    dims: tuple
        repeats over optional (batch, groundings)
    propositional: bool

    Examples
    --------
    fact_to_bounds(UNKNOWN, dims=[5])
    fact_to_bounds(UNKNOWN, dims=[2, 5])

    """
    if isinstance(fact, torch.Tensor):
        return fact
    sizes = dims + [1] if dims else ([0, 1] if init_empty and not propositional else 1)
    if isinstance(fact, tuple):
        _exceptions.AssertBoundsLen(fact)
    elif type(fact) in [World, Fact]:
        fact = fact.value
    return (
        torch.tensor(
            tuple(map(float, fact)) if isinstance(fact, tuple) else (fact, fact)
        )
        .repeat(sizes)
        .requires_grad_(requires_grad)
    )


def negate_bounds(
    bounds: Union[torch.Tensor, Tuple[float, float]], dim=-1
) -> Union[torch.Tensor, Tuple[float, float]]:
    """Negates a bounds tensor: (1 - U, 1 - L)"""
    if isinstance(bounds, tuple):
        return tuple((1 - torch.as_tensor(bounds)).flip(dim).tolist())
    return (1 - bounds).flip(dim)


def node_state(state: np.ScalarType):
    result = {
        "U": Fact.UNKNOWN,
        "T": Fact.TRUE,
        "F": Fact.FALSE,
        "C": Fact.CONTRADICTION,
        "~F": _Fact.APPROX_FALSE,
        "~U": _Fact.APPROX_UNKNOWN,
        "=U": _Fact.EXACT_UNKNOWN,
        "~T": _Fact.APPROX_TRUE,
    }
    return result[state.item()]


def is_classical_world(world: Tuple):
    """returns True if world is TRUE, FALSE, or CONTRADICTION"""
    _exceptions.AssertTupledBounds(world)
    _exceptions.AssertBoundsLen(world)
    options = [w.value for w in [World.AXIOM, World.CONTRADICTION, World.FALSE]]
    return world in options


def is_classical_proposition(fact: Tuple):
    """returns True if fact is TRUE, FALSE, or CONTRADICTION"""
    return is_classical_world(fact)


def dict_rekey(d, old_key, new_key):
    d[new_key] = d.pop(old_key)


param_symbols = {
    "alpha": "α",
    "bias": "β",
    "weights": "w",
    "weights.grad": "w.g",
    "bias.grad": "β.g",
}

Model = TypeVar("Model")


def val_clamp(x, _min: float = 0, _max: float = 1) -> torch.Tensor:
    """gradient-transparent clamping to clamp values between [min, max]"""
    clamp_min = (x.detach() - _min).clamp(max=0)
    clamp_max = (x.detach() - _max).clamp(min=0)
    return x - clamp_max - clamp_min


def list_to_str(my_list: Iterable, my_join: str = ", ") -> str:
    return my_join.join([str(v) for v in my_list])


checkpoints = []


def reset_checkpoints():
    global checkpoints
    checkpoints = []


def unpack_checkpoints():
    result = list()
    for t in range(len(checkpoints) - 1):
        result.append(
            (f"{checkpoints[t+1][1]}", checkpoints[t + 1][0] - checkpoints[t][0])
        )
    return result


def add_checkpoint(label):
    checkpoints.append((time.time(), label))


def average_time(multi_checkpoint: List[List[Tuple]]):
    result = dict()
    for name in multi_checkpoint[0]:
        result[name] = 0

    for c in multi_checkpoint:
        for name, point in enumerate(c):
            result[name] += point


def logger_setup(flush=False):
    for level in ["INFO"]:
        filename = f"LNN_{level}.log"
        if flush:
            with open(filename, "w"):
                pass
        logging.basicConfig(
            filename=filename, encoding="utf-8", level=eval(f"logging.{level}")
        )
