##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import time
from typing import Union, TypeVar, Tuple

import torch
import torchviz
import numpy as np

from . import _exceptions
from .constants import World, Fact, _Fact


class MultiInstance:
    def __init__(self, term: Union[str, Tuple[str, ...]]):
        if isinstance(term, tuple):
            # Decompose Tuple to create an instance for each in the Tuple
            # This allows _Grounding(('x', 'y')) to create individual
            # groundings, i.e. _Grounding('x'), _Grounding('y')
            [self.__class__(t) for t in term]
        elif isinstance(term, str):
            pass
        else:
            raise Exception(
                f"expected {self.__class__.__name__} inputs from "
                f"[str, list], received {type(term)}"
            )


class UniqueNameAssumption:
    """Class to store and retrieve instances using the unique name assumption

    if an instance is created with the same args, kwargs as an existing instance
        the original object is returned

    **Example**

    ```
    class MyClass(UniqueNameAssumption):
    ```
    """

    instances = dict()

    @classmethod
    def __getitem__(cls, key: str):
        if key in cls.instances:
            return cls.instances[key]
        raise LookupError(
            f"should not end up here, {type(key)} key {key} not found in "
            f"Groundings {list(cls.instances.keys())}"
        )

    @classmethod
    def __setitem__(cls, key: str, val: any):
        cls.instances[key] = val

    def __new__(cls, *args, **kwargs):
        unique_name = str(*args, **kwargs)
        instance = cls.instances.get(unique_name)
        if instance is None:
            instance = super(cls.__class__, cls).__new__(cls)
            cls.__setitem__(unique_name, instance)
        return instance

    @classmethod
    def keys(cls):
        return cls.instances.keys()

    @classmethod
    def values(cls):
        return cls.instances.values()

    @classmethod
    def __len__(cls):
        return len(cls.instances)

    @classmethod
    def items(cls):
        return cls.instances.items()

    @classmethod
    def rekey(cls, new_key: str, old_key: str):
        cls.instances[new_key] = cls.instances.pop(old_key)

    @classmethod
    def pop(cls, key: str):
        return cls.instances.pop(key)

    @classmethod
    def clear(cls):
        cls.instances = dict()

    @classmethod
    def __contains__(cls, key: str):
        return key in cls.instances


def fact_to_bounds(
    fact: Union[Fact, World],
    propositional: bool,
    dims: list = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create classical bounds tensor

    bounds tensor repeats over dims according to (batch, groundings)
    None facts assumes `Unknown`
        tuple facts required in bounds form `(Lower, Upper)`
        All facts are converted to bounds, see
            [F2 - table 3](https://arxiv.org/pdf/2006.13155.pdf) for
            additional description of how truths and bounds are mapped

    **Parameters**

    fact: Fact
    dims: tuple
        repeats over optional (batch, groundings)
    propositional: bool

    **Example**

    fact_to_bounds(UNKNOWN, dims=[5])
    fact_to_bounds(UNKNOWN, dims=[2, 5])

    """
    if isinstance(fact, torch.Tensor):
        return fact

    sizes = dims + [1] if dims else (1 if propositional else [1, 1])
    if isinstance(fact, tuple):
        _exceptions.AssertBoundsLen(fact)
    elif type(fact) in [World, Fact]:
        fact = fact.value
    return (
        torch.tensor(tuple(map(float, fact)))
        .repeat(sizes)
        .requires_grad_(requires_grad)
    )


def negate_bounds(bounds: torch.Tensor, dim=-1) -> torch.Tensor:
    """Negate a bounds tensor: (1 - U, 1 - L)"""
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


def dict_rekey(d, old_key, new_key) -> None:
    d[new_key] = d.pop(old_key)


param_symbols = {
    "alpha": "α",
    "bias": "β",
    "weights": "w",
    "weights.grad": "w.g",
    "bias.grad": "β.g",
}
Model = TypeVar("Model")


def plot_autograd(model: Model, loss: torch.Tensor, **kwargs) -> None:
    params = model.named_parameters()
    torchviz.make_dot(
        loss,
        params=params,
        show_attrs=kwargs.get("show_attrs", True),
        show_saved=kwargs.get("show_saved", True),
    ).render(f'graph_{kwargs.get("epoch", "")}', view=True)


def val_clamp(x, _min: float = 0, _max: float = 1) -> torch.Tensor:
    """gradient-transparent clamping to clamp values between [min, max]"""
    clamp_min = (x.detach() - _min).clamp(max=0)
    clamp_max = (x.detach() - _max).clamp(min=0)
    return x - clamp_max - clamp_min


checkpoints = []


def unpack_checkpoints():
    for t in range(len(checkpoints) - 1):
        print(f"{checkpoints[t+1][1]}: " f"{checkpoints[t+1][0] - checkpoints[t][0]}")


def add_checkpoint(label):
    checkpoints.append((time.time(), label))
