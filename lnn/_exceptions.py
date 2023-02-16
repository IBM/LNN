##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from .constants import Fact, World, Direction, NeuralActivation, Loss

import torch
from typing import Set, Tuple, Dict, Union


class AssertWorld:
    """AssertWorld(world: World)

    Raised when world not given as World object
    """

    def __init__(self, world: World):
        if world not in World:
            raise KeyError(
                "truth world assumption expects lnn.World object from "
                f"{[w.name for w in World]}, received "
                f"{world.__class__.__name__} {world}"
            )


class AssertBoundsBroadcasting:
    r"""AssertBoundsBroadcasting(bounds: Set).

    Raised when FOL bounds given as set of more than 1 item
    """

    def __init__(self, bounds: Set):
        if isinstance(bounds, set):
            if len(bounds) != 1:
                raise IndexError("broadcasting facts expects a set of 1 item")


class AssertBoundsType:
    r"""AssertBoundsType(bounds: Union[Fact, tuple])

    Raised when bounds given in the incorrect type.
    """

    def __init__(self, bounds: Union[Fact, tuple, torch.Tensor]):
        options = [bool, Fact, World, tuple, torch.Tensor, float]
        if type(bounds) not in options:
            raise TypeError(
                f"fact expected from [lnn.Fact, lnn.World, tuple, torch.Tensor] "
                f"received {bounds.__class__.__name__} {bounds}"
            )


class AssertTupledBounds:
    r"""Raised when bounds given in the incorrect type."""

    def __init__(self, bounds):
        if not isinstance(bounds, tuple):
            raise TypeError(
                f"input expected as tuple, "
                f"received {bounds.__class__.__name__} {bounds}"
            )


class AssertBoundsLen:
    r"""AssertBoundsLen(bounds: tuple)

    Raised when tuple of bounds given in the incorrect length
    """

    def __init__(self, bounds: tuple):
        if isinstance(bounds, tuple):
            if len(bounds) != 2:
                raise IndexError(
                    "bounds tuple expected to have 2 bounds (Lower, Upper), "
                    f"received {bounds}"
                )


class AssertBoundsInputs:
    r"""AssertBoundsInputs: Tuple[torch.Tensor, ...]

    Raised when incorrect bounds given
    """

    def __init__(self, bounds: Tuple[torch.Tensor, ...]):
        if isinstance(bounds, tuple):
            if any([not 0 <= b <= 1 for b in bounds]):
                raise IndexError(
                    "bounds expected to be in range [0, 1], " f"received {bounds}"
                )


class AssertBounds:
    r"""AssertBounds(bounds)

    Raised when tuple of bounds given in the incorrect length
    """

    def __init__(self, bounds):
        AssertBoundsType(bounds)
        AssertBoundsLen(bounds)
        AssertBoundsInputs(bounds)
        AssertBoundsBroadcasting(bounds)


class AssertPropositionalInheritance:
    """AssertPropositionalInheritance(node)

    Raise when node does not have propositional attribute
    """

    def __init__(self, node):
        if not hasattr(node, "propositional"):
            raise Exception(
                "should not end up here, propositional variable "
                "expected to be declared at proposition/predicate"
                "and inherited appropriately... you may have an"
                "incorrectly connected the node"
            )


class AssertPropositionalType:
    def __init__(self, propositional):
        if not isinstance(propositional, bool):
            raise TypeError(
                "propositional expected as bool, received "
                f"{propositional.__class__.__name__}"
            )


class AssertFormulaInModel:
    """AssertFormulaInModel(model: lnn.Model, formula: lnn.symbolic.logic.Formula)

    Raised when formula is not in the model
    """

    def __init__(self, model, formula):
        if formula not in model:
            raise Exception(f"{formula} is not a stored formula, can't set facts")


class AssertFormula:
    def __init__(self, formula):
        if isinstance(formula, tuple):
            raise Exception(
                f"expected formula, received a called formula "
                f"'{formula[0]}({formula[-1]})', "
                f"try removing the variables and the parenthesis"
            )


class AssertGroundingKeyType:
    r"""AssertGroundingKeyType(facts: dict)

    Raised when fact keys are not valid groundings
    """

    def __init__(self, facts: dict):
        if isinstance(facts, dict):
            if all([type(f) not in [tuple, Fact] for f in facts.keys()]):
                raise TypeError(
                    "fact keys expected as str or tuple of str" f"received {facts}"
                )


class AssertBindingsInputType:
    r"""AssertBindingsInputType(bindings)

    Raised when binding inputs are not valid as able, Union[str, List[str]]]
    """

    def __init__(self, bindings):
        for binding in bindings:
            if not isinstance(binding, str) or (
                isinstance(binding, list) and all([isinstance(b, str) for b in binding])
            ):
                raise TypeError(
                    f"bindings expected as str or list of str, received {type(binding)}"
                )


class AssertFOLFacts:
    r"""AssertFOLFacts(bounds: dict)

    Raised when FOL bounds expected as a dict of {groundings: facts}

    """

    def __init__(self, facts: Dict):
        if isinstance(facts, dict):
            for grounding, bounds in facts.items():
                AssertGroundingKeyType(grounding)
                AssertBounds(bounds)


class AssertDirection:
    r"""AssertDirection(direction: Direction)

    Raised when direction input is not valid
    """

    def __init__(self, direction: Direction):
        AssertDirectionType(direction)
        AssertValidDirection(direction)


class AssertValidDirection:
    r"""AssertValidDirection(direction: Direction)

    Raised when direction not upward/downward
    """

    def __init__(self, direction: Direction):
        if not isinstance(direction, Direction):
            raise KeyError(
                f"direction expected from class 'Direction', " f"found {direction}"
            )


class AssertDirectionType:
    r"""AssertDirectionType(direction: Direction)

    Raised when direction not a clarified str
    """

    def __init__(self, direction: Direction):
        if not isinstance(direction, Direction):
            raise TypeError(
                f"direction expected from lnn.Direction, " f"received {direction}"
            )


class AssertNeuronActivationType:
    r"""AssertNeuronActivationType(_type: NeuralActivation)

    Raised when direction not a clarified str
    """

    def __init__(self, _type: NeuralActivation):
        if not isinstance(_type, NeuralActivation):
            raise TypeError(
                f"Neural activation expected from lnn.NeuralActivation, "
                f"received {_type}"
            )


class AssertLossType:
    r"""AssertLossType(_type: Loss)

    Raised when direction not a clarified str
    """

    def __init__(self, _type: Loss):
        if not isinstance(_type, Loss):
            raise TypeError(f"loss expected from lnn.Loss, " f"received {_type}")


class AssertBias:
    """AssertBias(direction: float)

    Raised when direction not a clarified str
    """

    def __init__(self, bias):
        if not isinstance(bias, float):
            raise TypeError(f"bias expected as a float, received {type(bias)}: {bias}")


class AssertWeights:
    """AssertWeights(weights: Tuple, arity: int)

    Raised when direction not a clarified str
    """

    def __init__(self, weights: Tuple, arity: int):
        if not isinstance(weights, tuple):
            raise TypeError(
                "weights expected as a tuple of floats, received "
                f"{type(weights)}: {weights}"
            )
        if not len(weights) == arity:
            raise ValueError(
                f"weights expected as len {arity}, received {len(weights)}"
            )


class AssertAlphaNodeValue:
    """AssertAlphaNodeValue(alpha: torch.Tensor)

    Raised when alpha not in range
    """

    def __init__(self, alpha):
        if not (0.5 < alpha <= 1):
            raise ValueError(f"alpha expected between (.5, 1], received {alpha}")


class AssertAlphaNeuronArityValue:
    """AssertAlphaNeuronArityValue(alpha, arity: int)

    Raised when alpha not in range
    """

    def __init__(self, alpha, arity):
        constraint = arity / (arity + 1)
        if not (alpha >= constraint):
            raise ValueError(
                f"alpha expected greater than n/(n+1) ({constraint:<.3e}) "
                f"for n={arity}, received {alpha:<3e}"
            )


class AssertLeafFormulaNaming:
    """AssertLeafFormulaNaming(formula: Formula, name: str, isLeaf: bool)

    Raised when alpha not in range
    """

    def __init__(self, formula, name, is_leaf):
        if not is_leaf:
            raise ValueError(
                "Only Propositions and Predicates can be named, received "
                f"{name} for {formula.__class__.__name__}"
            )
