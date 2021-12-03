##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from .constants import Fact, World, Direction

import torch
from typing import Set, Tuple, Dict


class AssertWorld:
    """AssertWorld(bounds)

    Raised when world not given as World object
    """
    def __init__(self, world: World):
        if world not in World:
            raise KeyError(
                'truth world assumption expects lnn.World object from '
                f'{[w.name for w in World]}, received '
                f'{world.__class__.__name__} {world}')


class AssertBoundsBroadcasting:
    """AssertBoundsBroadcasting(bounds)

    Raised when FOL bounds given as set of more than 1 item
    """
    def __init__(self, bounds: Set):
        if isinstance(bounds, set):
            if len(bounds) != 1:
                raise IndexError('broadcasting facts expects a set of 1 item')


class AssertBoundsType:
    """Raised when bounds given in the incorrect type"""

    def __init__(self, bounds):
        options = [Fact, tuple]
        if type(bounds) not in options:
            raise TypeError(
                f'fact expected from [lnn.Fact, tuple] '
                f'received {bounds.__class__.__name__} {bounds}')


class AssertBoundsLen:
    """AssertBoundsLen(bounds)

    Raised when tuple of bounds given in the incorrect length
    """
    def __init__(self, bounds):
        if isinstance(bounds, tuple):
            if len(bounds) != 2:
                raise IndexError(
                    'bounds tuple expected to have 2 bounds (Lower, Upper), '
                    f'received {bounds}')


class AssertBoundsInputs:
    """AssertBounds(bounds)

    Raised when incorrect bounds given
    """
    def __init__(self, bounds: Tuple[torch.Tensor, ...]):
        if isinstance(bounds, tuple):
            if any([not 0 <= b <= 1 for b in bounds]):
                raise IndexError(
                    'bounds expected to be in range [0, 1], '
                    f'received {bounds}')


class AssertBounds:
    """AssertBounds(bounds)

    Raised when tuple of bounds given in the incorrect length
    """
    def __init__(self, bounds):
        AssertBoundsType(bounds)
        AssertBoundsLen(bounds)
        AssertBoundsInputs(bounds)
        AssertBoundsBroadcasting(bounds)


class AssertPropositionalInheritance:
    def __init__(self, node):
        if not hasattr(node, 'propositional'):
            raise Exception("should not end up here, propositional variable "
                            "expected to be declared at proposition/predicate"
                            "and inherited appropriately... you may have an"
                            "incorrectly connected the node")


class AssertFormulaInModel:
    def __init__(self, model, formula):
        if formula not in model:
            raise Exception(
                f"{formula} is not a stored formula, can't set facts")


class AssertGroundingKeyType:
    """AssertGroundingKeyType(facts)

    Raised when fact keys are not valid groundings
    """
    def __init__(self, facts):
        if isinstance(facts, dict):
            if all([type(f) not in [tuple, Fact] for f in facts.keys()]):
                raise TypeError(
                    'fact keys expected as str or tuple of str'
                    f'received {facts}')


class AssertFOLFacts:
    """AssertGroundedBounds(bounds)

    Raised when FOL bounds expected as a dict of {groundings: facts}

    """
    def __init__(self, facts: Dict):
        if isinstance(facts, dict):
            for grounding, bounds in facts.items():
                AssertGroundingKeyType(grounding)
                AssertBounds(bounds)


class AssertDirection:
    """AssertDirection(direction: Direction)

    Raised when direction input is not valid
    """
    def __init__(self, direction: Direction):
        AssertDirectionType(direction)
        AssertValidDirection(direction)


class AssertValidDirection:
    """AssertValidDirection(direction: Direction)

    Raised when direction not upward/downward
    """
    def __init__(self, direction: Direction):
        options = [Direction.UPWARD, Direction.DOWNWARD]
        if direction not in options:
            raise KeyError(
                f'direction expected from {options}, '
                f'found {direction}')


class AssertDirectionType:
    """AssertDirectionType(direction: Direction)

    Raised when direction not a clarified str
    """
    def __init__(self, direction: Direction):
        if not isinstance(direction, Direction):
            raise TypeError(f'direction expected as Direction, '
                            f'received {direction}')


class AssertBias:
    """AssertDirectionType(direction: Direction)

    Raised when direction not a clarified str
    """
    def __init__(self, bias):
        if not isinstance(bias, float):
            raise TypeError(
                f'bias expected as a float, received {type(bias)}: {bias}')


class AssertWeights:
    """AssertDirectionType(direction: Direction)

    Raised when direction not a clarified str
    """
    def __init__(self, weights: Tuple, arity: int):
        if not isinstance(weights, tuple):
            raise TypeError(
                'weights expected as a tuple of floats, received '
                f'{type(weights)}: {weights}')
        if not len(weights) == arity:
            raise ValueError(
                f'weights expected as len {arity}, received {len(weights)}')


class AssertAlphaNodeValue:
    """AssertAlphaInitValue(alpha: torch.Tensor)

    Raised when alpha not in range
    """

    def __init__(self, alpha):
        if not (.5 < alpha <= 1):
            raise ValueError(
                f'alpha expected between (.5, 1], received {alpha}')


class AssertAlphaNeuronArityValue:
    """AssertAlphaInitValue(arity: int)

    Raised when alpha not in range
    """

    def __init__(self, alpha, arity):
        constraint = arity/(arity+1)
        if not (alpha >= constraint):
            raise ValueError(
                f'alpha expected greater than n/(n+1) ({constraint:<.3e}) '
                f'for n={arity}, received {alpha:<3e}')
