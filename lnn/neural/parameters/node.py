##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import math
from typing import Union, Optional, Tuple, List, Dict, Set

from ... import _utils
from ... import _exceptions
from ...constants import Fact, World

import torch

from torch import nn
from torch.nn.parameter import Parameter


class _NodeParameters(nn.Module):
    """Node level parameters

    Parameters
    ----------
    propositional: bool
    bounds_table: tensor
    world: str
    alpha: torch float scalar
        used for alpha per node formulation

    Notes
    -----
    `bounds_table` follows the following convention:
        [batch, grounding, bounds, arity]

        Propositional:
        1D: [bounds] (proposition)
        2D: [batch, bounds] (batched proposition) `not implemented`

        FOL:
        2D: [grounding, bounds] (predicate)
        3D: [batch, grounding, bounds] (batched predicate) `not implemented`

        Connective inputs (FOL + propositional) are extended on `dim = -1`
    """

    def __init__(self, **kwds):
        super(_NodeParameters, self).__init__()
        propositional = kwds.get("propositional")
        _exceptions.AssertPropositionalType(propositional)
        self.propositional = propositional
        world = kwds.get("world", World.OPEN)
        _exceptions.AssertBounds(world)
        self.update_world(world)

        self.alpha = Parameter(
            torch.tensor(
                kwds.get("alpha", math.erf(kwds.get("alpha_sigma", 10) / math.sqrt(2)))
            ),
            requires_grad=kwds.get("alpha_learning", False),
        )

        _exceptions.AssertAlphaNodeValue(self.alpha)
        self.bounds_learning = kwds.get("bounds_learning", False)
        self.leaves = Parameter(
            _utils.fact_to_bounds(
                self.world,
                self.propositional,
                init_empty=True,
            ),
            requires_grad=self.bounds_learning,
        )
        self.bounds_table = self.leaves.clone()

    def reset_world(self, world: World):
        self.update_world(world)
        if self.bounds_table.shape[0] > 0:
            self.flush(self.world)

    def update_world(self, world: Union[Tuple, World]):
        self.world = world if isinstance(world, tuple) else world.value

    def flush(self, fact=Fact.UNKNOWN):
        if self.bounds_table.shape[0] > 0:
            self.add_data({fact})

    def get_data(self, grounding_rows: List[int] = None, default=False):
        """returns tuple of all facts given by `grounding_rows`
        The `bounds_table` is returned if no `grounding_rows` given
        """
        if self.propositional:
            return self.bounds_table
        if grounding_rows is None:
            return self.bounds_table
        if isinstance(grounding_rows, list):
            return torch.stack(
                [
                    (
                        torch.tensor(self.world)
                        if row is None and default
                        else self.bounds_table[row]
                    )
                    for row in grounding_rows
                ]
            )
        return self.bounds_table[grounding_rows]

    def add_data(self, facts: Union[Fact, Tuple, Set, Dict], update_leaves=True):
        """Populate formula with facts

        Facts given in bool, tuple or None, assumes a propositional formula
        Facts given in dict form assume FOL, keyed by the tensor index, w
            the value also required in bool, Tuple or None

        """
        if self.propositional:  # Propositional facts
            if isinstance(facts, set):
                facts = next(iter(facts))
            self.update_bounds(None, facts, update_leaves)
        else:  # FOL facts
            if isinstance(facts, dict):  # facts given per grounding
                for grounding_row, fact in facts.items():
                    if grounding_row < self.leaves.shape[self._grounding_dims]:
                        self.update_bounds(grounding_row, fact, update_leaves)
                    else:
                        raise Exception("groundings not extended correctly")
            if isinstance(facts, set):  # broadcast facts across groundings
                self.update_bounds(set(), next(iter(facts)), update_leaves)

    def update_bounds(
        self,
        grounding_rows: Optional[Union[int, Set, Dict, None]],
        facts: Union[torch.Tensor, Fact],
        update_leaves=False,
    ):
        """update bounds with facts for given grounding_rows

        if grounding_rows is None, assumes propositional

        """

        def func(grounding_row, fact):
            if update_leaves:
                fact = _utils.fact_to_bounds(
                    fact, self.propositional, requires_grad=self.bounds_learning
                )
                if self.propositional:
                    self.leaves = Parameter(fact, self.bounds_learning)
                    self.bounds_table = fact.clone()
                else:
                    clone = self.leaves.clone()
                    clone[grounding_row] = fact
                    self.leaves = Parameter(clone, self.bounds_learning)
                    self.bounds_table[grounding_row] = fact.clone()
            else:
                fact = _utils.fact_to_bounds(fact, self.propositional)
                if self.propositional:
                    self.bounds_table = fact
                else:
                    self.bounds_table[grounding_row] = fact

        if isinstance(grounding_rows, set):
            facts = _utils.fact_to_bounds(facts, self.propositional)
            self.bounds_table[..., 0] = facts[0]
            self.bounds_table[..., 1] = facts[1]
        elif isinstance(grounding_rows, dict) or isinstance(grounding_rows, list):
            [func(*_) for _ in zip(grounding_rows, facts)]
        else:
            func(grounding_rows, facts)

    def extend_groundings(self, n: int = None):
        """extend the `bounds_table` number of groundings by n

        returns list of new bounds_table row numbers

        """
        n_groundings = self.bounds_table.shape[self._grounding_dims]
        new_leaves = _utils.fact_to_bounds(
            self.world, self.propositional, [n], requires_grad=self.bounds_learning
        )
        self.leaves = Parameter(
            torch.cat([self.leaves, new_leaves]), self.bounds_learning
        )
        self.bounds_table = torch.cat([self.bounds_table, new_leaves.clone()])
        return list(range(n_groundings, n_groundings + n))

    @property
    def _grounding_dims(self):
        """returns int of grounding dim in bounds table"""
        if self.propositional:
            return 0
        return 1 if self.bounds_table.dim() == 3 else 0

    def reset_bounds(self):
        """restores bounds_table to default state of leaves"""
        self.bounds_table = self.leaves.clone()

    @torch.no_grad()
    def project_params(self):
        self.alpha.data = self.alpha.data.clamp(0.5, 1)

    @torch.no_grad()
    def project_bounds(self):
        self.bounds_table.data = self.bounds_table.data.clamp(0, 1)
