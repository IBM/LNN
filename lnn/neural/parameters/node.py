##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import math
from typing import Union, Optional, Tuple, Iterator, List, Dict, Set

import torch

from ... import _utils
from ...constants import Fact, World
from ... import _exceptions


class _NodeParameters:
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
        2D: [batch, bounds] (batched proposition) `not implimented`

        FOL:
        2D: [grounding, bounds] (predicate)
        3D: [batch, grounding, bounds] (batched predicate) `not implimented`

        Connective inputs (FOL + propositional) are extended on `dim = -1`
    """

    def __init__(self, propositional: bool, world: World, **kwargs):
        self.params = {}
        self.propositional = propositional
        self.world = world
        self.alpha = self.add_param(
            "alpha",
            torch.tensor(
                kwargs.get(
                    "alpha", math.erf(kwargs.get("alpha_sigma", 10) / math.sqrt(2))
                ),
                requires_grad=kwargs.get("alpha_learning", False),
            ),
        )
        _exceptions.AssertAlphaNodeValue(self.alpha)
        self.bounds_learning = kwargs.get("bounds_learning", False)
        self.leaves = _utils.fact_to_bounds(
            self.world, self.propositional, [0], requires_grad=self.bounds_learning
        )
        self.bounds_table = self.leaves.clone()
        if propositional:
            self.extend_groundings()

    def set_world(self, world: World):
        self.world = world

    def add_param(self, name, param):
        self.params[name] = param
        return param

    def flush(self, bounds_table=None):
        self.add_facts({Fact.UNKNOWN}, bounds_table)

    def get_facts(self, grounding_rows: List[int] = None):
        """returns tuple of all facts given by `grounding_rows`
        The `bounds_table` is returned if no `grounding_rows` given
        """
        if self.propositional:
            return self.bounds_table[0]
        if grounding_rows is None:
            return self.bounds_table
        if isinstance(grounding_rows, list):
            return self.bounds_table[grounding_rows]
        return self.bounds_table[grounding_rows]

    def add_facts(self, facts: Union[Fact, Tuple, Set, Dict], update_leaves=False):
        """Populate formula with facts

        Facts given in bool, tuple or None, assumes a propositional formula
        Facts given in dict form assume FOL, keyed by the tensor index, w
            the value also required in bool, Tuple or None

        """
        if self.propositional:  # Propositional facts
            if isinstance(facts, set):
                facts = next(iter(facts))
            self.update_bounds(0, facts, update_leaves)
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
                self.leaves[grounding_row] = fact
                self.bounds_table[grounding_row] = fact.clone()
            else:
                fact = _utils.fact_to_bounds(fact, self.propositional)
                self.bounds_table[grounding_row] = fact

        if isinstance(grounding_rows, set):
            facts = _utils.fact_to_bounds(facts, self.propositional)
            if not self.propositional:
                facts = facts[0]
            self.bounds_table[..., 0] = facts[0]
            self.bounds_table[..., 1] = facts[1]
        elif isinstance(grounding_rows, dict) or isinstance(grounding_rows, list):
            [func(*_) for _ in zip(grounding_rows, facts)]
        else:
            func(grounding_rows, facts)

    def extend_groundings(self, n: int = 1):
        """extend the `bounds_table` number of groundings by n

        returns list of new bounds_table row numbers

        """
        if n <= 0:
            raise Exception(f"n expected as int > 0, received {type(n), n}")
        n_groundings = self.bounds_table.shape[self._grounding_dims]
        new_leaves = _utils.fact_to_bounds(
            self.world, self.propositional, [n], requires_grad=self.bounds_learning
        )
        self.leaves = torch.cat([self.leaves, new_leaves])
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

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, param in self.params.items():
            yield name, param
        if self.bounds_learning:
            for idx, param in enumerate(self.leaves):
                if param.requires_grad and param.is_leaf:
                    yield f"bounds_{idx}", param
