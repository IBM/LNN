##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import logging
from typing import Union, Tuple, Set, Dict

import torch

from .connective_formula import _ConnectiveFormula
from .formula import Formula
from .node_activation import _NodeActivation
from .. import _gm
from ... import _utils
from ...constants import Fact

_utils.logger_setup()


class _NAryOperator(_ConnectiveFormula):
    r"""N-ary connective operator"""

    def __init__(self, *formula, **kwds):
        super().__init__(*formula, arity=len(formula), **kwds)


class Congruent(_NAryOperator):
    r"""Symbolic Congruency

    This is used to define nodes that are symbolically equivalent to one another
    (despite the possibility of neural differences)

    """

    def __init__(self, *formulae: Formula, **kwds):
        self.connective_str = "≅"
        super().__init__(*formulae, **kwds)
        kwds.setdefault("propositional", self.propositional)
        self.neuron = _NodeActivation()(**kwds.get("activation", {}), **kwds)

    def __contains__(self, item):
        return True if item in self.congruent_nodes else False

    def add_data(self, facts: Union[Fact, Tuple, Set, Dict]):
        """Should not be called by the user"""
        raise AttributeError(
            "Should not be called directly by the user, instead use "
            "`congruent_node.upward()` to evaluate the facts from the operands"
        )

    def upward(
        self, groundings: Set[Union[str, Tuple[str, ...]]] = None, **kwds
    ) -> float:
        r"""Upward inference from the operands to the operator.

        Parameters
        ----------
        groundings : str or tuple of str
            restrict upward inference to a specific grounding or row in the truth table

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """
        upward_bounds = _gm.upward_bounds(self, self.operands, groundings)
        if upward_bounds is None:  # contradiction arresting
            return
        input_bounds, groundings = upward_bounds
        grounding_rows = (
            None
            if self.propositional
            else (
                self.grounding_table.values()
                if groundings is None
                else [self.grounding_table.get(g) for g in groundings]
            )
        )
        input_bounds = torch.stack(
            [
                input_bounds[..., 0, :].max(-1)[0],
                input_bounds[..., 1, :].max(-1)[0],
            ],
            dim=-1,
        )
        result = self.neuron.aggregate_bounds(grounding_rows, input_bounds)
        if result:
            logging.info(
                "↑ BOUNDS UPDATED "
                f"TIGHTENED:{result} "
                f"FOR:'{self.name}' "
                f"FORMULA:{self.formula_number} "
            )
        return result

    def downward(
        self,
        index: int = None,
        groundings: Set[Union[str, Tuple[str, ...]]] = None,
        **kwds,
    ) -> Union[torch.Tensor, None]:
        r"""Downward inference from the operator to the operands.

        Parameters
        ----------
        index : int, optional
            restricts downward inference to an operand at the specified index. If unspecified, all operands are updated.
        groundings : str or tuple of str, optional
            restrict upward inference to a specific grounding or row in the truth table

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """
        downward_bounds = _gm.downward_bounds(self, self.operands, groundings)
        if downward_bounds is None:  # contradiction arresting
            return
        parent, _, groundings = downward_bounds
        op_indices = (
            enumerate(self.operands)
            if index is None
            else ([(index, self.operands[index])])
        )
        result = 0
        for op_index, op in op_indices:
            if op.propositional:
                op_grounding_rows = None
            else:
                if groundings is None:
                    op_grounding_rows = op.grounding_table.values()
                else:
                    op_grounding_rows = [None] * len(groundings)
                    for g_i, g in enumerate(groundings):
                        op_g = [
                            str(g.partial_grounding[slot])
                            for slot in self.operand_map[op_index]
                        ]
                        op_g = tuple(op_g)
                        op_grounding_rows[g_i] = op.grounding_table.get(op_g)
            op_aggregate = op.neuron.aggregate_bounds(op_grounding_rows, parent)
            if op_aggregate:
                logging.info(
                    "↓ BOUNDS UPDATED "
                    f"TIGHTENED:{op_aggregate} "
                    f"FOR:'{op.name}' "
                    f"FROM:'{self.name}' "
                    f"FORMULA:{op.formula_number} "
                    f"PARENT:{self.formula_number} "
                )
            result = result + op_aggregate
        return result

    def extract_congruency(self, *formulae):
        for idx, formula in enumerate(formulae):
            if self not in formula.congruent_nodes:
                formula.congruent_nodes.append(self)

    def set_congruency(self):
        for formula in self.operands:
            if self not in formula.congruent_nodes:
                formula.congruent_nodes.append(self)

    def upward(
        self, groundings: Set[Union[str, Tuple[str, ...]]] = None, **kwds
    ) -> float:
        upward_bounds = _gm.upward_bounds(self, self.operands, groundings)
        if upward_bounds is None:  # contradiction arresting
            return
        input_bounds, groundings = upward_bounds
        grounding_rows = (
            None
            if self.propositional
            else (
                self.grounding_table.values()
                if groundings is None
                else [self.grounding_table.get(g) for g in groundings]
            )
        )
        input_bounds = torch.stack(
            [
                input_bounds[..., 0, :].max(-1)[0],
                input_bounds[..., 1, :].min(-1)[0],
            ],
            dim=-1,
        )
        result = self.neuron.aggregate_bounds(grounding_rows, input_bounds)
        if result:
            logging.info(
                "↑ BOUNDS UPDATED "
                f"TIGHTENED:{result} "
                f"FOR:'{self.name}' "
                f"FORMULA:{self.formula_number} "
            )
        return result
