##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import logging
from typing import Union, Tuple, Set

import torch

from .connective_formula import _ConnectiveFormula
from .neural_activation import _NeuralActivation
from .. import _gm
from ... import _utils
from ...constants import Direction

_utils.logger_setup()
subclasses = {}


def _isinstance(obj, class_str) -> bool:
    """
    Returns True if an object is an instance of a class give that class name as a
    string, otherwise False. The check is performed using the subclasses dictionary.
    To see what the subclasses dictionary is populated with, refer to the init file of
    this module.
    """
    return isinstance(obj, subclasses[class_str])


class _ConnectiveNeuron(_ConnectiveFormula):
    def __init__(self, *formula, **kwds):
        super().__init__(*formula, **kwds)
        kwds.setdefault("arity", self.arity)
        kwds.setdefault("propositional", self.propositional)
        if kwds.get("activation", {}).get("negative_weights"):
            self.negation_absorption()
        self.neuron = _NeuralActivation(kwds.get("activation", {}).get("type"))(
            **kwds.get("activation", {}), **kwds
        )
        self.func = self.neuron.activation(
            self.__class__.__name__, direction=Direction.UPWARD
        )
        self.func_inv = self.neuron.activation(
            self.__class__.__name__, direction=Direction.DOWNWARD
        )

    def upward(
        self, groundings: Set[Union[str, Tuple[str, ...]]] = None, **kwds
    ) -> float:
        r"""Upward inference from the operands to the operator.

        Parameters
        ----------
        groundings : str or tuple of str, optional
            restrict upward inference to a specific grounding or row in the truth table

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """
        upward_bounds = _gm.upward_bounds(self, self.operands, groundings)
        if upward_bounds is None:  # contradiction arresting
            return 0.0
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
        result = self.neuron.aggregate_bounds(grounding_rows, self.func(input_bounds))
        if result:
            logging.info(
                "↑ BOUNDS UPDATED "
                f"TIGHTENED:{result} "
                f"FOR:'{self.name}' "
                f"FORMULA:{self.formula_number} "
            )
        if self.is_contradiction():
            logging.info(
                "↑ CONTRADICTION "
                f"FOR:'{self.name}' "
                f"FORMULA:{self.formula_number} "
            )
        return result

    def downward(
        self,
        index: int = None,
        groundings: Set[Union[str, Tuple[str, ...]]] = None,
        **kwds,
    ) -> float:
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
            return 0.0
        out_bounds, input_bounds, groundings = downward_bounds
        new_bounds = self.func_inv(out_bounds, input_bounds)
        op_indices = (
            enumerate(self.operands)
            if index is None
            else ([(index, self.operands[index])])
        )
        result = 0.0
        for op_index, op in op_indices:
            op_grounding_rows = None
            duplicates = False
            unique_grounding_rows = set()

            if not op.propositional:
                if groundings is None:
                    op_grounding_rows = op.grounding_table.values()
                else:
                    op_grounding_rows = [None] * len(groundings)
                    for g_i, g in enumerate(groundings):
                        if self.operand_map[op_index]:
                            op_g = [g[slot] for slot in self.operand_map[op_index]]
                            op_g = tuple(op_g)
                            row = op.grounding_table.get(op_g)
                            op_grounding_rows[g_i] = row
                            duplicates = duplicates or row in unique_grounding_rows
                            unique_grounding_rows.add(row)

            op_aggregate = op.neuron.aggregate_bounds(
                op_grounding_rows, new_bounds[..., op_index], duplicates=duplicates
            )
            if op_aggregate:
                logging.info(
                    "↓ BOUNDS UPDATED "
                    f"TIGHTENED:{op_aggregate} "
                    f"FOR:'{op.name}' "
                    f"FROM:'{self.name}' "
                    f"FORMULA:{op.formula_number} "
                    f"PARENT:{self.formula_number} "
                )
            if op.is_contradiction():
                logging.info(
                    "↓ CONTRADICTION "
                    f"FOR:'{op.name}' "
                    f"FROM:'{self.name}' "
                    f"FORMULA:{op.formula_number} "
                    f"PARENT:{self.formula_number} "
                )
            result = result + op_aggregate
        return result

    def _logical_loss(
        self, coeff: float = None, slacks: Union[bool, float] = None
    ) -> torch.Tensor:
        r"""Logical loss to create a loss on logical constraint violation.

        Assumes a soft logic computation and calculates the loss on constraints
        as defined in [equations 86-89](https://arxiv.org/pdf/2006.13155.pdf)
        when slacks are given, the constraints are allowed to be violated
        however this affects the neuron interpretability and should only be
        used if the model is not strictly required to obey a classical
        definition of logic

        """
        a = self.neuron.alpha
        b = self.neuron.bias
        w = self.neuron.weights
        T, F = a, 1 - a
        coeff = 1 if coeff is None else coeff
        if _isinstance(self, "And"):
            TRUE = b - (w * (1 - T)).sum()
            FALSE = b - (w * (1 - F))
            true_hinge = torch.where(TRUE < T, T - TRUE, TRUE * 0)
            false_hinge = torch.where(FALSE > F, FALSE - F, FALSE * 0)
            if slacks:
                if slacks is True:
                    slacks_false = false_hinge * (false_hinge > 0)
                    slacks_true = true_hinge * (true_hinge > 0)
                    false_hinge -= slacks_false
                    true_hinge -= slacks_true
                    self.neuron.slacks = (
                        slacks_true.detach().clone(),
                        slacks_false.detach().clone(),
                    )
                else:
                    false_hinge -= slacks
            self.neuron.feasibility = (
                true_hinge.detach().clone(),
                false_hinge.detach().clone(),
            )

        elif _isinstance(self, "Or"):
            TRUE = 1 - b + (w * T)
            FALSE = 1 - b + (w * F).sum()
            true_hinge = torch.where(TRUE < T, T - TRUE, TRUE * 0).sum()
            false_hinge = torch.where(FALSE > F, FALSE - F, FALSE * 0)
        elif _isinstance(self, "Implies"):
            TRUE = 1 - b + (w * T)  # T = 1-F for x and T for y
            FALSE = 1 - b + (w[0] * (1 - T)) + (w[1] * F)
            true_hinge = torch.where(TRUE < T, T - TRUE, TRUE * 0).sum()
            false_hinge = torch.where(FALSE > F, FALSE - F, FALSE * 0)
        result = (true_hinge.square() + false_hinge.square()).sum()
        return coeff * result

    def neural_equivalence(self, other):
        if (
            isinstance(self.neuron, other.neuron)
            and self.neuron.bias == other.neuron.bias
            and len(self.neuron.weights) == len(other.neuron.weights)
            and all(
                self.neuron.weights[idx] == other.neuron.weights[idx]
                for idx in range(len(self.neuron.weights))
            )
        ):
            return True
        return False
