##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ...constants import Direction
from ..._utils import val_clamp, negate_bounds as _not
from ..activations.neuron.static import _StaticActivation

import torch


class LukasiewiczTransparent(_StaticActivation):
    """Weighted Lukasiewicz with gradient transparent clamping"""

    def _and_upward(self, operand_bounds: torch.Tensor):
        return val_clamp(self.bias - ((1 - operand_bounds) @ self.weights))

    def _and_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        f_inv = (
            operator_bounds
            + (
                (operator_bounds <= 0).float()
                * torch.stack([self.bias - self.weights.sum(), torch.tensor(0.0)])
            )
            + (
                (operator_bounds >= 1).float()
                * torch.stack([torch.tensor(0.0), self.bias - 1])
            )
        )
        input_terms = (1 - operand_bounds) * self.weights
        result = self.downward_conditional(operator_bounds, f_inv, input_terms)
        return result

    def _or_upward(self, operand_bounds: torch.Tensor):
        return val_clamp(
            1
            - self.bias
            - self.weights.minimum(torch.zeros_like(self.weights)).sum()
            + (operand_bounds @ self.weights)
        )

    def _or_downward(self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor):
        return _not(
            self.activation("AND", Direction.DOWNWARD)(
                _not(operator_bounds), _not(operand_bounds, dim=-2)
            ),
            dim=-2,
        )

    def _implies_upward(self, operand_bounds: torch.Tensor):
        result = val_clamp(
            1
            - self.bias
            + (self.weights[0] * (1 - operand_bounds[..., 0].flip(-1)))
            + (self.weights[1] * operand_bounds[..., 1])
        )
        return result

    def _implies_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        lhs, rhs = operand_bounds[..., 0], _not(operand_bounds[..., 1])
        tmp_bounds = self.activation("AND", Direction.DOWNWARD)(
            _not(operator_bounds), torch.stack([lhs, rhs], dim=-1)
        )
        return torch.stack(
            [tmp_bounds[..., 0], _not(tmp_bounds[..., 1], dim=-1)], dim=-1
        )
