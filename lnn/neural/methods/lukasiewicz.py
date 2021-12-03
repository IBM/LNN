##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ...constants import Direction
from ..._utils import negate_bounds as _not
from ..activations.neuron.static import _StaticActivation
import torch


class Lukasiewicz(_StaticActivation):
    """Weighted Lukasiewicz"""

    def _and(self, direction: Direction):
        def upward(in_bounds: torch.Tensor):
            return (self.bias - ((1 - in_bounds) @ self.weights)).clamp(0, 1)

        def downward(out_bounds: torch.Tensor,
                     in_bounds: torch.Tensor):
            f_inv = (out_bounds
                     + ((out_bounds <= 0).float()
                        * torch.stack([self.bias - self.weights.sum(),
                                       torch.tensor(0.)]))
                     + ((out_bounds >= 1).float()
                        * torch.stack([torch.tensor(0.), self.bias - 1])))
            input_terms = (1 - in_bounds) * self.weights
            result = self.downward_conditional(
                out_bounds, f_inv, input_terms)
            return result

        return upward if direction is Direction.UPWARD else (
            downward if direction is Direction.DOWNWARD else None)

    def _or(self, direction: Direction):
        def upward(in_bounds: torch.Tensor):
            return (1 - self.bias + (in_bounds @ self.weights)).clamp(0, 1)

        def downward(out_bounds: torch.Tensor,
                     in_bounds: torch.Tensor):
            return _not(
                self._and(Direction.DOWNWARD)(
                    _not(out_bounds), _not(in_bounds, dim=-2)),
                dim=-2)

        return upward if direction is Direction.UPWARD else (
            downward if direction is Direction.DOWNWARD else None)

    def _implies(self, direction: Direction):
        def upward(in_bounds: torch.Tensor):
            result = (1 - self.bias
                      + (self.weights[0] * (1 - in_bounds[..., 0].flip(-1)))
                      + (self.weights[1] * in_bounds[..., 1])).clamp(0, 1)
            return result

        def downward(out_bounds: torch.Tensor,
                     in_bounds: torch.Tensor):
            lhs, rhs = in_bounds[..., 0], _not(in_bounds[..., 1])
            tmp_bounds = self._and(Direction.DOWNWARD)(
                _not(out_bounds), torch.stack([lhs, rhs], dim=-1))
            return torch.stack(
                [tmp_bounds[..., 0], _not(tmp_bounds[..., 1], dim=-1)],
                dim=-1)

        return upward if direction is Direction.UPWARD else (
            downward if direction is Direction.DOWNWARD else None)
