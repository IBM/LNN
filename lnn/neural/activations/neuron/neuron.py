##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##
from typing import Callable

import torch

from ..node import _NodeActivation
from ...parameters.neuron import _NeuronParameters
from ...._utils import val_clamp
from ....constants import Direction


class _NeuronActivation(_NodeActivation, _NeuronParameters):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.Xf = 1 - self.alpha
        self.Xt = self.alpha
        self.Xmid = 0.5

    # The type hint Callable can't be further specified since these functions differ
    #   def upward(operand_bounds: torch.Tensor)
    #   def downward(operator_bounds: torch.Tensor, operand_bounds: torch.Tensor)
    def activation(self, func: str, direction: Direction) -> Callable:
        """
        A convenient way to switch between different activations and directions
        :param func: The name of the activation operator
        :param direction: The direction of the activation function
        :return: The activation function
        """
        return getattr(self, f"_{func.lower()}_{direction.name.lower()}")

    @torch.no_grad()
    def downward_conditional(
        self,
        operator_bounds: torch.Tensor,
        f_inv: torch.Tensor,
        input_terms: torch.Tensor,
    ) -> torch.Tensor:
        full_and_input = input_terms.sum(dim=-1)[..., None]
        partial_and_input = (full_and_input - input_terms).flip([-2])
        result = 1 + (
            (f_inv[..., None] - self.bias + partial_and_input)
            / self.weights.clamp(min=0)
        )
        unknown = torch.ones_like(result)
        unknown[..., 0, :] = 0
        out_repeat = operator_bounds[..., None].repeat_interleave(
            unknown.shape[-1], dim=-1
        )
        result[..., 0, :] = result[..., 0, :].where(
            out_repeat[..., 0, :] > 1 - self.alpha, unknown[..., 0, :]
        )
        result[..., 1, :] = result[..., 1, :].where(
            out_repeat[..., 1, :] < self.alpha, unknown[..., 1, :]
        )

        weight_repeat = self.weights[None].repeat_interleave(2, dim=0)
        result = result.where(weight_repeat != 0, unknown)

        return val_clamp(result)

    def _and_upward(self, operand_bounds: torch.Tensor):
        pass

    def _and_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        pass

    def _or_upward(self, operand_bounds: torch.Tensor):
        pass

    def _or_downward(self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor):
        pass

    def _implies_upward(self, operand_bounds: torch.Tensor):
        pass

    def _implies_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        pass

    def _iff_upward(self, operand_bounds: torch.Tensor):
        r"""Placeholder for the Iff Neuron."""
        pass

    def _iff_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        r"""Placeholder for the Iff Neuron."""
        pass

    def _xor_upward(self, operand_bounds: torch.Tensor):
        r"""Placeholder for the XOr Neuron."""
        pass

    def _xor_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        r"""Placeholder for the XOr Neuron."""
        pass
