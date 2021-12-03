##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ...._utils import val_clamp
from ..node import _NodeActivation
from ....constants import Direction
from ...parameters.neuron import _NeuronParameters

import torch

"""
Dynamic activation function
"""


class _NeuronActivation(_NodeActivation, _NeuronParameters):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.Xf = 1 - self.alpha
        self.Xt = self.alpha
        self.Xmid = 0.5

    def function(self, func: str, direction: Direction) -> torch.Tensor:
        return getattr(self, '_'+func.lower())(direction)

    @torch.no_grad()
    def downward_conditional(self,
                             out_bounds: torch.Tensor,
                             f_inv: torch.Tensor,
                             input_terms: torch.Tensor
                             ) -> torch.Tensor:
        full_and_input = input_terms.sum(dim=-1)[..., None]
        partial_and_input = (full_and_input - input_terms).flip([-2])
        result = 1 + (
            (f_inv[..., None] - self.bias + partial_and_input)
            / self.weights.clamp(min=0))
        unknown = torch.ones_like(result)
        unknown[..., 0, :] = 0
        out_repeat = out_bounds[..., None].repeat_interleave(
            unknown.shape[-1], dim=-1)
        result[..., 0, :] = result[..., 0, :].where(
            out_repeat[..., 0, :] > 1 - self.alpha,
            unknown[..., 0, :])
        result[..., 1, :] = result[..., 1, :].where(
            out_repeat[..., 1, :] < self.alpha,
            unknown[..., 1, :])

        weight_repeat = self.weights[None].repeat_interleave(2, dim=0)
        result = result.where(weight_repeat != 0, unknown)

        return val_clamp(result)

    def _bidirectional(self, direction: Direction):
        """placeholder for the Bidirectional Neuron

        This neuron currently uses an AND reformulation to execute
        """
        pass
