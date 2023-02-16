##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from .neuron import _NeuronActivation

import torch


class _DynamicActivation(_NeuronActivation):
    """
    Dynamic neuron activation function
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.Xmin = 0
        self.Xf = 1 - self.alpha
        self.Xt = self.alpha
        self.Xmid = 0.5
        self.Xmax = 1

    def input_regions(self, bounds) -> torch.Tensor:
        result = torch.zeros_like(bounds)
        result = result.masked_fill((self.Xmin <= bounds) * (bounds <= self.Xf), 1)
        result = result.masked_fill((self.Xf < bounds) * (bounds < self.Xt), 2)
        result = result.masked_fill((self.Xt <= bounds) * (bounds <= self.Xmax), 3)
        if any(result == 0):
            raise ValueError(
                "Unknown input regions. Expected all values from "
                f"[1, 2, 3], received  {result}"
            )
        return result

    class TransparentMax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bounds):
            ctx.save_for_backward(bounds)
            return bounds.max()

        @staticmethod
        def backward(ctx, grad_output):
            (bounds,) = ctx.saved_tensors
            flags = bounds == bounds.max()
            grad_input = torch.ones_like(bounds) * grad_output.clone()
            grad_input = grad_input.where(flags, torch.zeros_like(bounds))
            return grad_input

    @staticmethod
    def divide(divident: torch.Tensor, divisor: torch.Tensor, fill=1.0) -> torch.Tensor:
        """
        Divide the bounds tensor (divident) by weights (divisor) while
            respecting gradient connectivity
        shortcurcuits a div 0 error with the fill value
        """
        shape = divident.shape
        if divident.dim() < 2:
            divident = divident.reshape(1, -1)
        div = divident.masked_select(divisor != 0) / divisor.masked_select(divisor != 0)
        result = divident.masked_scatter(divisor != 0, div)
        result = result.masked_fill(divisor == 0, fill)
        return result.reshape(shape)
