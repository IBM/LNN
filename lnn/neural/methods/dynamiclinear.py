##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import torch

from ...constants import Direction
from ..._utils import negate_bounds as _not
from ..activations.neuron.dynamic import _DynamicActivation


class DynamicLinear(_DynamicActivation):
    """Interpolated Linear"""

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.w_m_slacks = kwds.get('w_m_slacks', 'max')
        self.kappa = self.add_param('kappa', torch.tensor(1.))
        self.kappa.requires_grad_(kwds.get('kappa_learning', False))

    def _and(self, direction: Direction):
        self.kappa.data = torch.tensor(1.)
        self.update_activation()
        return self._and_or(direction)

    def _or(self, direction: Direction):
        self.kappa.data = torch.tensor(0.)
        self.update_activation()
        return self._and_or(direction)

    def _atleast(self, direction: Direction):
        self.update_activation()
        return self._and_or(direction)

    def _implies(self, direction: Direction):
        def upward(in_bounds: torch.Tensor):
            lhs = in_bounds[..., 0]
            rhs = _not(in_bounds[..., 1])
            implies_bounds = torch.stack((lhs, rhs), dim=2)
            return _not(self._and(Direction.UPWARD)(implies_bounds))

        def downward(out_bounds: torch.Tensor, in_bounds: torch.Tensor):
            lhs = in_bounds[..., 0]
            rhs = _not(in_bounds[..., 1])
            implies_bounds = torch.stack((lhs, rhs), dim=2)
            tmp_bounds = self._and(Direction.DOWNWARD)(
                _not(out_bounds), implies_bounds)
            return torch.stack(
                [tmp_bounds[..., 0], _not(tmp_bounds[..., 1], dim=-1)],
                dim=-1)

        return upward if direction is Direction.UPWARD else (
            downward if direction is Direction.DOWNWARD else None)

    def _and_or(self, direction: Direction):
        def upward(in_bounds: torch.Tensor):
            x = in_bounds @ self.weights
            y = torch.zeros_like(x) - 1
            regions = self.input_regions(x.clone())
            y = torch.where(regions == 1, x * self.Gf, y)
            y = torch.where(regions == 2, self.Yf + (x - self.Xf) * self.Gz, y)
            y = torch.where(regions == 3, self.Yt + (x - self.Xt) * self.Gt, y)
            if any(y < 0) or any(y > 1) or any(y == -1):
                raise ValueError('output of activation expected in [0, 1], '
                                 f'received {y}')
            return y

        def downward(out_bounds: torch.Tensor, in_bounds: torch.Tensor):
            x = out_bounds.clone()
            regions = self.output_regions(x)
            x = torch.where(regions == 1, x * self.Gf_inv, x)
            x = torch.where(regions == 2,
                            self.Xf + (x - self.Yf) * self.Gz_inv, x)
            x = torch.where(regions == 3,
                            self.Xt + (x - self.Yt) * self.Gt_inv, x)
            if any(x < 0) or any(x > self.bias):
                raise ValueError('input to activation expected in [0, 1], '
                                 f'received {x}')

            input_terms = in_bounds * self.weights
            result = (x[:, None] - (input_terms.sum(dim=1)[:, None]
                                    - input_terms)) / self.weights
            return result.clamp(0, 1)

        return upward if direction is Direction.UPWARD else (
            downward if direction is Direction.DOWNWARD else None)

    def update_activation(self, **kwds):
        bias = self.weights[0] = self.weights[1:].sum()
        self.Xmax = bias
        w_m = self.TransparentMax.apply(
            self.weights.max() if self.w_m_slacks == 'max' else
            self.weights.mean() if self.w_m_slacks == 'mean' else
            self.weights.min())
        n = self.weights.shape[-1]
        k = 1 + self.kappa * (n - 1)
        self.Xf = bias - self.alpha * (
                w_m + ((n - k) / (n - 1 + self.eps)) * (bias - w_m))
        self.Xt = self.alpha * (w_m + ((k - 1) / (n - 1 + self.eps)) * (
                    bias - w_m))
        self.Gf = self.divide(self.Yf, self.Xf,
                              fill=0)
        self.Gz = self.divide(self.Yt - self.Yf, self.Xt - self.Xf,
                              fill=float('inf'))
        self.Gt = self.divide(1 - self.Yt, self.Xmax - self.Xt,
                              fill=0)
        self.Gf_inv = self.divide(torch.ones_like(self.Gf), self.Gf,
                                  fill=float('inf'))
        self.Gz_inv = self.divide(torch.ones_like(self.Gz), self.Gz,
                                  fill=0)
        self.Gt_inv = self.divide(torch.ones_like(self.Gt), self.Gt,
                                  fill=float('inf'))
        uniques = [self.Xmin, self.Xf, self.Xt, self.Xmax]
        if len(uniques) < len(set(uniques)):
            raise ValueError('expected unique values for input control '
                             f'points, received {uniques}')
