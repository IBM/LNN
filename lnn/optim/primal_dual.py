##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from .constraint_optim import ConstraintOptimizer


class PrimalDual(ConstraintOptimizer):
    """Implements Primal Dual algorithm.

    **Example**
    ```python
    from lnn.optim import PrimalDual
    model.train(
        optimizer=PrimalDual(
            model.parameters_grouped_by_neuron(),
            lr=.1
        ), ...
    )
    ```

    .. _Primal Dual algorithm:
        https://ieeexplore.ieee.org/abstract/document/9415044
    """

    def __init__(self, params, lr, **kwds):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, **kwds)
        super(PrimalDual, self).__init__(params, defaults)

    def and_step(self, **params):
        """updating the conjunction to handle constraints"""
        lr = params['lr']
        alpha = params['alpha']
        bias = params['bias']
        weights = params['weights']
        a_update = alpha.grad  # custom alpha calculation here
        if alpha.grad is not None:
            alpha.add_(a_update * -lr)
        w_update = weights.grad  # custom weight calculation here
        if weights.grad is not None:
            weights.add_(w_update * - lr)
        b_update = bias.grad  # custom bias calculation here
        if bias.grad is not None:
            bias.add_(b_update * -lr)

    def or_step(self, **params):
        """updating the disjunction to handle constraints"""
        lr = params['lr']
        alpha = params['alpha']
        bias = params['bias']
        weights = params['weights']
        a_update = alpha.grad  # custom alpha calculation here
        if alpha.grad is not None:
            alpha.add_(a_update * -lr)
        w_update = weights.grad  # custom weight calculation here
        if weights.grad is not None:
            weights.add_(w_update * - lr)
        b_update = bias.grad  # custom bias calculation here
        if bias.grad is not None:
            bias.add_(b_update * -lr)

    def implies_step(self, **params):
        """updating the implication to handle constraints"""
        lr = params['lr']
        alpha = params['alpha']
        bias = params['bias']
        weights = params['weights']
        a_update = alpha.grad  # custom alpha calculation here
        if alpha.grad is not None:
            alpha.add_(a_update * -lr)
        w_update = weights.grad  # custom weight calculation here
        if weights.grad is not None:
            weights.add_(w_update * - lr)
        b_update = bias.grad  # custom bias calculation here
        if bias.grad is not None:
            bias.add_(b_update * -lr)
