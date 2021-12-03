##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from .constraint_optim import ConstraintOptimizer
import torch


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

    def __init__(self, params, **kwds):
        super(PrimalDual, self).__init__(params, kwds)

    def and_step(self, **neuron):
        """updating the conjunction to handle constraints
        
        ** Usage **
        
        ```python
        self.param_resize_(neuron['dual'], len(neuron['weights'])+1)
        self.state['lr_dual'] = 1  # directly modifiable
        neuron['dual'] += 2  # only modifiable via in-place operations

        a_update = neuron['alpha'].grad  # custom alpha calculation here
        if neuron['alpha'].grad is not None:
            neuron['alpha'] += a_update * -self.state['lr_primal']
        w_update = neuron['weights'].grad  # custom weight calculation here
        if neuron['weights'].grad is not None:
            neuron['weights'] += w_update * -self.state['lr_primal']
        b_update = neuron['bias'].grad  # custom bias calculation here
        if neuron['bias'].grad is not None:
            neuron['bias'] += b_update * -self.state['lr_primal']
        ```

        """

        alpha = neuron['alpha']
        beta = neuron['bias']
        weights = neuron['weights']
        true_constraint = beta - weights * alpha - 1 + alpha + neuron['slacks']
        false_constraint = (1 - alpha) * weights.sum() - beta + 1 + (alpha - 1)
        constraints = torch.cat((
            true_constraint, torch.as_tensor([false_constraint])))
        # constraints[:-1] *= neuron['weights'] != 0
        self.param_resize_like_(neuron['dual'], constraints)

        # update weights
        jacobian = torch.vstack((
            torch.diag(torch.ones_like(weights) * (-alpha)),
            torch.ones_like(weights) * (1 - alpha)))
        neuron['weights'] += - self.state['lr_primal'] * (
                neuron['weights'].grad
                + neuron['slacks']
                + torch.mm(jacobian.transpose(0, 1),
                           (neuron['dual']
                            + self.state['lr_dual'] * constraints
                            )[None].transpose(0, 1).clamp(min=0)
                           ).transpose(0, 1).squeeze()
        )
        # neuron['weights'].clamp_(min=0, max=1)

        # update bias (beta)
        # jacobian_bias = torch.hstack(
        #     (torch.ones_like(weights), -1 * torch.ones(1)))
        # b_update = (neuron['bias'].grad +
        #             torch.mm(
        #                 jacobian_bias[None],
        #                 (neuron['dual']
        #                  + self.state['lr_dual']
        #                  * constraints
        #                  )[None].transpose(0, 1).clamp(min=0))
        #             ).squeeze()
        # neuron['bias'] += b_update * -self.state['lr_primal']

        # update slack variables
        # slack_grad = torch.dot(torch.ones_like(weights), neuron['weights'])
        # slack_aux = (torch.cat([neuron['dual'][1:len(weights)+1]]) +
        #              self.state['lr_dual'] * true_constraint).clamp(min=0)
        # print(slack_aux)
        # slack_jacobian = torch.dot(torch.ones_like(weights), slack_aux)
        # neuron['slacks'] -= self.state['lr_primal'] * (
        #             slack_grad + slack_jacobian)
        # neuron['slacks'].clamp_(max=0)

        # update dual variable
        if 'smooth' in neuron:
            self.param_resize_like_(neuron['smooth'], constraints)
            neuron['smooth'] += self.state['smooth_coeff'] * (
                    neuron['smooth'] - neuron['dual'])
            # neuron['dual'] += -neuron['dual'] + (
            #         neuron['smooth'] + self.state['lr_dual'] * constraints)
            neuron['dual'] += self.state['lr_dual'] * constraints
            neuron['dual'].clamp_(min=0)
        print(f'slack: {neuron["slacks"]} '
              'constraints: '
              # f'{constraints} '
              f'{(constraints * (constraints > 0)).sum()}')

    def or_step(self, **neuron):
        """updating the disjunction to handle constraints"""
        alpha = neuron['alpha']
        bias = neuron['bias']
        weights = neuron['weights']
        a_update = alpha.grad  # custom alpha calculation here
        if alpha.grad is not None:
            alpha.add_(a_update * -self.state['lr_primal'])
        w_update = weights.grad  # custom weight calculation here
        if weights.grad is not None:
            weights.add_(w_update * - self.state['lr_primal'])
        b_update = bias.grad  # custom bias calculation here
        if bias.grad is not None:
            bias.add_(b_update * -self.state['lr_primal'])

    def implies_step(self, **neuron):
        """updating the implication to handle constraints"""
        alpha = neuron['alpha']
        bias = neuron['bias']
        weights = neuron['weights']
        a_update = alpha.grad  # custom alpha calculation here
        if alpha.grad is not None:
            alpha.add_(a_update * -self.state['lr_primal'])
        w_update = weights.grad  # custom weight calculation here
        if weights.grad is not None:
            weights.add_(w_update * - self.state['lr_primal'])
        b_update = bias.grad  # custom bias calculation here
        if bias.grad is not None:
            bias.add_(b_update * -self.state['lr_primal'])
