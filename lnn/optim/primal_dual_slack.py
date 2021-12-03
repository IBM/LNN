##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import torch
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

    def __init__(self, params, **kwds):
        super(PrimalDual, self).__init__(params, kwds)

    def and_step(self, **neuron):
        """updating the conjunction to handle constraints"""
     #   self.state['lr_dual'] = 1  # directly modifiable
      #  neuron['dual'] += 2  # only modifiable via in-place operations
        alpha = neuron['alpha']
        beta = neuron['bias']
        weight = neuron['weights']

        # if "slack" not in self.state:
        #     self.state['slack'] = torch.ones_like(weight) * slack
        constraint = - alpha*weight + beta - 1 + alpha
        slacks = constraint * (constraint > 0)
        constraint -= slacks if self.state.get('slacks', False) else 0
        constraint_sum = (1 - alpha)*weight.sum() - beta + 1 + (alpha - 1)
        cat = torch.cat((constraint, torch.as_tensor([constraint_sum])))

        if 'dual_and' not in self.state:
            self.state['dual_and'] = torch.zeros_like(cat)
        if "smooth" not in self.state:
            self.state['smooth'] = torch.zeros_like(cat)

        """Construct Jacobian Matrix"""
        jacobian = torch.vstack((torch.diag(torch.ones_like(weight)*(-alpha)), torch.ones_like(weight)*(1-alpha)))

        """Update weights"""
        neuron['weights'] += - self.state['lr_primal'] * (
                    neuron['weights'].grad + slacks
                    + torch.mm(jacobian.transpose(0, 1),
                               (self.state['dual_and']
                                + self.state['lr_dual'] * cat)[None].transpose(0, 1).clamp(min=0)
                               ).transpose(0, 1).squeeze()
            )
        neuron['weights'].clamp_(min=0)

        # neuron['weights'].clamp_(max=self.state.get('w_max', 1))

       # neuron['weights'] +=  - self.state['lr_primal']*(neuron['weights'].grad)
       # print('smooth',self.state['smooth'])
       # print('dual_and', self.state['dual_and'])
       # a_update = neuron['alpha'].grad  # custom alpha calculation here
       # if neuron['alpha'].grad is not None:
       #    neuron['alpha'] += a_update * -self.state['lr_primal']
       # w_update = neuron['weights'].grad  # custom weight calculation here

        """ Update bias (beta)"""
        if self.state.get('bias_learning', True):
            jacobian_bias = torch.hstack((torch.ones_like(weight), -1*torch.ones(1)))
            b_update = (neuron['bias'].grad +
                       torch.mm(
                           jacobian_bias[None],
                           (self.state['dual_and'] + self.state['lr_dual'] * cat
                            )[None].transpose(0, 1).clamp(min=0))
                        ).squeeze()  # custom bias calculation here
            #if neuron['bias'].grad is not None:
            #     neuron['bias'] += b_update * -self.state['lr_primal']
            neuron['bias'] += -b_update * self.state['lr_primal']
            # neuron['bias'].clamp_(min=0)

        ''' Update Slack variable'''
        # slack_grad = torch.dot(torch.ones_like(weight), neuron['weights'])
        # slack_aux = (torch.cat([self.state['dual_and'][1:len(weight)+1]])
        #              + self.state['lr_dual'] * constraint).clamp(min=0)
        # print(slack_aux)
        # slack_jaco = torch.dot(torch.ones_like(weight), slack_aux)
        # neuron['slacks'] += -self.state['lr_primal'] * (slack_grad + slack_jaco)
        # neuron['slacks'].clamp_(max=0)

        """ Update dual variable"""
        self.state['smooth'] += 1 * (self.state['smooth'] - self.state['dual_and'])
        #    self.state['dual_and'] = self.state['dual_and'] + self.state['lr_dual'] * cat
        self.state['dual_and'] += self.state['lr_dual'] * cat
        self.state['dual_and'].clamp_(min=0)

        print('weights', neuron['weights'])
        print('slacks', slacks)
        print('constraints', cat)
        print()

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