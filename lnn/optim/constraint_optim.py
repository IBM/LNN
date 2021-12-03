##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import torch
from torch.optim.optimizer import Optimizer


class ConstraintOptimizer(Optimizer):

    def __init__(self, params, defaults):
        defaults['neuron_type'] = None
        defaults['param_names'] = None
        super(ConstraintOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['neuron_type'] in ['And', 'Or', 'Implies']:
                kwds = dict(zip(group['param_names'], group['params']))
                if group['neuron_type'] == 'And':
                    self.and_step(**kwds, **group)
                elif group['neuron_type'] == 'Or':
                    self.or_step(**kwds, **group)
                elif group['neuron_type'] == 'Implies':
                    self.implies_step(**kwds, **group)
        return loss

    def and_step(self, **kwds):
        """
        This functions should be implemented for a custom logical optimizer
        constraints are applied for a conjunction
        """
        pass

    def or_step(self, **kwds):
        """
        This functions should be implemented for a custom logical optimizer
        constraints are applied for a disjunction
        """
        pass

    def implies_step(self, **kwds):
        """
        This functions should be implemented for a custom logical optimizer
        constraints are applied for an implication
        """
        pass
