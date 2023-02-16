##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import torch
from torch.optim.optimizer import Optimizer


class ConstraintOptimizer(Optimizer):
    def __init__(self, params, default):
        super(ConstraintOptimizer, self).__init__(
            params,
            {
                "neuron_type": default.get("neuron_type"),
                "param_names": default.get("param_names"),
            },
        )
        for k, v in default.get("per_neuron", {}).items():
            for neuron in self.param_groups:
                neuron[k] = (
                    v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v)
                )
        default.pop("per_neuron")
        self.state.update(default)

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
            if group["neuron_type"] in ["And", "Or", "Implies"]:
                kwds = dict(zip(group["param_names"], group["params"]))
                if group["neuron_type"] == "And":
                    self.and_step(**kwds, **group)
                elif group["neuron_type"] == "Or":
                    self.or_step(**kwds, **group)
                elif group["neuron_type"] == "Implies":
                    self.implies_step(**kwds, **group)
        return loss

    @staticmethod
    def param_resize_like_(param: torch.Tensor, tensor: torch.Tensor):
        """
        use inplace operations to resize the "param" to the len of "tensor"
        this operation should ideally only run once, at the first call of the
        neuron step. Expects the input as a 1D tensor, will broadcast the
        value at index [0] to all inputs

        Examples
        --------
        ```python
        self.param_resize_like_(neuron["dual"], neuron["weights"])
        ```

        """
        if not isinstance(tensor, torch.Tensor) or param.dim() > 1:
            raise ValueError(f"expected 1D tensor, received {tensor}")
        ConstraintOptimizer.param_resize_(param, len(tensor))

    @staticmethod
    def param_resize_(param: torch.Tensor, n: int):
        """
        use inplace operations to resize the param to the size of n
        this operation should ideally only run once, at the first call of the
        neuron step. Expects the input as a 1D tensor, will broadcast the
        value at index [0] to all inputs

        Examples
        --------
        ```python
        self.param_resize_(neuron["dual"], len(neuron["weights"])+1)
        ```

        """
        if isinstance(param, torch.Tensor):
            if param.dim() == 0:
                param.resize_(1)
            elif param.dim() > 1:
                raise ValueError(f"expected 1D tensor, " f"received {param}")
        else:
            raise ValueError(
                f"expected param as tensor, "
                f"received {param.__class__.__name__} {param}"
            )
        if len(param) != n:
            param.resize_(n)
            param[:] = param[0]

    def and_step(self, **neuron):
        """
        This functions should be implemented for a custom logical optimizer
        constraints are applied for a conjunction
        """
        raise NotImplementedError

    def or_step(self, **neuron):
        """
        This functions should be implemented for a custom logical optimizer
        constraints are applied for a disjunction
        """
        raise NotImplementedError

    def implies_step(self, **neuron):
        """
        This functions should be implemented for a custom logical optimizer
        constraints are applied for an implication
        """
        raise NotImplementedError
