##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from typing import List

from ... import _exceptions
from .node import _NodeParameters

import torch
from torch.nn.parameter import Parameter


class _NeuronParameters(_NodeParameters):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.arity = kwds.get("arity")
        _exceptions.AssertAlphaNeuronArityValue(self.alpha, self.arity)
        bias = kwds.get("bias", 1.0)
        _exceptions.AssertBias(bias)

        self.bias = Parameter(
            torch.tensor(bias), requires_grad=kwds.get("bias_learning", False)
        )

        weights = kwds.get("weights", (1.0,) * self.arity)
        _exceptions.AssertWeights(weights, self.arity)
        self.weights = Parameter(torch.tensor(weights))

        self.w_max = kwds.get("w_max")
        self.b_max = kwds.get("b_max")
        self.negative_weights = kwds.get("negative_weights", False)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        condition = True
        params = dict(list(self.named_parameters()))
        other_params = dict(list(other.named_parameters()))

        if len(params) != len(other_params):
            return False

        for param_name in params:
            if param_name not in other_params:
                return False
            else:
                if params[param_name].dim() != other_params[param_name].dim():
                    return False
                if params[param_name].dim() > 0:
                    condition = condition and torch.equal(
                        params[param_name], other_params[param_name]
                    )
        return condition

    def __hash__(self):
        return hash(tuple(self.named_parameters(recurse=False)))

    def set_negative_weights(self, is_negatable: bool):
        self.negative_weights = is_negatable

    def update_weights_where(self, condition: List[bool], other: torch.Tensor):
        self.weights = self.weights.data.where(torch.as_tensor(condition), other)

    @torch.no_grad()
    def project_params(self):
        if self.negative_weights:
            if self.w_max:
                self.weights.data = self.weights.data.clamp(-self.w_max, self.w_max)
            else:
                pass  # weights have freedom to  move as pleased
        else:
            self.weights.data = self.weights.data.clamp(0, self.w_max)
        self.bias.data = self.bias.data.clamp(0, self.b_max)
