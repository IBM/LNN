##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from typing import List

from ... import _exceptions
from .node import _NodeParameters

import torch


class _NeuronParameters(_NodeParameters):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.arity = kwds.get("arity")
        _exceptions.AssertAlphaNeuronArityValue(self.alpha, self.arity)
        bias = kwds.get("bias", 1.0)
        _exceptions.AssertBias(bias)
        self.bias = self.add_param("bias", torch.tensor(bias))
        self.bias.requires_grad_(kwds.get("bias_learning", False))
        weights = kwds.get("weights", (1.0,) * self.arity)
        _exceptions.AssertWeights(weights, self.arity)
        self.weights = self.add_param("weights", torch.tensor(weights))
        self.weights.requires_grad_(kwds.get("weights_learning", True))
        self.w_max = kwds.get("w_max")
        self.b_max = kwds.get("b_max")
        self.negative_weights = kwds.get("negative_weights", False)

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
