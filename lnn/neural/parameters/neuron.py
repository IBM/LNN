##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import torch

from ... import _exceptions
from .node import _NodeParameters


class _NeuronParameters(_NodeParameters):
    def __init__(self, propositional, arity, truth, **kwargs):
        super().__init__(propositional, truth, **kwargs)
        self.arity = arity
        _exceptions.AssertAlphaNeuronArityValue(self.alpha, self.arity)
        bias = kwargs.get("bias", 1.0)
        _exceptions.AssertBias(bias)
        self.bias = self.add_param("bias", torch.tensor(bias))
        self.bias.requires_grad_(kwargs.get("bias_learning", True))
        weights = kwargs.get("weights", (1.0,) * arity)
        _exceptions.AssertWeights(weights, arity)
        self.weights = self.add_param("weights", torch.tensor(weights))
        self.weights.requires_grad_(kwargs.get("weights_learning", True))

    @torch.no_grad()
    def project_params(self):
        self.weights.data = self.weights.data.clamp(0, 1)
        self.bias.data = self.bias.data.clamp(0, self.arity)
