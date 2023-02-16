##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import importlib

from ... import _utils, _exceptions
from ...constants import NeuralActivation

_utils.logger_setup()


class _NeuralActivation:
    r"""Switch class, to choose a method from the correct activation class"""

    def __init__(self, type=None):
        self.neuron_type = type if type else NeuralActivation.LukasiewiczTransparent
        _exceptions.AssertNeuronActivationType(self.neuron_type)
        self.module = importlib.import_module(
            f"lnn.neural.methods.{self.neuron_type.name.lower()}"
        )

    def __call__(self, **kwds):
        return getattr(self.module, self.neuron_type.name)(**kwds)
