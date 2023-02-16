##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import importlib

from ... import _utils

_utils.logger_setup()


class _NodeActivation:
    def __call__(self, **kwds):
        return getattr(
            importlib.import_module("lnn.neural.activations.node"),
            "_NodeActivation",
        )(**kwds)
