##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from .formula import Formula
from ... import _utils

_utils.get_logger()


class _ConnectiveFormula(Formula):
    def __init__(self, *formula: Formula, **kwds):
        super().__init__(*formula, **kwds)
