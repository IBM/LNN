##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from typing import Optional, Union, Tuple

from ... import _utils, utils

_utils.logger_setup()


class Variable:
    r"""Free variables to quantify first-order logic formulae

    Parameters
    ------------
    name : str
        name of the free variable
    type : str, optional
        constant of the type associated with the free variable

    Examples
    --------
    ```python
    x = Variable('x', 'person')
    ```

    """

    def __init__(self, name: str, type: Optional[str] = None):
        self.name = name
        self.type = type

    def __str__(self) -> str:
        r"""Returns the name of the free variable"""
        return self.name


def Variables(*variables: str, **kwds) -> Union[Variable, Tuple[Variable, ...]]:
    """Instantiates multiple variables.

    Examples
    --------
    ```python
    x, y = Variables("x", "y")
    ```

    """
    return utils.return1([Variable(v, **kwds) for v in variables])
