##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from typing import Union, List

from .variable import Variable
from .grounding import _Grounding
from ... import _utils

_utils.logger_setup()


class Function:
    r"""Creates functions in first-order logic formulae

    Parameters
    ------------
    name : str
        name of the function
    term : Variable, str, Grounding, or tuple of their combination
        function arguments

    Examples
    --------
    ```python
    plus_func = Function('plus', X, Y)
    x, y, z = Variables('x', 'y', 'z')
    ```
    """

    # Add output arity
    def __init__(self, name: str = "", input_dim: int = 1):
        self.name = name
        # The arity is expected to be known at construction.
        self.input_dim = input_dim

        # Constants and functions seen so far for each dimension.
        self.groundings = dict()

    def __str__(self):
        args = ""
        for arg_pos in range(self.input_dim):
            args += "dim_" + str(arg_pos) + ", "

        return self.name + "(" + args[0:-2] + ")"

    def __repr__(self):
        return self.__str__()

    def __call__(
        self,
        *args: Union[
            _Grounding,
            List[_Grounding],
            str,
            List[str],
            Variable,
            List[Variable],
            List[Union[str, _Grounding, Variable]],
        ],
    ) -> Union[_Grounding, "Function"]:
        r"""Calls a function with arguments.

        Parameters
        ------------
            args : List of _Grounding and/or output of called Function

        Examples
        --------
        ```python
        y = plus_func(zero, one)
        ```
        """

        # If no input provided, it must map all available groundings.
        if len(args) != self.input_dim and len(args) != 0:
            raise Exception(
                f"expected {self.input_dim} arguments" f"Received {len(args)}"
            )

        if all(
            [
                True if (isinstance(g, str) or isinstance(g, _Grounding)) else False
                for g in args
            ]
        ):
            # Full grounding
            ground_str = ""
            grounding = []
            for arg_pos, arg in enumerate(args):
                ground_str += str(arg) + ", "
                grounding.append(str(arg))

            if len(grounding) > 1:
                grounding = tuple(grounding)
            else:
                grounding = (grounding[0],)
            ground_out = self.groundings.get(grounding)
            if ground_out is None:
                ground_out = _Grounding(self.name + "(" + ground_str[0:-2] + ")")
                self.groundings[grounding] = ground_out

            return str(ground_out)

        if all([isinstance(g, Variable) for g in args]):
            # All variables
            return self

        # If not all groundings or variables we have a binding.
        bindings = {}
        for arg_pos, arg in enumerate(args):
            if isinstance(arg, tuple) or isinstance(arg, Function):
                bindings[arg_pos] = arg
            elif isinstance(arg, str) or isinstance(arg, _Grounding):
                bindings[arg_pos] = [arg]
            else:
                if not isinstance(arg, Variable):
                    raise TypeError(
                        f"Expected str, _Grounding, Variable, "
                        f"tuple or Function. Got {type(arg)}"
                    )
        return self, bindings
