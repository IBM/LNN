##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from typing import Union, Tuple

from ... import _utils

_utils.logger_setup()


class _Grounding(_utils.MultiInstance, _utils.UniqueNameAssumption):
    r"""Propositionalises constants for first-order logic

    Returns a container for a string or a tuple of strings.
    Follows the unique name assumption so that given constant(s) return the
        same object
    Decomposes multiple constants (from the tuple) by storing each str as a
        separate grounding object but returns only the compound container.
        This decomposition is used in grounding management to ensure that all
        partial strings also follow the unique name assumption by returning the
        same container

    Parameters
    ------------
    constants : str or tuple of str

    Examples
    --------
    ```python
    _Grounding('person1')
    _Grounding(('person1', 'date1'))
    ```

    Attributes
    ----------
    name : str
        conversion of 'constants' param to str form
    grounding_arity : int
        length of the 'constants' param
    partial_grounding : tuple(_Grounding)
        tuple of groundings for decomposition when constants given as tuple

    """

    def __init__(self, constants: Union[str, Tuple[str, ...]]):
        super().__init__(constants)
        self.name = str(constants)
        if isinstance(constants, tuple):
            self.grounding_arity = len(constants)
            self.partial_grounding = tuple(
                map(self._partial_grounding_from_str, constants)
            )
        else:
            self.grounding_arity = 1
            self.partial_grounding = (self,)

    @classmethod
    def _partial_grounding_from_str(cls, constant: str) -> "_Grounding":
        r"""Returns partial Grounding given grounding str"""
        return _Grounding.instances[constant]

    @classmethod
    def ground_by_groundings(cls, *grounding: "_Grounding"):
        r"""Reduce a tuple of groundings to a single grounding"""
        return (
            grounding[0]
            if len(grounding) == 1
            else cls.__class__(tuple(str(g) for g in grounding))
        )

    def __len__(self) -> int:
        r"""Returns the length of the grounding arity"""
        return self.grounding_arity

    def __str__(self) -> str:
        r"""Returns the name of the grounding"""
        return self.name

    @staticmethod
    def eval(grounding: "_Grounding") -> Union[str, Tuple[str, ...]]:
        r"""Returns the original constant(s) in str or tuple of str form"""
        return eval(grounding.name) if grounding.grounding_arity > 1 else grounding.name
