##
# Copyright 2021 IBM Corp All Rights Reserved
#
# SPDX-License-Identifier: Apache-20
##

from enum import Enum, auto


class AutoName(Enum):
    r"""Automatic enumeration of Enum classes from the variable name

    The name and value both become str wrapped versions of the given variable
    e.g. ENUM_VALUE_1 below:
    ```
    MyEnumClass.ENUM_VALUE_1.name = 'ENUM_VALUE_1'
    MyEnumClass.ENUM_VALUE_1.value = 'ENUM_VALUE_1'
    ```

    **Examples**

    ```python
    class MyEnumClass(AutoName):
        ENUM_VALUE_1 = auto()
        ENUM_VALUE_2 = auto()
        ENUM_VALUE_3 = auto()
    ```

    """

    def _generate_next_value_(self, start, count, last_values):
        return self


class World(Enum):
    r"""An enumeration for world assumptions to for incomplete knowledge

    See the discussion of
    [OPEN vs CLOSED](https://en.wikipedia.org/wiki/Open-world_assumption)
    world assumptions for more information

    **Attributes**

    ```raw
    OPEN : Formulae that follow the open world assumption
    CLOSED : Formulae that follow the open world assumption
    AXIOM : Formulae that follow assumptions of universally being TRUE
    FALSE : Alias for CLOSED
    ```

    """
    OPEN = (0, 1)
    CLOSED = (0, 0)
    FALSE = (0, 0)
    AXIOM = (1, 1)


class Fact(Enum):
    r"""An enumeration for facts

    **Attributes**

    ```raw
    TRUE : Classically True inputs as represented by LNN bounds
    FALSE : Classically False inputs as represented by LNN bounds
    UNKNOWN : A classical interpretation of what would be an unknown input,
              this is represented by LNN bounds with complete uncertainty
    CONTRADICTION : A classical interpretation of a contradictory input.
                    Contradictions arise by disagreement of two signal,
                    most often coming from different directions
                    (UPWARD+DOWNWARD). This can be interpreted as a
                    disagreement between the data and the logic, or
                    more appropriately - introducing data to a logic that
                    violates soundness and thereby moves the model into
                    worlds that are not consistent with the logic.
    ```

    """
    UNKNOWN = (0, 1)
    FALSE = (0, 0)
    TRUE = (1, 1)
    CONTRADICTION = (1, 0)


class _Fact(AutoName):
    r"""An enumeration for facts used in state printing"""
    APPROX_FALSE = auto()
    APPROX_TRUE = auto()
    APPROX_UNKNOWN = auto()
    EXACT_UNKNOWN = auto()


class Direction(AutoName):
    r"""An enumeration for the direction of inference

    **Attributes**

    ```raw
    UPWARD : Evaluation of the operator given the operands
    DOWNWARD : Evaluation of operands given the operator and the rest of
               the operands
    ```

    """
    UPWARD = auto()
    DOWNWARD = auto()


class Join(AutoName):
    r"""An enumeration for joining methods

    **Attributes**

    ```raw
    INNER : (default) Extended version of inner db joins that allows
            groundings to be propagated between operands
    OUTER : Extended outer db join with grounding propagation
    OUTER_PRUNED : A reduced outer join
    ```

    """
    OUTER = auto()
    INNER = auto()
    OUTER_PRUNED = auto()
