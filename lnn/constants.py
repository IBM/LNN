##
# Copyright 2023 IBM Corp All Rights Reserved
#
# SPDX-License-Identifier: Apache-20
##

# flake8: noqa: E501

from enum import Enum, auto


class _AutoName(Enum):
    r"""Automatic enumeration of Enum classes from the variable name.

    The name and value both become str wrapped versions of the given variable
    e.g. ENUM_VALUE_1 below:
    ```python
    MyEnumClass.ENUM_VALUE_1.name = "ENUM_VALUE_1"
    MyEnumClass.ENUM_VALUE_1.value = "ENUM_VALUE_1"
    ```

    Examples
    --------

    ```python
    class MyEnumClass(_AutoName):
        ENUM_VALUE_1 = auto()
        ENUM_VALUE_2 = auto()
        ENUM_VALUE_3 = auto()
    ```

    """

    def _generate_next_value_(self, start, count, last_values):
        return self


class _Fact(_AutoName):
    r"""An enumeration for facts used in state printing."""

    APPROX_FALSE = auto()
    APPROX_TRUE = auto()
    APPROX_UNKNOWN = auto()
    EXACT_UNKNOWN = auto()


class Bound(_AutoName):
    r"""An enumeration for the Bound of inference.

    Used to restrict an inference to a particular specified bound.

    Parameters
    ----------
    LOWER
        Evaluation of the lower bound of the range, or minimum possible value that truth can take wihtin the given range
    UPPER
        Evaluation of the upper bound of the range, or maximum possible value that truth can take wihtin the given range

    """

    LOWER = auto()
    UPPER = auto()


class Direction(_AutoName):
    r"""An enumeration for the direction of inference.

    Used to restrict an inference pass (of a particular node or the entire model) to a single pass of the specified direction.

    Parameters
    ----------
    DOWNWARD
        Evaluation of operands given the operator and the rest of the operands
    UPWARD
        Evaluation of the operator given the operands

    """

    DOWNWARD = auto()
    UPWARD = auto()


class Fact(Enum):
    r"""An enumeration for facts.

    Parameters
    ----------
    CONTRADICTION
        A classical interpretation of a contradictory input. Contradictions arise by disagreement of two signal, most often coming from different directions (UPWARD+DOWNWARD). This can be interpreted as a disagreement between the data and the logic, or more appropriately - introducing data to a logic that violates soundness and thereby moves the model into worlds that are not consistent with the logic.
    FALSE
        Classically False inputs as represented by LNN bounds
    TRUE
        Classically True inputs as represented by LNN bounds
    UNKNOWN
        A classical interpretation of what would be an unknown input, this is represented by LNN bounds with complete uncertainty

    """

    CONTRADICTION = (1.0, 0.0)
    FALSE = (0.0, 0.0)
    TRUE = (1.0, 1.0)
    UNKNOWN = (0.0, 1.0)


class Join(_AutoName):
    r"""An enumeration for joining methods.

    Parameters
    ----------
    INNER : default
        Extended version of inner db joins that allows groundings to be propagated between operands
    OUTER
        Extended outer db join with grounding propagation
    OUTER_PRUNED
        A reduced outer join

    """

    INNER = auto()
    INNER_EXTENDED = auto()
    OUTER = auto()
    OUTER_PRUNED = auto()


class Loss(_AutoName):
    r"""An enumeration for standard loss functions used during training.

    Parameters
    ----------
    CONTRADICTION
        Ensures logical consistency.
    CUSTOM
        Not Implemented.
    LOGICAL
        Enforces logical constraints.
    SUPERVISED
        Targets [labels](LNN.html#lnn.Model.add_labels).
    UNCERTAINTY
        Reduces the uncertainty of bounds.

    """

    CONTRADICTION = auto()
    CUSTOM = auto()
    LOGICAL = auto()
    SUPERVISED = auto()
    UNCERTAINTY = auto()


class NeuralActivation(_AutoName):
    r"""An enumeration of [t-norms](https://en.wikipedia.org/wiki/T-norm#Prominent_examples) for alternate neural computations.

    Used to replace the activation of a connective with the specified activation.

    Parameters
    ----------
    Frechet
        Unweighted [Frechet](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inequalities) - only implemented for propositional logic.
    Godel
        Unweighted [Godel](https://en.wikipedia.org/wiki/G%C3%B6del_logic) - only implemented for propositional logic.
    Lukasiewicz
        Weighted [Łukasiewicz](https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic), supported for full first order logic.
    LukasiewiczTransparent : default
        Modification of weighted Łukasiewicz that supports learning by allowing identity gradients through the clamped regions.
    Product
        Unweighted [Product](https://en.wikipedia.org/wiki/Product_fuzzy_logic) - only implemented for propositional logic.

    """

    Frechet = auto()
    Godel = auto()
    LukasiewiczTransparent = auto()
    Lukasiewicz = auto()
    Product = auto()


class World(Enum):
    r"""An enumeration for world assumptions to for incomplete knowledge.

    See the discussion of [OPEN vs CLOSED](https://en.wikipedia.org/wiki/Open-world_assumption) world assumptions for more information.

    Parameters
    ----------
    AXIOM
        Formulae that follow assumptions of universally being TRUE
    CONTRADICTION
        Formulae that are in a contradictory state
    CLOSED
        Formulae that follow the closed world assumption
    FALSE
        Alias for CLOSED
    OPEN
        Formulae that follow the open world assumption

    """

    AXIOM = (1.0, 1.0)
    CONTRADICTION = (1.0, 0.0)
    CLOSED = (0.0, 0.0)
    FALSE = (0.0, 0.0)
    OPEN = (0.0, 1.0)
