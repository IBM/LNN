##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import copy
import itertools
from typing import Set, Union, Tuple

from .formula import Formula
from .unary_operator import Not
from .connective_neuron import _ConnectiveNeuron
from ... import _utils
from ...constants import Direction, NeuralActivation

_utils.logger_setup()


class _NAryNeuron(_ConnectiveNeuron):
    r"""N-ary connective neuron."""

    def __init__(self, *formula, **kwds):
        super().__init__(*formula, arity=len(formula), **kwds)


class And(_NAryNeuron):
    r"""Symbolic n-ary [conjunction](https://en.wikipedia.org/wiki/Logical_conjunction).

    Returns a logical conjunction where inputs can be [propositions](LNN.html#lnn.Proposition), `called` first-order logic [predicates](LNN.html#lnn.Predicate) or any other [connective formulae](LNN.html#symbolic-structure).
    Propositional inputs yield a propositional node, whereas if any input is a predicate it will cause the connective to increase its dimension to also be a FOL node (i.e. stores a table of facts).

    Parameters
    ----------
    ``*formula`` : Formula
        A variable length argument list that accepts any number of input formulae objects as arguments.
    name : str, optional
        A custom name for the node to be used for identification and custom printing. If unspecified, defaults the structure of the node.
    activation : dict, optional
        Parameters given as a dictionary of configuration options, see the [neural configuration](../usage.html#neural-configuration) for more details

    Examples
    --------
    ```python
    # Propositional
    A, B, C = Propositions('A', 'B', 'C')
    And(A, B, C)
    ```
    ```python
    # First-order logic
    x, y = Variables('x', 'y')
    A, C = Predicates('A', 'C')
    B = Predicate('B', arity=2)
    And(A(x), B(x, y), C(y)))
    ```

    """

    def __init__(self, *formula: Formula, **kwds):
        kwds.setdefault("activation", {})
        self.connective_str = "∧"
        super().__init__(*formula, **kwds)


class Or(_NAryNeuron):
    r"""Symbolic n-ary [disjunction](https://en.wikipedia.org/wiki/Logical_disjunction).

    Returns a logical disjunction where inputs can be [propositions](LNN.html#lnn.Proposition), `called` first-order logic [predicates](LNN.html#lnn.Predicate) or any other [connective formulae](LNN.html#symbolic-structure).
    Propositional inputs yield a propositional node, whereas if any input is a predicate it will cause the connective to increase its dimension to also be a FOL node (i.e. stores a table of facts).

    Parameters
    ----------
    ``*formula`` : Formula
        A variable length argument list that accepts any number of input formulae objects as arguments.
    name : str, optional
        A custom name for the node to be used for identification and custom printing. If unspecified, defaults the structure of the node.
    activation : dict, optional
        Parameters given as a dictionary of configuration options, see the [neural configuration](../usage.html#neural-configuration) for more details

    Examples
    --------
    ```python
    # Propositional
    A, B, C = Propositions('A', 'B', 'C')
    Or(A, B, C)
    ```
    ```python
    # First-order logic
    x, y = Variables('x', 'y')
    A, C = Predicates('A', 'C')
    B = Predicate('B', arity=2)
    Or(A(x), B(x, y), C(y)))
    ```

    """

    def __init__(self, *formula, **kwds):
        kwds.setdefault("activation", {})
        self.connective_str = "∨"
        super().__init__(*formula, **kwds)


class XOr(_NAryNeuron):
    r"""
    Symbolic nAry [Exclusive or](https://en.wikipedia.org/wiki/Exclusive_or).

    Returns a logical exclusive disjunction node where inputs can be [propositions](
    LNN.html#lnn.Proposition), `called` first-order logic [predicates](
    LNN.html#lnn.Predicate) or any other [connective formulae](
    LNN.html#symbolic-structure). Propositional inputs yield a propositional node,
    whereas if any input is a predicate it will cause the connective to increase its
    dimension to also be a FOL node (i.e. stores a table of facts).

    Parameters
    ----------
    ``*formula`` : Formula
        A variable length argument list that accepts any number of input formulae objects as arguments.
    name : str, optional
        A custom name for the node to be used for identification and custom printing. If unspecified, defaults the structure of the node.
    activation : dict, optional
        Parameters given as a dictionary of configuration options, see the [neural configuration](../usage.html#neural-configuration) for more details

    Examples
    --------
    ```python
    # Propositional
    A, B, C = Propositions('A', 'B', 'C')
    XOr(A, B, C)
    ```
    ```python
    # First-order logic
    x, y = Variables('x', 'y')
    A, C = Predicates('A', 'C')
    B = Predicate('B', arity=2)
    XOr(A(x), B(x, y), C(y)))
    ```

    """

    def __init__(self, *formula, **kwds):
        self.connective_str = "∧"
        kwds.setdefault("activation", {})
        conjunction_activation = copy.copy(kwds["activation"])
        conjunction_activation.setdefault("bias_learning", False)
        conjunction_activation.setdefault("weights_learning", False)
        self.conjunctions = [
            And(*f, activation=conjunction_activation)
            for f in itertools.combinations(formula, 2)
        ]
        self.negations = [Not(f) for f in self.conjunctions]
        self.disjunction = Or(*formula, **kwds)
        super().__init__(*self.negations, self.disjunction, **kwds)
        self.func = self.neuron.activation("And", direction=Direction.UPWARD)
        self.func_inv = self.neuron.activation("And", direction=Direction.DOWNWARD)

    def upward(
        self, groundings: Set[Union[str, Tuple[str, ...]]] = None, **kwds
    ) -> float:
        r"""Upward inference from the operands to the operator.

        Parameters
        ----------
        groundings : str or tuple of str
            restrict upward inference to a specific grounding or row in the truth table

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """
        [node.upward(**kwds) for node in self.conjunctions]
        [node.upward(**kwds) for node in self.negations]
        self.disjunction.upward(**kwds)
        return super().upward(**kwds)

    def downward(
        self,
        index: int = None,
        groundings: Set[Union[str, Tuple[str, ...]]] = None,
        **kwds,
    ) -> float:
        r"""Downward inference from the operator to the operands.

        Parameters
        ----------
        index : int, optional
            restricts downward inference to an operand at the specified index. If unspecified, all operands are updated.
        groundings : str or tuple of str, optional
            restrict upward inference to a specific grounding or row in the truth table

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """
        result = super().downward(index, groundings, **kwds)
        [node.downward(**kwds) for node in self.negations]
        [node.downward(**kwds) for node in self.conjunctions]
        self.disjunction.downward(**kwds)
        return result
