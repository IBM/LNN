##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from typing import Union, Tuple, Set

from .connective_neuron import _ConnectiveNeuron
from .formula import Formula
from ... import _utils
from ...constants import Direction, NeuralActivation

_utils.logger_setup()


class _BinaryNeuron(_ConnectiveNeuron):
    r"""Restrict neurons to 2 inputs."""

    def __init__(self, *formula, **kwds):
        if len(formula) != 2:
            raise Exception(
                "Binary neurons expect 2 formulae as inputs, received "
                f"{len(formula)}"
            )
        super().__init__(*formula, arity=2, **kwds)


class Implies(_BinaryNeuron):
    r"""
    Symbolic binary [implication](https://en.wikipedia.org/wiki/Logical_implication).

    Returns a logical implication node where inputs can be [propositions](
    LNN.html#lnn.Proposition), `called` first-order logic [predicates](
    LNN.html#lnn.Predicate) or any other [connective formulae](
    LNN.html#symbolic-structure). Propositional inputs yield a propositional node,
    whereas if any input is a predicate it will cause the connective to increase its
    dimension to also be a FOL node (i.e. stores a table of facts).

    Parameters
    ----------
    lhs : Formula
        The left-hand side formula of the binary inputs to the connective.
    rhs : Formula
        The right-hand side formula of the binary inputs to the connective.
    name : str, optional
        A custom name for the node to be used for identification and custom printing. If unspecified, defaults the structure of the node.
    activation : dict, optional
        Parameters given as a dictionary of configuration options, see the [neural configuration](../usage.html#neural-configuration) for more details

    Examples
    --------
    ```python
    # Propositional
    A, B = Propositions('A', 'B')
    Implies(A, B)
    ```
    ```python
    # First-order logic
    x, y = Variables('x', 'y')
    A = Predicate('A')
    B = Predicate('B', arity=2)
    Implies(A(x), B(x, y)))
    ```

    """

    def __init__(self, lhs: Formula, rhs: Formula, **kwds):
        self.connective_str = "→"
        kwds.setdefault("activation", {})
        kwds["activation"].setdefault("bias_learning", True)
        super().__init__(lhs, rhs, **kwds)


class Iff(_BinaryNeuron):
    r"""Symbolic Equivalence - a bidirectional binary implication or IFF
    (if and only if) node.

    Returns a logical bidirectional equivalence node where inputs can be [
    propositions](LNN.html#lnn.Proposition), `called` first-order logic [predicates](
    LNN.html#lnn.Predicate) or any other [connective formulae](
    LNN.html#symbolic-structure). Propositional inputs yield a propositional node,
    whereas if any input is a predicate it will cause the connective to increase its
    dimension to also be a FOL node (i.e. stores a table of facts).

    Parameters
    ----------
    lhs : Formula
        The left-hand side formula of the binary inputs to the connective.
    rhs : Formula
        The right-hand side formula of the binary inputs to the connective.
    name : str, optional
        A custom name for the node to be used for identification and custom printing. If unspecified, defaults the structure of the node.
    activation : dict, optional
        parameters given as a dictionary of configuration options, see the [neural configuration](../usage.html#neural-configuration) for more details

    Examples
    --------
    ```python
    # Propositional
    A, B = Propositions('A', 'B')
    Iff(A, B)
    ```
    ```python
    # First-order logic
    x, y = Variables('x', 'y')
    A = Predicate('A')
    B = Predicate('B', arity=2)
    Iff(A(x), B(x, y)))
    ```

    """

    def __init__(self, lhs: Formula, rhs: Formula, **kwds):
        self.connective_str = "∧"
        self.Imp1, self.Imp2 = Implies(lhs, rhs, **kwds), Implies(rhs, lhs, **kwds)
        super().__init__(self.Imp1, self.Imp2, **kwds)
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
        self.Imp1.upward(groundings, **kwds)
        self.Imp2.upward(groundings, **kwds)
        return super().upward(groundings, **kwds)

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
        self.Imp1.downward(index, groundings, **kwds)
        self.Imp2.downward(index, groundings, **kwds)
        return result
