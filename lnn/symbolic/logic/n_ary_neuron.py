##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from .formula import Formula
from .connective_neuron import _ConnectiveNeuron
from ... import _utils
from ...constants import NeuralActivation

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
        self.connective_str = self.get_connective_str(
            kwds["activation"].get("type", None)
        )
        super().__init__(*formula, **kwds)

    @staticmethod
    def get_connective_str(type: NeuralActivation) -> str:
        return f"{type.name[0]}∧" if type else "∧"


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
        self.connective_str = self.get_connective_str(
            kwds["activation"].get("type", None)
        )
        super().__init__(*formula, **kwds)

    @staticmethod
    def get_connective_str(type: NeuralActivation) -> str:
        return f"{type.name[0]}∨" if type else "∨"
