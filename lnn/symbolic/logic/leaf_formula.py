##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from typing import Union

from .formula import Formula
from .node_activation import _NodeActivation
from .variable import Variable
from ... import _utils, utils
from ...constants import Fact

_utils.logger_setup()


class _LeafFormula(Formula):
    r"""Specifies activation functionality as nodes instead of neurons.

    Assumes that all leaf formulae are propositions or predicates, therefore
        uses the _NodeActivation accordingly

    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        kwds.setdefault("propositional", self.propositional)
        self.neuron = _NodeActivation()(**kwds.get("activation", {}), **kwds)


class Predicate(_LeafFormula):
    r"""Creates a container for a predicate

    Stores a table of truths, with columns specified by the arity and rows
        indexed by the grounding

    Parameters
    ----------
    name : str
        name of the predicate
    arity : int, optional
        If unspecified, assumes a unary predicate

    Examples
    --------
    ```python
    P1 = Predicate('P1')
    P2 = Predicate('P2', arity=2)
    ```

    """

    def __init__(self, name: str, arity: int = 1, **kwds):
        if arity is None:
            raise Exception(f"arity expected as int > 0, received {arity}")
        super().__init__(name=name, arity=arity, propositional=False, **kwds)
        self._update_variables(tuple(Variable(f"?{i}") for i in range(self.arity)))

    def add_data(self, facts: Union[dict, set]):
        r"""Populate predicate with facts

        Facts required in dict or set
            - dict for grounding-based facts
            - set for broadcasting facts across all groundings
              requires a set of 1 item
        dict keys for groundings and values as facts
        tuple facts required in bounds form `(Lower, Upper)`

        """
        super().add_data(facts)

    def __call__(self, *args, **kwds):
        r"""A called first-order logic predicate

        This correctly instantiates a predicate with variables - which is required when
        using the predicate in a compound formula. Calling the predicate allows the LNN
        to construct the inheritance tree from subformulae.

        Examples
        --------
        ```python
        P, Q = Predicates('P', 'Q')
        x, y = Variables('x', 'y')
        And(P(x), Q(y))  # calling both predicates
        ```
        Here the conjunction inherits its variables from all subformulae, treating it as
        an ordered unique collection (list).
        """
        return super().__call__(*args, **kwds)


def Predicates(*predicates: str, **kwds):
    r"""Instantiates multiple predicates.

    Examples
    --------
    ```python
    P1, P2 = Predicates("P1", "P2", arity=2)
    ```

    """
    return utils.return1([Predicate(p, **kwds) for p in predicates])


class Proposition(_LeafFormula):
    r"""Creates propositional containers

    Stores and retrieves single truth bounds instead of tables as in FOL case

    Parameters
    ----------
    name : str
        name of the proposition

    Examples
    --------
    ```python
    P = Proposition('Person')
    ```

    """

    def __init__(self, name: str, **kwds):
        super().__init__(name=name, arity=1, propositional=True, **kwds)

    def add_data(self, fact: Union[Fact, bool]):
        """Populate proposition with facts

        Facts required in bool, tuple or None
        None fact assumes `Unknown`
        tuple fact required in bounds form `(Lower, Upper)`

        """
        super().add_data(fact)


def Propositions(*propositions: str, **kwds):
    r"""Instantiates multiple propositions.

    Examples
    --------
    ```python
    P1, P2 = Propositions("P1", "P2")
    ```

    """
    return utils.return1([Proposition(p, **kwds) for p in propositions])
