##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from typing import Union, TypeVar

from .formula import Formula
from .node_activation import _NodeActivation
from .variable import Variable
from ... import utils
from ...constants import Fact


class _LeafFormula(Formula):
    r"""Specifies activation functionality as nodes instead of neurons.

    Assumes that all leaf formulae are propositions or predicates, therefore
        uses the _NodeActivation accordingly

    """

    def __init__(self, name, **kwds):
        self.model = kwds.get("model")
        super().__init__(name, syntax=name, **kwds)
        kwds.setdefault("propositional", self.propositional)
        self.neuron = _NodeActivation()(**kwds)


Model = TypeVar("lnn.Model")


class Predicate(_LeafFormula):
    r"""Creates a container for a predicate

    Stores a table of truths, with columns specified by the arity and rows
        indexed by the grounding

    Parameters
    ----------
    name : str
        Name of the predicate.
    model : lnn.Model
        Model that the predicate is inserted into.
    arity : int, optional
        If unspecified, assumes a unary predicate.

    Examples
    --------
    ```python
    model = Model()
    P1 = Predicate('P1', model)
    P2 = Predicate('P2', model, arity=2)
    ```

    """

    def __init__(self, name: str, model: Model, arity: int = 1, **kwds):
        if arity is None:
            raise Exception(f"arity expected as int > 0, received {arity}")
        super().__init__(
            name=name, model=model, arity=arity, propositional=False, **kwds
        )
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


def Predicates(*predicates: str, model: Model, arity=1, **kwds):
    r"""Instantiates multiple predicates and adds it to the model.

    Examples
    --------
    ```python
    P1, P2 = Predicates("P1", "P2", model=model, arity=2)
    ```

    """
    return utils.return1(
        [Predicate(p, model=model, arity=arity, **kwds) for p in predicates]
    )


class Proposition(_LeafFormula):
    r"""Creates propositional containers

    Stores and retrieves single truth bounds instead of tables as in FOL case

    Parameters
    ----------
    name : str
        name of the proposition.
    model : lnn.Model
        Model that the proposition is inserted into.

    Examples
    --------
    ```python
    model = Model()
    P = Proposition('Person', model)
    ```

    """

    def __init__(self, name: str, model: Model, **kwds):
        super().__init__(name=name, model=model, arity=1, propositional=True, **kwds)

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
    model = Model()
    P1, P2 = Propositions("P1", "P2", model=model)
    ```

    """
    return utils.return1([Proposition(p, **kwds) for p in propositions])
