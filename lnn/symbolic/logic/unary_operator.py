##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import logging
from typing import Union, Tuple, Set

import torch

from .connective_formula import _ConnectiveFormula
from .formula import Formula
from .grounding import _Grounding
from .neural_activation import _NeuralActivation
from .node_activation import _NodeActivation
from .variable import Variable
from .. import _gm
from ... import _utils
from ...constants import Fact, World, Direction, Bound

_utils.logger_setup()


class _UnaryOperator(_ConnectiveFormula):
    r"""Restrict operators to 1 input."""

    def __init__(self, *formula: Formula, **kwds):
        if len(formula) != 1:
            raise Exception(
                "Unary operator expect 1 formula as input, received " f"{len(formula)}"
            )
        super().__init__(*formula, arity=1, **kwds)


class _Quantifier(_UnaryOperator):
    r"""Symbolic container for quantifiers.

    Parameters
    ------------
    kwds : dict
        fully_grounded : bool
            specifies if a full upward inference can be done on a
            quantifier due to all the groundings being present inside it.
            This applies to the lower bound of a `ForAll` and upper bound
            of an `Exists`

    Attributes
    ----------
    fully_grounded : bool
    unique_var_slots : tuple of int
        returns the slot index of each unique variable

    """

    def __init__(self, *args, **kwds):
        variables = (
            args[:-1]
            if len(args) > 1
            else args[-1][0].unique_vars
            if isinstance(args[-1], tuple)
            else args[-1].unique_vars
        )
        super().__init__(args[-1], variables=variables, **kwds)
        self.fully_grounded = kwds.get("fully_grounded", False)
        self._grounding_set = set()
        self._set_activation(**kwds)

    @property
    def expanded_unique_vars(self):
        result = list(self.unique_vars)
        for idx, v in enumerate(self.variables):
            result.append(v)
        return tuple(result)

    @staticmethod
    def _unique_variables_overlap(
        source: Tuple[Variable, ...], destination: Tuple[Variable, ...]
    ) -> Tuple[Variable, ...]:
        """combines all predicate variables into a unique tuple
        the tuple is sorted by the order of appearance of variables in
        the operands
        """
        result = list()
        for dst_var in destination:
            if dst_var not in source:
                result.append(dst_var)
        return tuple(result)

    def upward(self, **kwds) -> float:
        r"""Upward inference from the operands to the operator.

        Parameters
        ----------
        lifted : bool, optional
            flag that determines if lifting should be done on this node.

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """

        # Create (potentially) new groundings from functions
        if not self.propositional:
            self._ground_functions()

        if kwds.get("lifted"):
            result = self.neuron.aggregate_world(self.operands[0].world)
            if result:
                if self.propositional:
                    self.neuron.reset_world(self.world)
                logging.info(
                    "↑ WORLD FREE-VARIABLE UPDATED "
                    f"TIGHTENED:{result} "
                    f"FOR:'{self.name}' "
                    f"FORMULA:{self.formula_number} "
                )
        else:
            n_groundings = len(self._grounding_set)
            input_bounds = self._upward_bounds(self.operands[0])
            if input_bounds is None:
                return 0.0
            if len(self._grounding_set) > n_groundings:
                self._set_activation(world=self.world)
            result = self.neuron.aggregate_bounds(
                None,
                self.func(input_bounds.permute([1, 0])),
                bound=(
                    (Bound.UPPER if isinstance(self, ForAll) else Bound.LOWER)
                    if not self.fully_grounded
                    else None
                ),
            )
            if result:
                logging.info(
                    "↑ BOUNDS UPDATED "
                    f"TIGHTENED:{result} "
                    f"FOR:'{self.name}' "
                    f"FORMULA:{self.formula_number} "
                )
            if self.is_contradiction():
                logging.info(
                    "↑ CONTRADICTION "
                    f"FOR:'{self.name}' "
                    f"FORMULA:{self.formula_number} "
                )
        return result

    def _set_activation(self, **kwds):
        """Updates the neural activation according to grounding dimension size

        The computation of a quantifier is implemented via one of the weighed
            neurons, And/Or for ForAll/Exists.
        At present, weighted quantifiers have not been well studied and is
            therefore turned off
        However the dimension of computation is different, computing over the
            groundings of the input formula instead of multiple formulae, since
            there can only be one formula to quantify over.
        The activation is therefore required to grow according to number of
            groundings present in the formula, which can grow as groundings
            propagate via inference.

        """
        operator = (
            "And"
            if isinstance(self, ForAll)
            else ("Or" if isinstance(self, Exists) else None)
        )
        kwds.setdefault("arity", len(self._grounding_set))
        kwds.setdefault("propositional", self.propositional)
        self.neuron = _NeuralActivation()(
            activation={"weights_learning": False}, **kwds
        )
        self.func = self.neuron.activation(operator, direction=Direction.UPWARD)

    @staticmethod
    def _has_free_variables(variables: Tuple[Variable, ...], operand: Formula) -> bool:
        r"""Returns True if the quantifier contains free variables."""
        return len(set(variables)) != len({*operand.unique_vars})

    @property
    def true_groundings(
        self,
    ) -> Set[Union[str, Tuple[str, ...]]]:
        r"""Returns a set of groundings that are True."""
        return {
            g
            for g in self.operands[0].groundings
            if self.operands[0].state(g) is Fact.TRUE
        }

    def _upward_bounds(self, operand: Formula) -> Union[torch.Tensor, None]:
        r"""Set Quantifier grounding table and return operand tensor."""
        operand_grounding_set = set(operand.grounding_table)
        if len(operand_grounding_set) == 0:
            return

        self._grounding_set = set(
            [
                grounding
                for grounding in operand_grounding_set
                if _gm.is_grounding_in_bindings(self, 0, grounding)
            ]
        )
        return operand.get_data(*self._grounding_set) if self._grounding_set else None

    def _groundings(self, groundings=None) -> Set[Union[str, Tuple[str, ...]]]:
        """Internal usage to extract groundings as _Grounding object"""
        return set(map(_Grounding.eval, groundings)) if groundings else self.groundings

    @property
    def groundings(self) -> Set[Union[str, Tuple[str, ...]]]:
        r"""returns a set of groundings as str or tuple of str"""
        return set(map(_Grounding.eval, self._grounding_set))

    def add_data(self, facts: Union[Tuple[float, float], Fact, Set]):
        super().add_data(facts)
        self._set_activation(world=self.world)


class Not(_UnaryOperator):
    r"""Symbolic Negation

    Returns a logical negation where inputs can be propositional,
    first-order logic predicates or other connectives.

    Parameters
    ------------
    formula : Formula
        accepts a unary input Formula

    Examples
    --------
    ```python
    # Propositional
    A = Proposition('A')
    Not(A)
    ```
    ```python
    # First-order logic
    x, y = Variables('x', 'y')
    A = Predicate('A', arity=2)
    Not(A(x, y)))
    ```

    """

    def __init__(self, formula: Formula, **kwds):
        self.connective_str = "¬"
        super().__init__(formula, **kwds)
        kwds.setdefault("propositional", self.propositional)
        self.neuron = _NodeActivation()(**kwds.get("activation", {}), **kwds)

    def upward(self, **kwds) -> float:
        r"""Upward inference from the operands to the operator.

        Parameters
        ----------
        lifted : bool, optional
            flag that determines if lifting should be done on this node.

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """

        # Create (potentially) new groundings from functions
        if not self.propositional:
            self._ground_functions()

        if kwds.get("lifted"):
            self.neuron.aggregate_world(
                tuple(
                    _utils.negate_bounds(torch.tensor(self.operands[0].world)).tolist()
                )
            )
        else:
            if self.propositional:
                groundings = {None}
            else:
                groundings = tuple(self.operands[0]._groundings)
                for g in groundings:
                    if g not in self.grounding_table:
                        self._add_groundings(g)
            bounds = self.neuron.aggregate_bounds(
                None, _utils.negate_bounds(self.operands[0].get_data(*groundings))
            )
            if self.is_contradiction():
                logging.info(
                    "↑ CONTRADICTION "
                    f"FOR:'{self.name}' "
                    f"FORMULA:{self.formula_number} "
                )
            return bounds

    def downward(self, **kwds) -> torch.Tensor:
        r"""Downward inference from the operator to the operands.

        Parameters
        ----------
        lifted : bool, optional
            flag that determines if lifting should be done on this node.

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """
        # Create (potentially) new groundings from functions
        if not self.propositional:
            self._ground_functions()

        if kwds.get("lifted"):
            self.operands[0].neuron.aggregate_world(
                tuple(_utils.negate_bounds(torch.tensor(self.world)).tolist())
            )
        else:
            if self.propositional:
                groundings = {None}
            else:
                groundings = tuple(self._groundings)
                for g in groundings:
                    if g not in self.operands[0]._groundings:
                        self.operands[0]._add_groundings(g)
            bounds = self.operands[0].neuron.aggregate_bounds(
                None, _utils.negate_bounds(self.get_data(*groundings))
            )
            if self.operands[0].is_contradiction():
                logging.info(
                    "↓ CONTRADICTION "
                    f"FOR:'{self.operands[0].name}' "
                    f"FROM:'{self.name}' "
                    f"FORMULA:{self.operands[0].formula_number} "
                    f"PARENT:{self.formula_number} "
                )
            return bounds


class Exists(_Quantifier):
    r"""Symbolic existential quantifier.

    When working with belief bounds - existential operators restrict upward inference to only work with the given formula's lower bound. Downward inference behaves as usual.

    Parameters
    ----------
    ``*variables`` : Variable
    formula : Formula
        The FOL formula to quantify over, may be a connective formula or a Predicate.


    Examples
    --------
    No free variables, quantifies over all of the variables in the formula.
    ```python
    Some_1 = Exists(birthdate(p, d)))
    Some_2 = Exists(p, d, birthdate(p, d)))
    ```

    Free variables, quantifies over a subset of variables in the formula.
    ```python
    Some = Exists(p, birthdate(p, d)))
    ```

    Warning
    -------
    Quantifier with free variables, not yet implemented. It is required that we quantify over all the variables given in the formula, either by specifying all the variables or but not specifying any variables - which is equivalent to quantifying over all variables.

    """

    def __init__(self, *args, **kwds):
        self.connective_str = "∃"
        super().__init__(*args, **kwds)


class ForAll(_Quantifier):
    r"""Symbolic universal quantifier.

    When working with belief bounds - universal operators restrict upward inference to only work with the given formula's upper bound. Downward inference behaves as usual.

    Parameters
    ----------
    ``*variables`` : Variable
    formula : Formula
        The FOL formula to quantify over, may be a connective formula or a Predicate.

    Examples
    --------
    No free variables, quantifies over all of the variables in the formula.
    ```python
    All_1 = ForAll(birthdate(p, d)))
    All_2 = ForAll(p, d, birthdate(p, d)))
    ```

    Free variables, quantifies over a subset of variables in the formula.
    ```python
    All = ForAll(p, birthdate(p, d)))
    ```

    Warning
    -------
    Quantifier with free variables, not yet implemented. It is required that we quantify over all the variables given in the formula, either by specifying all the variables or but not specifying any variables - which is equivalent to quantifying over all variables.

    """

    def __init__(self, *args, **kwds):
        self.connective_str = "∀"
        kwds.setdefault("world", World.AXIOM)
        super().__init__(*args, **kwds)

    def downward(self, **kwds) -> Union[torch.Tensor, None]:
        r"""Downward inference from the operator to the operands.

        Parameters
        ----------
        lifted : bool, optional
            flag that determines if lifting should be done on this node.

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """
        # Create (potentially) new groundings from functions
        if not self.propositional:
            self._ground_functions()

        if kwds.get("lifted"):
            result = self.operands[0].neuron.aggregate_world(self.world)
            if result:
                logging.info(
                    "↓ WORLD FREE-VARIABLE UPDATED "
                    f"TIGHTENED:{result} "
                    f"FOR:'{self.operands[0].name}' "
                    f"FROM:'{self.name}' "
                    f"FORMULA:{self.operands[0].formula_number} "
                    f"PARENT:{self.formula_number} "
                )
        else:
            if not self._grounding_set:
                return
            operand = self.operands[0]
            current_bounds = self.get_data()
            groundings = operand.grounding_table.keys()
            result = operand.neuron.aggregate_bounds(
                [operand.grounding_table.get(g) for g in groundings], current_bounds
            )
            if result:
                logging.info(
                    "↓ BOUNDS UPDATED "
                    f"TIGHTENED:{result} "
                    f"FOR:'{self.operands[0].name}' "
                    f"FROM:'{self.name}' "
                    f"FORMULA:{self.operands[0].formula_number} "
                    f"PARENT:{self.formula_number} "
                )
            if operand.is_contradiction():
                logging.info(
                    "↓ CONTRADICTION "
                    f"FOR:'{operand.name}' "
                    f"FROM:'{self.name}' "
                    f"FORMULA:{operand.formula_number} "
                    f"PARENT:{self.formula_number} "
                )
        return result
