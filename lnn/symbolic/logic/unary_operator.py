##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import logging
from typing import Union, Tuple, Set

import pandas as pd
import torch

from .connective_formula import _ConnectiveFormula
from .formula import Formula
from .neural_activation import _NeuralActivation
from .node_activation import _NodeActivation
from .variable import Variable
from .. import _gm
from ... import _utils
from ...constants import Fact, Direction, Bound
from torch.nn.parameter import Parameter

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
            This applies to the lower bound of a `Forall` and upper bound
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
        self.operator = None

        if isinstance(self, Forall):
            self.operator = "And"

        if isinstance(self, Exists):
            self.operator = "Or"

        self._set_activation(**kwds)
        self.neurons = []
        self.free_vars = [
            self.operands[0].unique_vars.index(v) for v in self.unique_vars
        ]

        if len(self.free_vars) == 0:
            self.neuron.weights = Parameter(
                torch.tensor([1.0]), self.neuron.weights.requires_grad
            )

        self._new_groundings = []

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
        operand = self.operands[0]

        if len(operand.grounding_table) == 0:
            return 0.0

        if len(self.free_vars) == 0:
            return self._fully_quantified_upward()

        input_bounds = operand.get_data()
        if input_bounds is None:
            return 0.0

        df, grouped = self._get_groupings()

        bound = None
        if not self.fully_grounded:
            bound = Bound.UPPER if isinstance(self, Forall) else Bound.LOWER

        result = 0
        bounds_table = []
        for indices in grouped:
            grounding = tuple(df.iloc[indices[0], :].tolist()[:-1])
            arity = len(indices)

            if self.grounding_table is None:
                self.grounding_table = {}

            if grounding in self.grounding_table:
                neuron_id = self.grounding_table[grounding]
                neuron = self.neurons[neuron_id]

                if neuron.arity != len(indices):
                    neuron = self._add_neuron(arity, neuron_id)
            else:
                self._add_grounding(grounding)
                neuron = self._add_neuron(len(indices))

            ib = input_bounds[indices].permute([1, 0])[None, :, :]
            result += neuron.aggregate_bounds([0], neuron.func(ib), bound)
            bounds_table.append(neuron.get_data())

        self.neuron.bounds_table = torch.vstack(bounds_table)

        return result

    def downward(self, **kwds) -> Union[torch.Tensor, None]:
        operand = self.operands[0]

        if len(operand.grounding_table) == 0:
            return 0.0

        if len(self.free_vars) == 0:
            return self._fully_quantified_downward()

        self._propagate_groundings()

        input_bounds = operand.get_data()
        if input_bounds is None:
            return 0.0

        df, grouped = self._get_groupings()
        bounds = input_bounds.detach().clone()
        for indices in grouped:
            grounding = tuple(df.iloc[indices[0], :].tolist()[:-1])
            ib = bounds[indices].permute([1, 0])[None, :, :]
            neuron_id = self.grounding_table[grounding]
            neuron = self.neurons[neuron_id]

            if neuron.arity != len(indices):
                neuron = self._add_neuron(len(indices), neuron_id)

            bounds[indices] = neuron.func_inv(neuron.get_data(), ib)[0].permute([1, 0])

        if isinstance(operand, _Quantifier):
            result = 0
            for i, neuron in enumerate(operand.neurons):
                result += neuron.aggregate_bounds([0], bounds[None, i])

            return result

        return operand.neuron.aggregate_bounds(df.index.to_list(), bounds)

    def _add_grounding(self, grounding: tuple[str]):
        self.grounding_table[grounding] = len(self.grounding_table)

    def _add_groundings(self, *groundings: tuple[str]):
        for g in groundings:
            if g not in self.grounding_table:
                self._add_grounding(g)
                self._add_neuron(len(g))
                self.neuron.extend_groundings(1)
                self._new_groundings.append(g)

    def _get_groupings(self):
        operand = self.operands[0]
        groundings = operand.grounding_table.keys()
        df = pd.DataFrame(groundings)
        df = df[self.free_vars]
        df["index"] = df.index
        grouped = df.groupby(by=self.free_vars)["index"].apply(list)
        return df, grouped

    def _create_neuron(self, arity):
        kwds = {"propositional": False, "arity": arity, "world": self.world}
        neuron = _NeuralActivation()(activation={"weights_learning": False}, **kwds)
        neuron.extend_groundings(1)
        neuron.func = neuron.activation(self.operator, direction=Direction.UPWARD)
        neuron.func_inv = neuron.activation(self.operator, direction=Direction.DOWNWARD)
        return neuron

    def _add_neuron(self, arity, index=None):
        neuron = self._create_neuron(arity)

        if index is None:
            self.neurons.append(neuron)
        else:
            self.neurons[index] = neuron

        return neuron

    def _fully_quantified_upward(self):
        operand = self.operands[0]

        input_bounds = operand.get_data()
        if input_bounds is None:
            return 0.0

        bound = None
        if not self.fully_grounded:
            bound = Bound.UPPER if isinstance(self, Forall) else Bound.LOWER

        input_bounds = input_bounds.permute([1, 0])[None, :, :]

        self.neuron = self._create_neuron(arity=len(operand.grounding_table))
        self.func = self.neuron.func
        result = self.neuron.aggregate_bounds([0], self.func(input_bounds), bound)
        self.neuron.bounds_table = self.neuron.bounds_table[0]
        return result

    def _fully_quantified_downward(self):
        operand = self.operands[0]

        if isinstance(operand, _Quantifier):
            result = 0
            for i, operand_neuron in enumerate(operand.neurons):
                ib = operand_neuron.get_data().permute([1, 0])[None, :, :]
                bounds = self.func_inv(self.get_data()[None, :], ib)
                bounds = bounds[0].permute([1, 0])
                result += operand_neuron.aggregate_bounds([0], bounds[None, 0])

            return result

        groundings = list(operand.grounding_table.values())
        bounds = self.func_inv(
            self.get_data().repeat(len(operand.get_data()), 1),
            operand.get_data()[:, :, None],
        )
        return operand.neuron.aggregate_bounds(groundings, bounds[..., 0])

    def _propagate_groundings(self):
        if len(self._new_groundings):
            operand = self.operands[0]
            groundings = operand.grounding_table.keys()
            new_groundings_df = pd.DataFrame(self._new_groundings)
            df = pd.DataFrame(groundings)
            new_groundings_df.columns = df[self.free_vars].columns
            df = df.drop(self.free_vars, axis=1)
            merged = _gm._full_outer_join(new_groundings_df, df)
            merged.sort_index(axis=1, inplace=True)

            for grounding in merged.itertuples(index=False, name=None):
                grounding_object = operand._ground(grounding)
                operand._add_groundings(grounding_object)

            self._new_groundings = []

    def _set_activation(self, **kwds):
        """Updates the neural activation according to grounding dimension size

        The computation of a quantifier is implemented via one of the weighed
            neurons, And/Or for Forall/Exists.
        At present, weighted quantifiers have not been well studied and is
            therefore turned off
        However the dimension of computation is different, computing over the
            groundings of the input formula instead of multiple formulae, since
            there can only be one formula to quantify over.
        The activation is therefore required to grow according to number of
            groundings present in the formula, which can grow as groundings
            propagate via inference.

        """
        kwds.setdefault("arity", len(self._grounding_set))
        kwds.setdefault("propositional", self.propositional)
        self.neuron = _NeuralActivation()(
            activation={"weights_learning": False}, **kwds
        )
        self.func = self.neuron.activation(self.operator, direction=Direction.UPWARD)
        self.func_inv = self.neuron.activation(
            self.operator, direction=Direction.DOWNWARD
        )

    @staticmethod
    def _has_free_variables(variables: Tuple[Variable, ...], operand: Formula) -> bool:
        r"""Returns True if the quantifier contains free variables."""
        return len(set(variables)) != len({*operand.unique_vars})

    @property
    def true_groundings(
        self,
    ) -> Set[Union[str, Tuple[str, ...]]]:
        r"""Returns a set of groundings that are True."""
        if isinstance(self.operands[0], _Quantifier):
            return self.operands[0].true_groundings

        return {
            g
            for g in self.operands[0].groundings
            if self.operands[0].state(g) is Fact.TRUE
        }

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
        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """

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

        Returns
        -------
        tightened_bounds : float
            The amount of bounds tightening or new information that is leaned by the inference step.

        """

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
        variables = (
            args[:-1]
            if len(args) > 1
            else args[-1][0].unique_vars
            if isinstance(args[-1], tuple)
            else args[-1].unique_vars
        )

        a = variables[0]
        formula = args[-1]

        if len(variables) > 1:
            args = [a, Exists(*variables[1:], formula, **kwds)]

        self.connective_str = "∃"
        super().__init__(*args, **kwds)


class Forall(_Quantifier):
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
    All_1 = Forall(birthdate(p, d)))
    All_2 = Forall(p, d, birthdate(p, d)))
    ```

    Free variables, quantifies over a subset of variables in the formula.
    ```python
    All = Forall(p, birthdate(p, d)))
    ```

    Warning
    -------
    Quantifier with free variables, not yet implemented. It is required that we quantify over all the variables given in the formula, either by specifying all the variables or but not specifying any variables - which is equivalent to quantifying over all variables.

    """

    def __init__(self, *args, **kwds):
        variables = (
            args[:-1]
            if len(args) > 1
            else args[-1][0].unique_vars
            if isinstance(args[-1], tuple)
            else args[-1].unique_vars
        )

        if len(variables) > 1:
            a = variables[0]
            formula = args[-1]
            args = [a, Forall(*variables[1:], formula, **kwds)]

        self.connective_str = "∀"
        super().__init__(*args, **kwds)
