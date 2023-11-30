##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import logging
import itertools
import typing
from typing import Optional, Union, Tuple, Iterator, Set, List, Dict

from abc import ABC

from .variable import Variable
from .. import _trace
from ... import _utils, _exceptions, utils
from ...constants import Fact, World

import copy
import torch
import numpy as np

_utils.get_logger()
subclasses: typing.Dict[str, object] = {}


def _isinstance(obj, class_str) -> bool:
    """
    Returns True if an object is an instance of a class give that class name as a
    string, otherwise False. The check is performed using the subclasses dictionary.
    To see what the subclasses dictionary is populated with, refer to the init file of
    this module.
    """
    return isinstance(obj, subclasses[class_str])


class Formula(ABC):
    r"""Symbolic container for a generic formula."""

    def __new__(cls, *args, **kwds):
        instance = super(Formula, cls).__new__(cls)
        instance.__init__(*args, **kwds)
        return instance._add_to_model()

    def __init__(
        self,
        *formulae: "Formula",
        name: Optional[str] = "",
        syntax: str = None,
        arity: int = None,
        **kwds,
    ):
        # placeholder for neural variables and functions
        self.initialised = True
        self.neuron = None
        self.func = None
        self.func_inv = None
        if not hasattr(self, "formula_number"):
            self.formula_number = None
        if not hasattr(self, "operands_by_number"):
            self.operands_by_number = list()
        self.congruent_nodes = list()

        # debugging
        self.logger = _utils.get_logger()

        # construct edge and operand list for each formula
        self.edge_list = list()
        self.operands: List[Formula] = list()

        # formula arity
        self.arity: int = arity

        # inherit propositional, variables, and graphs
        self.propositional: bool = kwds.get("propositional")
        self.variables: Tuple[Variable, ...] = kwds.get("variables")
        if not _isinstance(self, "_LeafFormula"):
            self._inherit_from_subformulae(*formulae)

        # formula naming
        self.name = name if name else ""
        if syntax:
            self.syntax = syntax
            if not name:
                self.name = syntax
        if not self.name or not syntax:
            self.syntax = self._formula_syntax()
            self.name = name if name else self._formula_syntax(lambda x: x.name)

        # formula grounding table maps grounded objects to table rows
        self.grounding_table = None if self.propositional else dict()

    def _add_to_model(self):
        r"""Inherits model from operands and insert the formula into the model."""
        if not hasattr(self, "model"):
            self.model = self.operands[0].model
        if self not in self.model:
            self.model.add_knowledge(self)
            return self
        else:
            return self.model[self]

    ##
    # External function definitions
    ##

    def add_facts(self, *args, **kwds):
        raise NameError(f"`add_facts` is deprecated, use `add_data` instead")

    def add_data(
        self,
        data: Union[
            Union[bool, Fact, float, Tuple[float, float]],
            Dict[
                Union[str, Tuple[str, ...]],
                Union[bool, Fact, float, Tuple[float, float]],
            ],
        ],
    ):
        r"""Add data to the formula in the form of classical facts or belief bounds.

        Data given is a Fact or belief bounds assumes a propositional formula.
        Data given in a dict assumes a first-order logic formula,
            keyed by the grounding and a value given as a Fact or belief bounds.

        Parameters
        ----------
        data : Fact, belief bounds or dict
            For propositional formulae, truths is given as either Facts or belief bounds. These beliefs can be given as a bool, float or a float-range, i.e. a tuple of 2 floats. For first-order logic formula, inputs truths are given as a dict. It is keyed by the grounding (a str for unary formlae or tuple of strings of larger arities), with values also as Facts or bounds on beliefs.

        Examples
        --------
        ```python
        # propositional
        P, Q = Propositions("P", "Q")
        P.add_data(Fact.TRUE)
        Q.add_data((.1, .4))
        ```
        ```python
        # first-order logic
        Person = Predicate("Person")
        Person.add_data({
            "Barack Obama": Fact.TRUE,
            "Bo": (.1, .4)
        })

        # FOL with arity > 2
        BD = Predicate("Birthdate", 2)
        BD.add_data({
            ("Barack Obama", "04 August 1961"): Fact.TRUE,
            ("Bo", "09 October 2008"): (.6, .75)
        })
        ```

        """
        if self.propositional:  # Propositional facts
            if isinstance(data, bool):
                data = Fact.TRUE if data else Fact.FALSE

            _exceptions.AssertBounds(data)
            self.neuron.add_data(data)
            return

        # FOL facts
        if not isinstance(data, dict):  # facts given per grounding
            raise Exception(
                "FOL facts should be from [dict, set], "
                f'"{self}" received {type(data)}'
            )

        # replace fact keys (str -> tuple[str])
        groundings = tuple(data.keys())
        for g in groundings:
            bounds = data.pop(g)

            if isinstance(bounds, bool):
                bounds = Fact.TRUE if bounds else Fact.FALSE

            data[self._ground(g)] = bounds

        # add missing groundings to `grounding_table`
        groundings = tuple(data.keys())
        self._add_groundings(*groundings)

        # set facts for all groundings
        table_facts = {self.grounding_table[g]: data[g] for g in groundings}
        self.neuron.add_data(table_facts)

    def add_labels(
        self,
        labels: Union[
            Union[Fact, Tuple[float, float]],
            Dict[Union[str, Tuple[str, ...]], Union[Fact, Tuple[float, float]]],
        ],
    ):
        r"""Store labels as data within the symbolic container.

        Labels given is a Fact or belief bounds assumes a propositional formula.
        Labels given in a dict assumes a first-order logic formula,
            keyed by the grounding and a value given as a Fact or belief bounds.

        Parameters
        ----------
        labels : Fact, belief bounds or dict
            For propositional formulae, facts are given as either Facts or bounds on beliefs (a tuple of 2 floats). For first-order logic formula, inputs are given as a dict. It is keyed by the grounding (a str for unary formlae or tuple of strings of larger arities), with values also as Facts or bounds on beliefs.

        Examples
        --------
        ```python
        # propositional
        P, Q = Propositions("P", "Q")
        P.add_labels(Fact.TRUE)
        Q.add_labels((.1, .4))
        ```
        ```python
        # first-order logic
        Person = Predicate("Person")
        Person.add_labels({
            "Barack Obama": Fact.TRUE,
            "Bo": (.1, .4)
        })

        # FOL with arity > 2
        BD = Predicate("Birthdate", 2)
        BD.add_labels({
            ("Barack Obama", "04 August 1961"): Fact.TRUE,
            ("Bo", "09 October 2008"): (.6, .75)
        })
        ```

        """
        if self.propositional:  # Propositional labels
            _exceptions.AssertBounds(labels)
            self.labels = _utils.fact_to_bounds(labels, self.propositional)

        else:  # FOL labels
            if not hasattr(self, "labels"):
                self.labels = dict()
            if isinstance(labels, dict):  # labels given per grounding
                _labels = copy.copy(labels)
                for g in tuple(_labels.keys()):
                    # set labels for groundings, replace str keys -> Groundings
                    self.labels[self._ground(g)] = _utils.fact_to_bounds(
                        _labels.pop(g), self.propositional
                    )
            else:
                raise Exception(
                    "FOL facts should be from [dict, set], "
                    f'"{self}" received {type(labels)}'
                )

    def flush(self):
        r"""Set all facts in formula to `Fact.UNKNOWN`."""
        self.neuron.flush()

    def get_facts(self, *args, **kwds):
        raise NameError(f"`get_facts` is deprecated, use `get_data` instead")

    def get_data(
        self, *groundings: Union[str, Tuple[str, ...], tuple[str]]
    ) -> torch.Tensor:
        r"""Returns the current beliefs of the formula.

        Uses the given groundings or the entire `groundind_table` to slice out the
        current belief bounds from the `bounds_table`.

        Parameters
        -------
        ``*groundings``: str or tuple of str
            NB - if unspecified, defaults to returning all the facts for the given formula. If specified, the table will be ordered according to the input grounding order.

        """
        if self.propositional or len(groundings) == 0:
            return self.neuron.get_data()
        table_rows = [self.grounding_table.get(self._ground(g)) for g in groundings]
        return self.neuron.get_data(table_rows, default=True)

    def get_labels(self, *groundings: str) -> torch.Tensor:
        r"""Returns the stored labels from the symbolic container."""
        if self.propositional or len(groundings) == 0:
            return self.labels
        result = torch.stack([self.labels.get(self._ground(g)) for g in groundings])
        return result[0] if len(groundings) == 1 else result

    @property
    def groundings(self) -> Set[Union[str, Tuple[str, ...]]]:
        r"""Returns the groundings to the user as str or tuple of str."""
        if self.grounding_table:
            return set(self.grounding_table.keys())

        return set()

    @property
    def is_classically_resolved(self):
        r"""Checks if the query node is in a classical state besides `Fact.UNKNOWN`."""
        if self.propositional and _utils.is_classical_proposition(
            tuple(self.get_data().tolist())
        ):
            return True
        return False

    def is_contradiction(
        self, bounds: torch.Tensor = None, stacked=False
    ) -> torch.BoolTensor:
        r"""Check if there are any bounds in contradiction."""
        return self.contradicting_bounds(bounds, stacked).any()

    def contradicting_bounds(
        self, bounds: torch.Tensor = None, stacked=False
    ) -> torch.BoolTensor:
        r"""Check which bounds are in contradiction."""
        if stacked:
            tensor = torch.logical_or(
                self.neuron.is_contradiction(bounds[..., 0]),
                self.neuron.is_contradiction(bounds[..., 1]),
            )
            return tensor.type(torch.bool)

        return self.neuron.is_contradiction(bounds)

    def contradicting_stacked_bounds(
        self, bounds: torch.Tensor = None
    ) -> torch.BoolTensor:
        r"""Check if bounds are in contradiction."""

    def is_equal(self, other: "Formula") -> bool:
        r"""Returns True if two nodes are symbolically equivalent.

        Formulae can be symbolically equivalent yet neurally different, example:
        ```python
        f = And(A, B, activation={"type": NeuralActivation.Lukasiewicz})
        g = And(A, B, activation={"type": NeuralActivation.Godel})
        ```
        The above will return `True`, since `f` is symbolically equivalent to `g`
        despite different neural configurations.

        """
        if self is other:
            return True
        result = list()
        for f in self.congruent_nodes:
            if _isinstance(f, "Equal"):
                result += [other.syntax == operand.syntax for operand in f.operands]
        return any(result)

    def is_unweighted(self) -> bool:
        return all([self.neuron.bias == w for w in self.neuron.weights])

    def negation_absorption(
        self, store: bool = True
    ) -> (List[int], Tuple[Tuple["Formula"]]):
        n_negations = []
        edge_replace = []
        operands = []

        def recurse(formula):
            for operand in formula.operands:
                if _isinstance(operand, "Not"):
                    recurse(operand)
                    root_idx = len(n_negations) - 1
                    n_negations[root_idx] += 1
                    condition = [True] * len(self.neuron.weights)
                    condition[root_idx] = bool(root_idx % 2)
                    if store:
                        self.neuron.update_weights_where(
                            condition, -self.neuron.weights
                        )
                        self.operands[root_idx] = self.operands[root_idx].operands[0]
                    edge_replace.append(
                        (self.edge_list[root_idx], (self, self.operands[root_idx]))
                    )
                else:
                    n_negations.append(0)
                    operands.append(operand)

        recurse(self)
        if store:
            if edge_replace:
                self.logger.info(f"ABSORBED NEGATIONS INTO WEIGHTS FOR: '{self.name}'")
            return edge_replace, n_negations
        else:
            return operands, edge_replace, n_negations

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return self.neuron.named_parameters()

    def parameters(self) -> Iterator[torch.Tensor]:
        return self.neuron.parameters()

    def params(
        self, *params: str, detach: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        result = list()
        for param in params:
            if param in dict(self.neuron.named_parameters()):
                result.append(
                    getattr(self.neuron, param).clone().detach()
                    if detach
                    else getattr(self.neuron, param)
                )
            else:
                raise ValueError(f"{self} has no attribute: {param}")
        return result[0] if len(params) == 1 else result

    def print(
        self,
        header_len: int = 50,
        roundoff: int = 5,
        state: bool = False,
        params: bool = False,
        grads: bool = False,
        numbering: bool = False,
    ):
        r"""Print the states of groundings in a formula.

        ```
        OPEN   Proposition: A                 UNKNOWN (L, U)

        CLOSED Proposition: B                   FALSE (L, U)

        OPEN   Predicate: P1 (x)
        "const_1"                                TRUE (L, U)
        "const_2"                       CONTRADICTION (L, U)
        "const_3"                               FALSE (L, U)

        OPEN   Predicate: P2 (x, y)
        ("const_1", "const_5")                   TRUE (L, U)
        ("const_2", "const_6")          CONTRADICTION (L, U)
        ("const_3", "const_7")                  FALSE (L, U)

        OPEN   And: And_0 (x, y)
        Bindings: P1 (x: "const_1"), P2 (x: 'const_2', y: ['const_3', ...])
        ('const_1', 'const_3')                   TRUE (L, U)
        ('const_2', 'const_3')          CONTRADICTION (L, U)
        ('const_1', 'const_7')                  FALSE (L, U)

        TRUE   Forall: Forall_0 (y)           UNKNOWN (L, U)
        ```

        """

        def state_wrapper(grounding: Tuple[str]):
            return f"{grounding}"

        def round_bounds(grounding=None):
            data = self.get_data(grounding)
            if len(data.shape) > 1:
                data = data[0]
            return tuple([round(r, roundoff) for r in data.tolist()])

        header = (
            f"{str(self.world_state(True))} "
            f"{self.__class__.__name__}"
            f": {self.name}"
        )
        if params:
            params = dict()
            for name, symbol in _utils.param_symbols.items():
                if hasattr(self.neuron, name):
                    val = getattr(self.neuron, name)
                    params[symbol] = f"{np.around(val.tolist(), roundoff)}" + (
                        f" grads {np.around(val.grad.tolist(), roundoff)}"
                        if grads and val.grad is not None
                        else ""
                    )
            params = "params  " + ",  ".join([f"{k}: {v}" for k, v in params.items()])
        else:
            params = ""
        number = (
            f"{self.formula_number}"
            f"{' ' + str(self.operands_by_number) if self.operands_by_number else ''}: "
            if numbering and self.formula_number is not None
            else ""
        )

        # print propositional node - single bounds
        J = self.neuron.J if hasattr(self.neuron, "J") else ""
        if self.propositional:
            states = (
                f"{header:<{header_len}} "
                f"{self.state().name if state else '':>14} "
                f"{round_bounds()}"
                f"{f'  J: {J}' if J and str(self.neuron) == 'J()' else ''}\n"
                f"{params}"
            )
            print(f"{number}{states}")

        # print FOL node - table of bounds
        else:
            facts = (
                [""]
                if self.grounding_table is None
                else (
                    [
                        (
                            f"{state_wrapper(g):{header_len}} "
                            f"{self.state(g).name if state else '':>14} "
                            f"{round_bounds(g)}\n"
                        )
                        for g in self.grounding_table
                    ]
                )
            )
            binder = list()
            for op_idx, op in enumerate(self.operands):
                if not self.propositional and self._has_bindings(op_idx):
                    for i in range(len(self.var_remap[op_idx])):
                        if None not in self.bindings[op_idx][i]:
                            binder.append(
                                f"{str(op)} "
                                "{"
                                f"{str(self.operand_map[op_idx][i])}: "
                                f"{list(map(str, self.bindings[op_idx][i]))}"
                                "}"
                            )
            bind_str = ("\nBindings: " + ", ".join(binder)) if binder else ""
            header = f"{header} {bind_str}"
            params = f"\n{params}" if params else ""
            header = f"{header}{params}"
            result = f"{header}\n" + "".join(facts)
            print(f"{number}{result}")

    def project_params(self):
        self.neuron.project_params()

    def rename(self, name: str):
        self.name = name

    def reset_bounds(self):
        self.neuron.reset_bounds()

    def reset_world(self, world: World):
        _exceptions.AssertBoundsLen(world)
        self.world = world if isinstance(world, tuple) else world.value
        self.neuron.reset_world(world)

    def set_formula_number(self, idx) -> int:
        self.formula_number = idx
        for operand in self.operands:
            if operand.formula_number is None:
                idx = operand.set_formula_number(idx + 1)
            self.operands_by_number.append(operand.formula_number)
        return idx

    def set_negative_weights(
        self, is_negative: bool = True, store: bool = True
    ) -> (List[int], Tuple[Tuple["Formula"]]):
        """absorb `Not` into weights and allow negative computation"""
        self.neuron.set_negative_weights(is_negative)
        return self.negation_absorption(store=store)

    def set_propositional(self, propositional: bool):
        r"""Set's the neuron's world parameter."""
        self.propositional = propositional

    @property
    def shape(self) -> torch.Size:
        return self.neuron.bounds_table.shape

    def state(
        self,
        groundings: Union[
            str, Tuple[str, ...], List[str], List[Tuple[str, ...]]
        ] = None,
        to_bool: bool = False,
        bounds: torch.Tensor = None,
    ) -> Union[torch.Tensor, Dict[Union[str, Tuple[str, ...]], torch.Tensor]]:
        r"""Returns the state of a single grounded fact.

        if to_bool flag is True, will map classical Facts to bool:
            i.e. {Fact.True: True, FALSE': False}
        The rest of the node states will return as str

        Notes
        -----
        see section [F.2](https://arxiv.org/abs/2006.13155) for more
            information on node states

        """
        if self.propositional:
            result = self.neuron.state()
        else:
            if bounds is None:
                if groundings is None or isinstance(groundings, list):
                    result = {
                        g: self.state(g)
                        for g in (
                            self.grounding_table if groundings is None else groundings
                        )
                    }
                    return result
                groundings = self._ground(groundings)
                bounds = self.get_data(groundings)
            if len(bounds) == 0:
                raise LookupError(f"grounding {groundings} not found in {str(self)}")
            result = self.neuron.state(bounds[None, :])[0]
        result = _utils.node_state(result)
        return utils.fact_to_bool(result) if to_bool else result

    @property
    def unique_var_map(self) -> Dict:
        result = {u: i for i, u in enumerate(self.unique_vars)}
        if _isinstance(self, "_Quantifier"):
            for idx, v in enumerate(self.variables):
                result[v] = idx + len(self.unique_vars)
        return result

    @property
    def world(self) -> World:
        r"""Proxy to get the neuron's world parameter."""
        return self.neuron.world

    @world.setter
    def world(self, new_world: Union[Tuple, World]):
        r"""Proxy to set the neuron's world parameter."""
        self.neuron.world = new_world

    def world_state(self, name=False):
        r"""Returns the state of the `world` variable."""
        try:
            w = World(self.world)
            return w.name if name else w
        except ValueError:
            return self.world

    def And(self, *formulae: "Formula", **kwds) -> "And":
        return subclasses["And"](*self._formula_vars(self, *formulae), **kwds)

    def Or(self, *formulae: "Formula", **kwds) -> "Or":
        return subclasses["Or"](*self._formula_vars(self, *formulae), **kwds)

    def Implies(self, formula: "Formula", **kwds) -> "Implies":
        return subclasses["Implies"](*self._formula_vars(self, formula), **kwds)

    def Not(self, **kwds) -> "Not":
        return subclasses["Not"](*self._formula_vars(self), **kwds)

    def Iff(self, formula: "Formula", **kwds) -> "Iff":
        return subclasses["Iff"](*self._formula_vars(self, formula), **kwds)

    def Exists(self, **kwds) -> "Exists":
        return subclasses["Exists"](self.unique_vars, self, **kwds)

    def Forall(self, **kwds) -> "Forall":
        return subclasses["Forall"](*self.unique_vars, self, **kwds)

    ##
    # Internal function definitions
    ##

    def __call__(
        self,
        *variables: Variable,
    ) -> Tuple[
        "Formula",
        List[Variable],
        Tuple[Variable, ...],
        Tuple[Union[List[tuple[str]], List[None]]],
    ]:
        r"""Variable remapping between operator and operand variables.

        Examples
        --------
        FOL formulae that appear in connectives are callable:
            Note `A`, `B` in the `And`
        ```python
        x = Variable('x')
        model['A'] = Predicate('A')
        model['B'] = Predicate('B')
        model['AB'] = And(A(x), B(x))
        ```

        Returns
        -------
        Formula:
            reference to the child object
        variables:
            tuple of child's Variables
        var_remap:
            tuple of parent-to-child remapping Variables
        bindings:
            tuple of parent-to-child groundings to bind inference to single groundings

        """
        if len(variables) != self.num_unique_vars:
            raise Exception(
                f"please check if the FOL arity of {self} is correctly set "
                "for the number of variables "
                f"variables length ({len(variables)}) must be the same "
                f"length as `num_unique_vars` ({self.num_unique_vars})"
            )

        bindings = list()
        bind = dict()

        variable_objects = list()

        for v in variables:
            if isinstance(v, str):
                bindings.append([self._ground(v, arity_match=False)])
                continue
            elif isinstance(v, Variable):
                variable_objects.append(v)
            else:
                raise TypeError(f"expected variable, received {type(v)}")

            if v in bind:
                if isinstance(bind[v], str):  # a single binding
                    bindings.append([self._ground(bind[v], arity_match=False)])
                elif isinstance(bind[v], list):  # multiple bindings
                    bindings.append(
                        [self._ground(g, arity_match=False) for g in bind[v]]
                    )
                elif isinstance(bind[v], Function):  # A single function
                    bindings.append([bind[v]])
                elif isinstance(bind[v], tuple):  # A bound function
                    bindings.append([bind[v]])
                else:
                    raise TypeError(
                        f"bindings expected as str, "
                        f"Function or list of str or "
                        f"Function, received {type(bind[v])}"
                    )
            else:
                bindings.append([None])
        return self, self.variables, tuple(variable_objects), bindings

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other):
        eq_condition = (self.syntax == other.syntax) and (self.neuron == other.neuron)
        return eq_condition

    def __hash__(self):
        return hash(self.syntax)

    def _add_groundings(self, *groundings: tuple[str]):
        r"""Adds missing groundings to `grounding_table` for those not yet stored.

        Examples
        --------
        ```python
        # for formulae with arity == 1:
        formula._add_groundings(tuple_1, tuple_2)
        ```
        ```python
        # for formulae with arity > 1:
        groundings = ((tuple_1, tuple_7),
                      (tuple_2, tuple_8))
        formula._add_groundings(*groundings)
        ```

        Warning
        -------
        groundings must be given in grounded form: `self._ground(grounding)`

        """
        missing_groundings = {g for g in groundings if g not in self.grounding_table}
        if len(missing_groundings) == 0:
            self._has_new_groundings = False
            return
        self._has_new_groundings = True
        table_rows = self.neuron.extend_groundings(len(missing_groundings))
        self.grounding_table.update(
            {g: table_rows[i] for i, g in enumerate(missing_groundings)}
        )

    def _formula_syntax(self, get_str=lambda x: x.syntax, as_str: bool = True) -> str:
        r"""Determine a name for input formula(e)."""

        def subformula_syntax(
            subformula: Formula,
            operator: str = None,
            subformula_vars: Tuple = None,
            root_var_remap: dict = None,
        ) -> str:
            if subformula_vars is None:
                root_var_remap = dict(
                    (k, k)
                    for k in (
                        self.expanded_unique_vars
                        if _isinstance(self, "_Quantifier")
                        else self.unique_vars
                    )
                )
            else:
                if _isinstance(subformula, "_Quantifier"):
                    for v in subformula.expanded_unique_vars:
                        if v not in root_var_remap:
                            root_var_remap[v] = v
                if subformula_vars != subformula.unique_vars:
                    root_var_remap = dict(
                        (
                            (
                                subformula.unique_vars[i],
                                root_var_remap[subformula_vars[i]],
                            )
                            if subformula.unique_vars[i] != subformula_vars[i]
                            else (subformula_vars[i], subformula_vars[i])
                            for i in range(len(subformula_vars))
                        )
                    )
            result = [
                (
                    f"{f.name}("
                    + _utils.list_to_str(
                        [
                            root_var_remap[v]
                            if as_str
                            else subformula.unique_var_map[root_var_remap[v]]
                            if _isinstance(subformula, "_Quantifier")
                            else self.unique_var_map[root_var_remap[v]]
                            for v in [
                                subformula.expanded_unique_vars[_]
                                if _isinstance(subformula, "_Quantifier")
                                else subformula.unique_vars[_]
                                for _ in subformula.operand_map[i]
                            ]
                        ]
                        if subformula.operand_map[i]
                        else []
                    )
                    + ")"
                )
                if _isinstance(f, "Predicate")
                else f"{f.name}"
                if _isinstance(f, "Proposition")
                else f"{f.syntax}"
                if _isinstance(f, "_Quantifier")
                else (
                    "("
                    + subformula_syntax(
                        f, f.symbol, subformula.var_remap[i], root_var_remap
                    )
                    + ")"
                )
                if _isinstance(f, "_ConnectiveNeuron")
                else (
                    f"{f.symbol}"
                    + subformula_syntax(
                        f, None, subformula.var_remap[i], root_var_remap
                    )
                )
                if _isinstance(f, "Not")
                else ""
                for i, f in enumerate(subformula.operands)
            ]
            return f" {operator} ".join(result) if operator else "".join(result)

        if _isinstance(self, "_Quantifier"):
            quantified_idxs = _utils.list_to_str(
                [self.unique_var_map[v] for v in self.variables], ","
            )
            return f"({self.symbol}{quantified_idxs}, " f"{subformula_syntax(self)})"

        elif _isinstance(self, "Not"):
            return (
                f"{self.symbol}{get_str(self.operands[0])}"
                if self.propositional
                else f"{self.symbol}{subformula_syntax(self)}"
            )
        return (
            ("(" + f" {self.symbol} ".join([get_str(f) for f in self.operands]) + ")")
            if self.propositional
            else f"({subformula_syntax(self, self.symbol)})"
        )

    @staticmethod
    def _formula_vars(*formulae: "Formula") -> List[Union["Formula", Tuple]]:
        r"""Returns a list of formulae as wither called (when predicates)
        or uncalled as connectives.
        """
        variables = list(
            Variable(f"?{i}") for i in range(max([f.arity for f in formulae]))
        )
        return [
            f(*variables[: f.arity])
            if (not f.propositional and _isinstance(f, "Predicate"))
            else f
            for f in formulae
        ]

    def _ground(
        self,
        grounding: Union[str, Tuple[str, ...], tuple[str]],
        arity_match: bool = True,
    ) -> tuple[str]:
        r"""Returns a single grounded object given a grounding in str form.

        If the grounding is already an internal "Grounding" object, it will simply be
        returned to the user as is.

        Examples
        --------
        ```python
        # for arity == 1
        self._ground('str_1')
        ```
        ```python
        # for arity > 1
        grounding = ('str_1', 'str_2', ...)
        self._ground(grounding)
        ```

        Warning
        -------
        This function can only ground one object at a time due to tuple confusion
        for multiple objects

        """

        if isinstance(grounding, tuple):
            return grounding
        if isinstance(grounding, str):
            return (grounding,)

        if Formula._is_grounded(grounding):
            return grounding
        if isinstance(grounding, tuple):
            if not all([type(g) in [str, None] for g in grounding]):
                raise Exception(
                    "expected groundings as tuple of str for num_unique_vars "
                    f" > 1, received {type(grounding)} {grounding}"
                )
            if len(grounding) != self.num_unique_vars and arity_match:
                raise Exception(
                    "expected grounding length to be of "
                    f"arity {self.num_unique_vars}, received {grounding}"
                )
        else:
            if self.num_unique_vars != 1 and arity_match:
                raise Exception(
                    f"{self} received str as grounding, expected grounding "
                    f"({grounding}) as a tuple of len {self.num_unique_vars}"
                )
        return grounding

    @property
    def _groundings(self) -> Set[tuple[str]]:
        """Internal usage to extract groundings"""
        return set(self.grounding_table.keys())

    def _has_bindings(self, slot: int = None) -> bool:
        r"""Returns True if Formula has any bindings."""
        if isinstance(slot, int):
            return (
                True
                if self.bindings[slot]
                and any(
                    [
                        self.bindings[slot][i][j]
                        for i in range(len(self.bindings[slot]))
                        for j in range(len(self.bindings[slot][i]))
                    ]
                )
                else False
            )
        return any([self._has_bindings(i) for i in range(len(self.bindings))])

    def _inherit_from_subformulae(self, *subformula: Union["Formula", Tuple]):
        r"""Builds the local context for a formula using subformulae.

        provides manual variable remapping if given with variables:
            i.e. Formula used as a function via `__call__`
        alternatively inherits remapping from operands if formula is not called

        """

        # inheritance variables
        subformulae = list()
        _operand_vars: List[Union[Tuple[Variable, ...], None]] = [None] * self.arity
        _var_remap: List[Union[Tuple[Variable, ...], None]] = [None] * self.arity
        _bindings: List[Union[Tuple[tuple[str], ...], None]] = [None] * self.arity

        for slot, f in enumerate(subformula):
            # variable remapping from `called` operands:
            #
            #     ```python
            #     P = Predicate("P", arity=2)
            #     Q = Predicate("P", arity=2)
            #     And(P(x, y), Q(y, z))
            #     ```
            if isinstance(f, tuple):
                subformulae.append(f[0])
                _operand_vars[slot] = f[1]
                _var_remap[slot] = f[2]
                _bindings[slot] = f[3]

                self.edge_list.append((self, f[0]))
                self.operands.append(f[0])

            # automatically inherit variable remapping from `uncalled` operands
            #   for higher level connective formulae:
            #
            #     ```python
            #     Implies(And(...), Or(...))
            #     ```
            else:
                if not f.propositional:
                    _var_remap[slot] = f.unique_vars
                    _operand_vars[slot] = f.unique_vars
                    _bindings[slot] = [[None]] * len(f.unique_vars)
                subformulae.append(f)

                self.edge_list.append((self, f))
                self.operands.append(f)
        self.operands: Tuple[Formula] = tuple(self.operands)

        # set class variables as read-only tuples
        self.operand_vars: Tuple[Tuple[Variable, ...], ...] = tuple(_operand_vars)
        self.var_remap: Tuple[Tuple[Variable, ...], ...] = tuple(_var_remap)
        self.bindings: Tuple[Tuple[tuple[str], ...], ...] = tuple(_bindings)

        # inherit propositional flag from children
        if self.propositional is None:
            if _isinstance(self, "_Quantifier"):
                self.set_propositional(
                    not subclasses["_Quantifier"]._has_free_variables(
                        self.variables, self.operands[0]
                    )
                )
            else:
                self.set_propositional(
                    False if any([not f.propositional for f in subformulae]) else True
                )

        # inherit variables from children
        if _isinstance(self, "_Quantifier") or (
            not self.propositional and not _isinstance(self, "_LeafFormula")
        ):
            remap = self.var_remap
            if _isinstance(self, "_Quantifier") and hasattr(self, "variables"):
                remap = (
                    subclasses["_Quantifier"]._unique_variables_overlap(
                        self.variables, self.var_remap[0]
                    ),
                )
            self._update_variables(*remap)
            self._update_map()

        # expand formula graph to include all subformulae
        if subformulae:
            self.edge_list.extend([edge for f in subformulae for edge in f.edge_list])

    @staticmethod
    def _is_grounded(groundings: Union[tuple[str], str, Tuple[str, ...]]) -> bool:
        r"""Returns True if the grounding is given in internal "Grounded" form."""
        return isinstance(groundings, tuple)

    @staticmethod
    def _unique_variables(*variables: Tuple[Variable, ...]) -> Tuple:
        r"""Combines all predicate variables into a unique tuple
        the tuple is sorted by the order of appearance of variables in
        the operands.
        """
        result = list()
        for op_vars in variables:
            if op_vars:
                for v in op_vars:
                    if v not in result:
                        result.append(v)
        return tuple(result)

    def _update_map(self) -> None:
        r"""Update the 'operand_map' to map parent groundings to operand groundings."""
        self.operand_map: List[Tuple] = [None] * len(self.operand_vars)
        for op_idx in range(len(self.var_remap)):
            if self.var_remap[op_idx]:
                op_map = [
                    self.unique_vars.index(op_var)
                    if op_var in self.unique_vars
                    else self.variables.index(op_var) + len(self.unique_vars)
                    for op_var in self.var_remap[op_idx]
                ]
                self.operand_map[op_idx] = tuple(op_map)

    def _update_variables(self, *variables: Tuple[Variable, ...]) -> None:
        r"""
        **Notes**

        If the formula grounding_arity is not specified, it will be inherited
            from the variables. If variables are not specified, both the
            `grounding_arity` and variables are inherited from groundings

        """

        self.unique_vars = Formula._unique_variables(*variables)
        self.num_unique_vars = len(self.unique_vars)

    def _contradiction_loss(self, coeff: float = None) -> torch.Tensor:
        r"""Contradiction loss."""
        if coeff is None:
            coeff = 1
        bounds = self.get_data()
        x = bounds[..., 0] - bounds[..., 1]
        return (self.contradicting_bounds() * coeff * x).sum()

    def _uncertainty_loss(self, coeff: float = None) -> torch.Tensor:
        r"""Uncertainty loss."""
        if coeff is None:
            coeff = 0
        bounds = self.get_data()
        x = bounds[..., 1] - bounds[..., 0]
        return (self.is_contradiction().logical_not() * coeff * x).sum()

    def _supervised_loss(self, coeff: float = None) -> Union[None, torch.Tensor]:
        r"""Supervised loss."""
        if coeff is None:
            coeff = 1
        loss = torch.nn.MSELoss()
        if (
            not hasattr(self, "labels")
            or (self.propositional and not self.labels.numel())
            or (not self.propositional and not self.labels)
        ):
            return
        labels = self.get_labels()
        if self.propositional:
            return coeff * loss(self.get_data(), labels)
        groundings = [g for g in labels if g in self.grounding_table]
        if len(groundings) == 0:
            return
        return coeff * loss(self.get_data(*groundings), self.get_labels(*groundings))

    increment_param_history = _trace.increment_param_history
