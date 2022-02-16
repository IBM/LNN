##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from enum import auto
from importlib import import_module
from typing import Optional, Union, Tuple, Iterator, Set, List, Dict, TypeVar

import torch
import numpy as np

from . import _gm, _trace
from .. import _utils, _exceptions, utils
from ..constants import Fact, World, AutoName, Direction, Join


class Variable:
    r"""Creates free variables to quantify first-order logic formulae

    **Parameters**

        name : str
            name of the free variable
        ctype : str, optional
            constant of the type associated with the free variable

    **Example**

    ```python
    x = Variable('x', 'person')
    x, y, z = map(Variable, ['x', 'y', 'z'])
    ```

    """
    def __init__(self, name: str, ctype: Optional[str] = None):
        self.name = name
        self.ctype = ctype

    def __str__(self) -> str:
        r"""Returns the name of the free variable"""
        return self.name


# declaration to use hinting within the class definition
_Grounding = TypeVar('_Grounding')


class _Grounding(_utils.MultiInstance, _utils.UniqueNameAssumption):
    r"""Propositionalises constants for first-order logic

    Returns a container for a string or a tuple of strings.
    Follows the unique name assumption so that given constant(s) return the
        same object
    Decomposes multiple constants (from the tuple) by storing each str as a
        separate grounding object but returns only the compound container.
        This decomposition is used in grounding management to ensure that all
        partial strings also follow the unique name assumption by returning the
        same container

    **Parameters**

        constants : str or tuple-of-str

    **Example**

    ```python
    _Grounding('person1')
    _Grounding(('person1', 'date1'))
    ```

    **Attributes**
        name : str
            conversion of 'constants' param to str form
        grounding_arity : int
            length of the 'constants' param
        partial_grounding : tuple(_Grounding)
            tuple of groundings for decomposition when constants given as tuple

    """
    def __init__(self, constants: Union[str, Tuple[str, ...]]):
        super().__init__(constants)
        self.name = str(constants)
        if isinstance(constants, tuple):
            self.grounding_arity = len(constants)
            self.partial_grounding = tuple(map(
                self._partial_grounding_from_str, constants))
        else:
            self.grounding_arity = 1
            self.partial_grounding = (self, )

    @classmethod
    def _partial_grounding_from_str(cls, constant: str) -> _Grounding:
        r"""Returns partial Grounding given grounding str"""
        return _Grounding.instances[constant]

    @classmethod
    def ground_by_groundings(cls, *grounding: _Grounding):
        r"""Reduce a tuple of groundings to a single grounding"""
        return (grounding[0] if len(grounding) == 1
                else cls.__class__(tuple(str(g) for g in grounding)))

    def __len__(self) -> int:
        r"""Returns the length of the grounding arity"""
        return self.grounding_arity

    def __str__(self) -> str:
        r"""Returns the name of the grounding"""
        return self.name

    @staticmethod
    def eval(grounding: _Grounding) -> Union[str, Tuple[str, ...]]:
        r"""Returns the original constant(s) in str or tuple-of-str form"""
        return eval(grounding.name)


# declaration to use hinting within the class definition
_Formula = TypeVar('_Formula')


class _Formula:
    r"""_Formula(*formula: _Formula,
                name: Optional[str] = '',
                world: World = World.OPEN, **kwds)

    **Warnings**
    Formula should not be directly instantiated

    """
    unique_num: Dict[str, int] = dict()

    @property
    def class_name(self) -> str:
        _Formula.unique_num.setdefault(self.__class__.__name__, 0)
        return self.__class__.__name__

    def __init__(self,
                 *formula: _Formula,
                 name: Optional[str] = '',
                 arity: int = None,
                 world: World = World.OPEN,
                 **kwds):

        # formula naming
        self.name = name if name else (
            f'{self.class_name}_{_Formula.unique_num[self.class_name]}')
        _Formula.unique_num[self.class_name] += 1
        self.join_method = kwds.get('join', Join.INNER)

        # construct edge and operand list for each formula
        self.edge_list = list()
        self.operands = list()

        # formula arity
        self.arity = arity

        # inherit propositional, variables, and graphs
        self.propositional = kwds.get('propositional')
        self._inherit_from_subformulae(*formula)

        # formula truth world assumption
        _exceptions.AssertWorld(world)
        self.world = world

        # formula grounding table maps grounded objects to table rows
        self.grounding_table = None if self.propositional else dict()

        # placeholder for neural variables and functions
        self.variables = list([None] * arity)
        self.neuron = None
        self.func = None
        self.func_inv = None

    def set_propositional(self, propositional: bool) -> None:
        self.propositional = propositional

    def _inherit_from_subformulae(self,
                                  *subformula: Union[_Formula, Tuple]
                                  ) -> None:
        r"""
        _inherit_from_subformula(*Union[_Formulae, _Formula(*Variable)])

        provides manual manual variable remapping if given with variables:
            i.e. _Formula used as a function via `__call__`
        alternatively inherits remapping from operands if formula is not called

        """

        # inheritance variables
        self.operand_vars = [None] * self.arity
        self.var_remap = [None] * self.arity
        self.bindings = [None] * self.arity
        self.binding_str = [None] * self.arity
        self.has_remapping_vars = True

        subformulae = list()
        for slot, f in enumerate(subformula):
            # manual variable remapping from`__called__` operands
            # And(Predicate(x, y)) ... Or(Predicate(x, y)) ...
            if isinstance(f, tuple):
                self.operand_vars[slot] = f[1]
                self.var_remap[slot] = f[2]
                self.bindings[slot] = f[3]
                self.binding_str[slot] = f[4]
                subformulae.append(f[0])

                self.edge_list.append((self, f[0]))
                self.operands.append(f[0])

            # inherit variable remapping from 'uncalled' operands
            # for higher level connective formulae
            else:
                if not f.propositional:
                    self.var_remap[slot] = f.unique_vars
                    self.operand_vars[slot] = f.unique_vars
                    self.bindings[slot] = tuple([[None]])
                subformulae.append(f)

                self.edge_list.append((self, f))
                self.operands.append(f)

        # inherit propositional flag from children
        if self.propositional is None:
            self.set_propositional(False if any(
                [not f.propositional for f in subformulae]) else True)

        # inherit variables from children
        if not self.propositional and not isinstance(self, _LeafFormula):
            self._update_variables(*self.var_remap)
            self._update_map()

        # expand formula graph to include all subformulae
        if subformulae:
            self.edge_list.extend([edge for f in subformulae
                                   for edge in f.edge_list])

    def _update_variables(self, *variables: Tuple[Variable, ...]) -> None:
        r"""
        **Notes**

        If the formula grounding_arity is not specified, it will be inherited
            from the variables. If variables are not specified, both the
            `grounding_arity` and variables are inherited from groundings

        """
        self.variables = self.unique_vars = _gm.unique_variables(*variables)
        self.num_unique_vars = len(self.unique_vars)

    def _update_map(self) -> None:
        r"""
        Update the 'operand_map' to map parent groundings to operand groundings
        """
        self.operand_map = [None] * len(self.operand_vars)
        for op_index, op_vars in enumerate(self.var_remap):
            op_map = [var_index for var_op in self.var_remap[op_index]
                      for var_index, var in enumerate(self.unique_vars)
                      if var == var_op]
            self.operand_map[op_index] = tuple(op_map)

    def _set_world(self, world: World) -> None:
        _exceptions.AssertWorld(world)
        self.world = world
        self.neuron.set_world(world)

    @staticmethod
    def _is_grounded(groundings: Union[_Grounding,
                                       str,
                                       Tuple[str, ...]]
                     ) -> bool:
        return isinstance(groundings, _Grounding)

    def _ground(self,
                grounding: Union[_Grounding, str, Tuple[str, ...]],
                arity_match: bool = True
                ) -> _Grounding:
        r"""returns a single grounded object

        **Example**

        ```python
        # for arity == 1
        self._ground('str_1')
        ```
        ```python
        # for arity > 1
        grounding = ('str_1', 'str_2', ...)
        self._ground(grounding)
        ```

        **Warning**

        only grounds one object at a time due to tuple confusion
        for multiple objects

        """
        if _Formula._is_grounded(grounding):
            return grounding
        if isinstance(grounding, tuple):
            if not all([type(g) in [str, None] for g in grounding]):
                raise Exception(
                    'expected groundings as tuple of str for num_unique_vars '
                    f' > 1, received {type(grounding)} {grounding}')
            if len(grounding) != self.num_unique_vars and arity_match:
                raise Exception(
                    'expected grounding length to be of '
                    f'arity {self.num_unique_vars}, received {grounding}')
        else:
            if self.num_unique_vars != 1 and arity_match:
                raise Exception(
                    f'{self} received str as grounding, expected grounding '
                    f'({grounding}) as a tuple of len {self.num_unique_vars}')
        return _Grounding(grounding)

    def _add_groundings(self, *groundings: _Grounding) -> None:
        r"""add missing groundings to `grounding_table`

        **Returns**

        tuple:  groundings in grounded form

        **Example**

        ```python
        # for formulae with arity == 1:
        model['formula']._add_groundings(_Grounding_1, _Grounding_2)
        ```
        ```python
        # for formulae with arity > 1:
        groundings = ((_Grounding_1, _Grounding_7),
                      (_Grounding_2, _Grounding_8))
        model['formula']._add_groundings(*groundings)
        ```

        **Warning**

        groundings must be given in grounded form: `self._ground('grounding')`
        _add_groundings should not be directly called
        - instead use `model.add_facts('node': Fact)`

        """
        missing_groundings = {g for g in groundings
                              if g not in self.grounding_table}
        if len(missing_groundings) == 0:
            return
        table_rows = self.neuron.extend_groundings(len(missing_groundings))
        self.grounding_table.update(
            {g: table_rows[i] for i, g in enumerate(missing_groundings)})

    def _add_facts(self, facts: Union[Fact, Tuple, Set, Dict]) -> None:
        r"""Populate formula with fact

        Facts given in bool, tuple or None, assumes a propositional formula.
        Facts given in dict form assume FOL, keyed by the grounding, where the
            value also required in bool, tuple or None

        **Warning**

        Formulae facts should not be modified directly, rather provide facts
        on a per model basis
        All subsequent models that are instantiated with formulae that have
            existing facts will clone facts into the models

        **Example**

        usage from within a model
        ```python
        # Propositional:
        model['proposition']._add_facts(TRUE)
        ```
        ```python
        # First-order logic
        model['predicate']._add_facts({'grounding': TRUE})
        ```

        """
        if self.propositional:  # Propositional facts
            _exceptions.AssertBounds(facts)
            self.neuron.add_facts(facts, update_leaves=True)

        else:  # FOL facts
            if isinstance(facts, dict):  # facts given per grounding
                # replace fact keys (str -> _Groundings)
                groundings = tuple(facts.keys())
                for g in groundings:
                    facts[self._ground(g)] = facts.pop(g)

                # add missing groundings to `grounding_table`
                groundings = tuple(facts.keys())
                self._add_groundings(*groundings)

                # set facts for all groundings
                table_facts = {
                    self.grounding_table[g]: facts[g] for g in groundings}
            elif isinstance(facts, set):  # broadcast facts across groundings
                table_facts = facts
            else:
                raise Exception('FOL facts should be from [dict, set], '
                                f'"{self}" received {type(facts)}')
            self.neuron.add_facts(table_facts, update_leaves=True)

    def get_facts(self,
                  *groundings: Union[str, Tuple[str, ...], _Grounding]
                  ) -> torch.Tensor:
        r"""returns bounds_table slices

        if groundings is None, the whole table will return
        elif a tuple of groundings returns appropriate slices
        """
        if self.propositional or len(groundings) == 0:
            return self.neuron.get_facts()
        table_rows = list(self.grounding_table.get(self._ground(g))
                          for g in groundings)
        result = self.neuron.get_facts(table_rows)
        return result

    def get_labels(self, *groundings: str) -> torch.Tensor:
        r"""returns labels if no groundings else tuple of given bounds"""
        if self.propositional or len(groundings) == 0:
            return self.labels
        result = torch.stack(
            [self.labels.get(self._ground(g)) for g in groundings])
        return result[0] if len(groundings) == 1 else result

    def _add_labels(self,
                    labels: Union[Fact, Tuple[str, ...], Set, Dict]
                    ) -> None:
        r"""Populate labels with fact

        Facts given in bool, tuple or None, assumes a propositional formula.
        Facts given in dict form assume FOL, keyed by the grounding, where the
            value also required in bool, tuple or None

        **Example**

        ```python
        # Propositional
        model.add_labels({'proposition': TRUE})
        ```
        ```python
        # First-order logic
        model.add_labels({'predicate': {'grounding': TRUE}})
        ```

        """
        if self.propositional:  # Propositional labels
            _exceptions.AssertBounds(labels)
            self.labels = _utils.fact_to_bounds(labels, True)

        else:  # FOL labels
            if not hasattr(self, 'labels'):
                self.labels = dict()
            if isinstance(labels, dict):  # labels given per grounding
                for g in tuple(labels.keys()):
                    # set labels for groundings, replace str keys -> Groundings
                    self.labels[self._ground(g)] = _utils.fact_to_bounds(
                        labels.pop(g), True)
            else:
                raise Exception('FOL facts should be from [dict, set], '
                                f'"{self}" received {type(labels)}')

    @property
    def groundings(self) -> Set:
        return {g for g in self.grounding_table}

    def state(self,
              groundings: Union[str,
                                Tuple[str, ...],
                                List[str],
                                List[Tuple[str, ...]]] = None,
              to_bool: bool = False,
              bounds: torch.Tensor = None
              ) -> Union[torch.Tensor,
                         Dict[Union[str,
                                    Tuple[str, ...]],
                              torch.Tensor]]:
        r"""returns the state of a single grounded fact

        if to_bool flag is True, will map classical Facts to bool:
            i.e. {Fact.True: True, FALSE': False}
        The rest of the node states will return as str

        **Notes**

        see section (F.2)[https://arxiv.org/abs/2006.13155] for more
            information on node states

        """
        if self.propositional:
            result = self.neuron.state()
        else:
            if bounds is None:
                if groundings is None or isinstance(groundings, list):
                    result = {str(g): self.state(g)
                              for g in (self.grounding_table
                              if groundings is None else groundings)}
                    return result
                groundings = self._ground(groundings)
                bounds = self.get_facts(groundings)
            if len(bounds) == 0:
                raise LookupError(
                    f'grounding {groundings} not found in {str(self)}')
            result = self.neuron.state(bounds[None, :])[0]
        result = _utils.node_state(result)
        return utils.fact_to_bool(result) if to_bool else result

    def print(self,
              header_len: int = 50,
              roundoff: int = 5,
              params: bool = False,
              grads: bool = False
              ) -> None:
        r"""
        print the state of the formula

        ```
        OPEN   Proposition: A                 UNKNOWN (L, U)

        CLOSED Proposition: B                   FALSE (L, U)

        OPEN   Predicate: P1 (x)
        'const_1'                                TRUE (L, U)
        'const_2'                       CONTRADICTION (L, U)
        'const_3'                               FALSE (L, U)

        OPEN   Predicate: P2 (x, y)
        ('const_1', 'const_5')                   TRUE (L, U)
        ('const_2', 'const_6')          CONTRADICTION (L, U)
        ('const_3', 'const_7')                  FALSE (L, U)

        OPEN   And: And_0 (x, y)
        Bindings: P1 (x: 'const_1'), P2 (x: 'const_2', y: ['const_3', ...])
        ('const_1', 'const_3')                   TRUE (L, U)
        ('const_2', 'const_3')          CONTRADICTION (L, U)
        ('const_1', 'const_7')                  FALSE (L, U)

        TRUE   ForAll: ForAll_0 (y)           UNKNOWN (L, U)
        ```

        """
        def state_wrapper(grounding: _Grounding):
            return (f"'{grounding}'" if grounding.grounding_arity == 1
                    else f"{grounding}")

        def round_bounds(grounding=None):
            return tuple([round(r, roundoff)
                          for r in (self.get_facts(grounding)
                                    if self.propositional else
                                    self.get_facts(grounding)[0]).tolist()])

        header = (f'{self.world.name:<6} '
                  f'{self.__class__.__name__}: '
                  f'{str(self)}')
        if params:
            params = dict()
            for name, symbol in _utils.param_symbols.items():
                if hasattr(self.neuron, name):
                    val = getattr(self.neuron, name)
                    params[symbol] = (
                        f'{np.around(val.tolist(), roundoff)}'
                        + (f' grads {np.around(val.grad.tolist(), roundoff)}'
                           if grads and val.grad is not None else ''))
            params = ('params  ' + ',  '.join(
                [f'{k}: {v}' for k, v in params.items()]))
        else:
            params = ''

        # extract variables
        var_str = '' if not (
                hasattr(self, 'unique_vars') and self.unique_vars) else (
            str(tuple([str(v) for v in self.unique_vars])) if (
                    len(self.unique_vars) > 1) else (
                str(f'({self.unique_vars[0]})')))

        # print propositional node - single bounds
        if self.propositional:
            states = (
                f'{" ".join([header, var_str]):<{header_len}} '
                f'{self.state().name:>13} '
                f'{round_bounds()}\n'
                f'{params}')
            print(states)

        # print FOL node - table of bounds
        else:
            facts = [''] if self.grounding_table is None else (
                    [(f'{state_wrapper(g):{header_len}} '
                      f'{self.state(g).name:>13} '
                      f'{round_bounds(g)}\n')
                     for g in self.grounding_table])
            var_str = '' if not (
                hasattr(self, 'unique_vars') and self.unique_vars) else (
                    str(tuple([str(v) for v in self.unique_vars])) if (
                        len(self.unique_vars) > 1) else (
                            str(f'({self.unique_vars[0]})')))

            binder = list()
            for i, op in enumerate(self.operands):
                if (var_str
                        and not self.propositional
                        and self._has_bindings(i)):
                    binder.append(f'{op} ({self.binding_str[i]})')
            bind_str = ('\nBindings: ' + ', '.join(binder)) if binder else ''
            header = f'{header}{var_str} {bind_str}'
            params = (f'\n{params}' if params else '')
            header = f'{header}{params}'
            print(f'{header}\n' + ''.join(facts))

    def flush(self) -> None:
        r"""set all facts in formula to 'Unknown'"""
        self.neuron.flush()

    def is_contradiction(self,
                         bounds: torch.Tensor = None
                         ) -> torch.BoolTensor:
        r"""check if bounds are in contradiction"""
        return self.neuron.is_contradiction(bounds)

    def _has_bindings(self, slot: int = None) -> bool:
        r"""Returns True if Formula has any bindings"""
        if isinstance(slot, int):
            return True if any([self.bindings[slot][i][j]
                                for i in range(len(self.bindings[slot]))
                                for j in range(len(self.bindings[slot][i]))]
                               ) else False
        return any([self._has_bindings(i)
                    for i in range(len(self.bindings))])

    def __call__(self,
                 *variables: Union[Variable,
                                   Tuple[Variable, Union[str, List[str]]]]
                 ) -> Tuple[_Formula,
                            List[Union[Variable, None]],
                            Tuple[List[Union[Variable, None]], ...],
                            Tuple[Union[_Grounding, None], ...],
                            str]:
        r"""variable remapping between operator and operand variables

        **Example**

        FOL formulae that appear in connectives are callable:
            Note `A`, `B` in the `And`
        ```python
        x = Variable('x')
        model['A'] = Predicate()
        model['B'] = Predicate()
        model['AB'] = And(A(x), B(x))
        ```

        **Returns**

        _Formula: reference to the child object
        variables: tuple of child's Variables
        var_remap: tuple of parent-to-child remapping Variables
        bindings: tuple of parent-to-child groundings to bind inference to
            single groundings are _Grounding
            multiple groundings are list(_Grounding)
        binding_string: complete bindings given in string form

        """
        if len(variables) != self.num_unique_vars:
            raise Exception(
                f'{self} variables length ({len(variables)}) must be the same '
                f'length as `num_unique_vars` ({self.num_unique_vars})')
        bindings = list()
        _variables = list()  # variables to return
        binding_str = list()
        for v in variables:
            if isinstance(v, tuple):
                if not isinstance(v[0], Variable):
                    raise Exception(
                        'expected a Variable for first tuple input, received '
                        f'{[type(v[0]), v[0]]}')
                _variables.append(v[0])
                binding_str.append(f'{v[0]}: {v[1]}')
                if isinstance(v[1], list):  # Pred(x, ['str_1', ...]))
                    bindings.append([self._ground(g, arity_match=False)
                                     for g in v[1]])
                elif isinstance(v[1], str):  # Pred(x, 'str_1')
                    bindings.append([self._ground(v[1], arity_match=False)])
                else:
                    raise Exception('bindings expected from [str, List(str)],'
                                    f' received {type(v[1]), v[1]}')
            else:
                bindings.append([None])
                _variables.append(v)
                binding_str.append(f'{v}')
            binding_str = [', '.join(binding_str)]
        binding_str = ', '.join(binding_str)
        return (self,
                self.variables,
                tuple(_variables),
                tuple(bindings),
                binding_str)

    def __str__(self) -> str:
        return self.name

    def rename(self, name: str) -> None:
        self.name = name

    def parameters(self) -> Iterator[torch.Tensor]:
        for name, param in self.neuron.named_parameters():
            yield param

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from self.neuron.named_parameters()

    def params(self,
               *params: str,
               detach: bool = False
               ) -> Union[torch.Tensor, List[torch.Tensor]]:
        result = list()
        for param in params:
            if param in dict(self.neuron.named_parameters()):
                result.append(
                    getattr(self.neuron, param).clone().detach()
                    if detach else getattr(self.neuron, param))
            else:
                raise ValueError(f'{self} has no attribute: {param}')
        return result[0] if len(params) == 1 else result

    def contradiction_loss(self,
                           coeff: float = None
                           ) -> torch.Tensor:
        r"""Contradiction loss"""
        if coeff is None:
            coeff = 1
        bounds = self.get_facts()
        x = bounds[..., 0] - bounds[..., 1]
        return (self.is_contradiction() * coeff * x).sum()

    def uncertainty_loss(self,
                         coeff: float = None
                         ) -> torch.Tensor:
        r"""Uncertainty loss"""
        if coeff is None:
            coeff = 0
        bounds = self.get_facts()
        x = bounds[..., 1] - bounds[..., 0]
        return (self.is_contradiction().logical_not() * coeff * x).sum()

    def supervised_loss(self,
                        coeff: float = None
                        ) -> Union[None, torch.Tensor]:
        r"""supervised loss"""
        if coeff is None:
            coeff = 1
        loss = torch.nn.MSELoss()
        if (not hasattr(self, 'labels')
                or (self.propositional and not self.labels.numel())
                or (not self.propositional and not self.labels)):
            return
        labels = self.get_labels()
        if self.propositional:
            return coeff * loss(self.get_facts(), labels)
        groundings = [g for g in labels if g in self.grounding_table]
        if len(groundings) == 0:
            return
        return coeff * loss(
            self.get_facts(*groundings), self.get_labels(*groundings))

    def reset_bounds(self) -> None:
        self.neuron.reset_bounds()

    def project_params(self) -> None:
        self.neuron.project_params()

    increment_param_history = _trace.increment_param_history

    def is_unweighted(self) -> bool:
        bias, weights = self.params('bias', 'weights')
        return all([bias == w for w in weights])

    @property
    def shape(self) -> torch.Size:
        return self.neuron.bounds_table.shape

    @staticmethod
    def _formula_name(*formulae: _Formula, connective: str) -> str:
        r"""Come up with a name for input formula(e)"""
        return '(' + f' {connective} '.join([f.name for f in formulae]) + ')'

    @staticmethod
    def _formula_vars(*formulae: _Formula) -> List[Union[_Formula, Tuple]]:
        r"""
        Returns a list of formulae as wither called (when predicates)
        or uncalled as connectives
        """
        variables = list(Variable(f"x{i}") for i in range(
            max([f.arity for f in formulae])))
        return [f(*variables[:f.arity]) if not f.propositional else f
                for f in formulae]

    # declaration to use hinting within the function definition
    Not = TypeVar('Not')

    def Not(self, **kwds) -> Not:
        if 'name' not in kwds:
            kwds['name'] = f'¬({self.name})'
        return Not(*self._formula_vars(self), **kwds)

    # declaration to use hinting within the function definition
    And = TypeVar('And')

    def And(self, *formulae: _Formula, **kwds) -> And:
        if 'name' not in kwds:
            kwds['name'] = self._formula_name(self, *formulae, connective='∧')
        return And(*self._formula_vars(self), **kwds)

    # declaration to use hinting within the function definition
    Or = TypeVar('Or')

    def Or(self, *formulae: _Formula, **kwds) -> Or:
        if 'name' not in kwds:
            kwds['name'] = self._formula_name(self, *formulae, connective='∨')
        return Or(*self._formula_vars(self, *formulae), **kwds)

    # declaration to use hinting within the function definition
    Implies = TypeVar('Implies')

    def Implies(self, formula: _Formula, **kwds) -> Implies:
        if 'name' not in kwds:
            kwds['name'] = self._formula_name(self, formula, connective='→')
        return Implies(*self._formula_vars(self, formula), **kwds)

    # declaration to use hinting within the function definition
    Bidirectional = TypeVar('Bidirectional')

    def Bidirectional(self, formula: _Formula, **kwds) -> Bidirectional:
        if 'name' not in kwds:
            kwds['name'] = self._formula_name(self, formula, connective='↔')
        return Bidirectional(*self._formula_vars(self, formula), **kwds)


class _LeafFormula(_Formula):
    r"""Specifies activation functionality as nodes instead of neurons

    Assumes that all leaf formulae are propositions or predicates, therefore
        uses the _NodeActivation accordingly

    """
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.neuron = _NodeActivation()(
            self.propositional, self.world, **kwds.get('neuron', {}))


class Proposition(_LeafFormula):
    r"""Creates propositional containers

    Stores and retrieves single truth bounds instead of tables as in FOL case

    **Parameters**

        name : str
            name of the proposition

    **Example**

    ```python
    P = Proposition('Person')
    ```

    """
    def __init__(self,
                 name: Optional[str] = '',
                 **kwds):
        super().__init__(name=name, arity=1, propositional=True, **kwds)

    def _add_facts(self, fact: Fact) -> None:
        """Populate proposition with facts

        Facts required in bool, tuple or None
        None fact assumes `Unknown`
        tuple fact required in bounds form `(Lower, Upper)`

        """
        super()._add_facts(fact)


class Predicate(_LeafFormula):
    r"""Creates a container for a predicate

    Stores a table of truths, with columns specified by the arity and rows
        indexed by the grounding

    **Parameters**

        name : str
            name of the predicate
        arity : int, optional
            defaults to unary predicates ()

    **Example**

    ```python
    P1 = Predicate()
    P2 = Predicate(arity=2)
    ```

    """
    def __init__(self,
                 name: Optional[str] = '',
                 arity: int = 1,
                 **kwds):
        if arity is None:
            raise Exception(
                f"arity expected as int > 0, received {arity}")
        super().__init__(name=name, arity=arity, propositional=False, **kwds)
        self._update_variables(
            tuple(Variable(f"x{i}") for i in range(self.arity)))

    def _add_facts(self, facts: Union[dict, set]) -> None:
        r"""Populate predicate with facts

        Facts required in dict or set
            - dict for grounding-based facts
            - set for broadcasting facts across all groundings
              requires a set of 1 item
        dict keys for groundings and values as facts
        tuple facts required in bounds form `(Lower, Upper)`

        """
        super()._add_facts(facts)


class _ConnectiveFormula(_Formula):
    def __init__(self,
                 *formula: _Formula,
                 **kwds):
        super().__init__(*formula, **kwds)


class _ConnectiveNeuron(_ConnectiveFormula):
    def __init__(self, *formula, **kwds):
        super().__init__(*formula, **kwds)
        self.neuron = _NeuralActivation(
            kwds.get('neuron', {}).get('type'))(
                self.propositional, self.arity, self.world,
                **kwds.get('neuron', {}))
        self.func = self.neuron.function(
            self.__class__.__name__, direction=Direction.UPWARD)
        self.func_inv = self.neuron.function(
            self.__class__.__name__, direction=Direction.DOWNWARD)

    def upward(self,
               groundings: Set[Union[str, Tuple[str, ...]]] = None,
               **kwds
               ) -> Union[torch.Tensor, None]:
        upward_bounds = _gm.upward_bounds(self, self.operands, groundings)
        if upward_bounds is None:  # contradiction arresting
            return
        input_bounds, groundings = upward_bounds
        grounding_rows = None if self.propositional else (
            self.grounding_table.values() if groundings is None
            else [self.grounding_table.get(g) for g in groundings])
        result = self.neuron.aggregate_bounds(
            grounding_rows, self.func(input_bounds))
        return result

    def downward(self,
                 index: int = None,
                 groundings: Set[Union[str, Tuple[str, ...]]] = None,
                 **kwds
                 ) -> Union[torch.Tensor, None]:
        operands = tuple(self.operands)
        downward_bounds = _gm.downward_bounds(
            self, operands, groundings)
        if downward_bounds is None:  # contradiction arresting
            return
        out_bounds, input_bounds, groundings = downward_bounds
        new_bounds = self.func_inv(out_bounds, input_bounds)
        op_indices = enumerate(operands) if index is None else (
            [(index, operands[index])])
        result = 0
        for op_index, op in op_indices:
            if op.propositional:
                op_grounding_rows = None
            else:
                if groundings is None:
                    op_grounding_rows = op.grounding_table.values()
                else:
                    op_grounding_rows = [None] * len(groundings)
                    for g_i, g in enumerate(groundings):
                        op_g = [str(g.partial_grounding[slot])
                                for slot in self.operand_map[op_index]]
                        op_g = _Grounding(tuple(op_g) if len(op_g) > 1
                                          else op_g[0])
                        op_grounding_rows[g_i] = op.grounding_table.get(op_g)
            result = result + op.neuron.aggregate_bounds(
                op_grounding_rows, new_bounds[..., op_index])
        return result

    def logical_loss(self,
                     coeff: float = None,
                     slacks: Union[bool, float] = None
                     ) -> torch.Tensor:
        r"""Logical loss to create a loss on logical constraint violation

        Assumes a soft logic computation and calculates the loss on constraints
        as defined in [equations 86-89](https://arxiv.org/pdf/2006.13155.pdf)
        when slacks are given, the constraints are allowed to be violated
        however this affects the neuron interpretability and should only be
        used if the model is not strictly required to obey a classical
        definition of logic

        """
        a, b, w = self.params('alpha', 'bias', 'weights')
        T, F = a, 1-a
        coeff = 1 if coeff is None else coeff
        if isinstance(self, And):
            TRUE = b - (w * (1 - T)).sum()
            FALSE = b - (w * (1 - F))
            true_hinge = torch.where(
                TRUE < T, T - TRUE, TRUE * 0)
            false_hinge = torch.where(
                FALSE > F, FALSE - F, FALSE * 0)
            if slacks:
                if slacks is True:
                    slacks_false = false_hinge * (false_hinge > 0)
                    slacks_true = true_hinge * (true_hinge > 0)
                    false_hinge -= slacks_false
                    true_hinge -= slacks_true
                    self.neuron.slacks = (
                        slacks_true.detach().clone(),
                        slacks_false.detach().clone())
                else:
                    false_hinge -= slacks
            self.neuron.feasibility = (
                true_hinge.detach().clone(),
                false_hinge.detach().clone())

        elif isinstance(self, Or):
            TRUE = 1 - b + (w * T)
            FALSE = 1 - b + (w * F).sum()
            true_hinge = torch.where(
                TRUE < T, T - TRUE, TRUE * 0).sum()
            false_hinge = torch.where(
                FALSE > F, FALSE - F, FALSE * 0)
        elif isinstance(self, Implies):
            TRUE = 1 - b + (w * T)  # T = 1-F for x and T for y
            FALSE = 1 - b + (w[0] * (1 - T)) + (w[1] * F)
            true_hinge = torch.where(
                TRUE < T, T - TRUE, TRUE * 0).sum()
            false_hinge = torch.where(
                FALSE > F, FALSE - F, FALSE * 0)
        result = (true_hinge.square() + false_hinge.square()).sum()
        return coeff * result


class _BinaryNeuron(_ConnectiveNeuron):
    r"""Restrict neurons to 2 inputs"""
    def __init__(self, *formula, **kwds):
        if len(formula) != 2:
            raise Exception(
                'Binary neurons expect 2 formulae as inputs, received '
                f'{len(formula)}')
        super().__init__(*formula, arity=2, **kwds)


class _NAryNeuron(_ConnectiveNeuron):
    r"""N-ary connective neuron"""
    def __init__(self, *formula, **kwds):
        super().__init__(*formula, arity=len(formula), **kwds)


class And(_NAryNeuron):
    r"""Symbolic Conjunction

    Returns a logical conjunction where inputs can be propositional,
    first-order logic predicates or other connectives.

    **Parameters**

        formula : Union[_ConnectiveFormula, Proposition, Predicate(*Variable)]
            accepts n-ary inputs
        kwds : dict
            _Formula kwds

    **Example**

    ```python
    # Propositional
    A, B, C = map(Proposition, ['A', 'B', 'C'])
    And(A, B, C)
    ```
    ```python
    # First-order logic
    x, y = map(Variables, ['x', 'y'])
    A, C = map(Predicate, ['A', 'C'])
    B = Predicate('B', arity=2)
    And(A(x), B(x, y), C(y)))
    ```

    """
    pass


class Or(_NAryNeuron):
    r"""Symbolic Disjunction

    Returns a logical disjunction where inputs can be propositional,
    first-order logic predicates or other connectives.

    **Parameters**

        formula : Union[_ConnectiveFormula, Proposition, Predicate(*Variable)]
            accepts n-ary inputs
        kwds : dict
            _Formula kwds

    **Example**

    ```python
    # Propositional
    A, B, C = map(Proposition, ['A', 'B', 'C'])
    Or(A, B, C)
    ```
    ```python
    # First-order logic
    x, y = map(Variables, ['x', 'y'])
    A, C = map(Predicate, ['A', 'C'])
    B = Predicate('B', arity=2)
    Or(A(x), B(x, y), C(y)))
    ```

    """
    pass


class Implies(_BinaryNeuron):
    r"""Symbolic Implication

    Returns a logical implication where inputs can be propositional,
    first-order logic predicates or other connectives.

    **Parameters**

        formula : Union[_ConnectiveFormula, Proposition, Predicate(*Variable)]
            accepts binary inputs
        kwds : dict
            _Formula kwds

    **Example**

    ```python
    # Propositional
    A, B = map(Proposition, ['A', 'B'])
    Implies(A, B)
    ```
    ```python
    # First-order logic
    x, y = map(Variables, ['x', 'y'])
    A = Predicate('A')
    B = Predicate('B', arity=2)
    Implies(A(x), B(x, y)))
    ```

    """
    pass


class Bidirectional(_BinaryNeuron):
    r"""Symbolic Bidirectional Implication

    Returns a logical bidirectional implication where inputs can be
    propositional, first-order logic predicates or other connectives.
    Decomposes into a conjunction of implications, where the root conjunction
    is defined as the bidirectional neuron and named accordingly.

    **Parameters**

        formula : Union[_ConnectiveFormula, Proposition, Predicate(*Variable)]
            accepts binary inputs
        kwds : dict
            _Formula kwds

    **Example**

    ```python
    # Propositional
    A, B = map(Proposition, ['A', 'B'])
    Bidirectional(A, B)
    ```
    ```python
    # First-order logic
    x, y = map(Variables, ['x', 'y'])
    A = Predicate('A')
    B = Predicate('B', arity=2)
    Bidirectional(A(x), B(x, y)))
    ```

    """
    def __init__(self, *formula, **kwds):
        lhs, rhs = formula
        self.Imp1, self.Imp2 = Implies(lhs, rhs), Implies(rhs, lhs)
        super().__init__(self.Imp1, self.Imp2, **kwds)
        self.func = self.neuron.function(
            'And', direction=Direction.UPWARD)
        self.func_inv = self.neuron.function(
            'And', direction=Direction.DOWNWARD)

    def upward(self, *args, **kwds) -> torch.Tensor:
        self.Imp1.upward(*args, **kwds)
        self.Imp2.upward(*args, **kwds)
        return super().upward(*args, **kwds)

    def downward(self, *args, **kwds) -> torch.Tensor:
        self.Imp1.downward(*args, **kwds)
        self.Imp2.downward(*args, **kwds)
        return super().downward(*args, **kwds)


class _UnaryOperator(_ConnectiveFormula):
    r"""Restrict operators to 1 input"""
    def __init__(self,
                 *formula: _Formula,
                 **kwds):
        if len(formula) != 1:
            raise Exception(
                'Unary operator expect 1 formula as input, received '
                f'{len(formula)}')
        super().__init__(*formula, **kwds)


class Not(_UnaryOperator):
    r"""Symbolic Negation

    Returns a logical negation where inputs can be propositional,
    first-order logic predicates or other connectives.

    **Parameters**

        *formula : Union[_ConnectiveFormula, Proposition, Predicate(*Variable)]
            accepts a unary input
        kwds : dict
            _Formula kwds

    **Example**

    ```python
    # Propositional
    A, B = map(Proposition, ['A', 'B'])
    Bidirectional(A, B)
    ```
    ```python
    # First-order logic
    x, y = map(Variables, ['x', 'y'])
    A = Predicate('A')
    B = Predicate('B', arity=2)
    Bidirectional(A(x), B(x, y)))
    ```

    """
    def __init__(self, operand, **kwds):
        self.operand = operand[0] if isinstance(operand, Tuple) else operand
        kwds.setdefault('name', 'Not_' + self.operand.name)
        super().__init__(operand, arity=1, **kwds)
        self.neuron = _NodeActivation()(
            self.propositional, self.world, **kwds.get('neuron', {}))

    def upward(self, **kwds) -> torch.Tensor:
        if self.propositional:
            groundings = {None}
        else:
            groundings = tuple(self.operand.groundings)
            for g in groundings:
                if g not in self.grounding_table:
                    self._add_groundings(g)
        return self.neuron.aggregate_bounds(
            None, _utils.negate_bounds(self.operand.get_facts(*groundings)))

    def downward(self) -> torch.Tensor:
        if self.propositional:
            groundings = {None}
        else:
            groundings = tuple(self.groundings)
            for g in groundings:
                if g not in self.operand.groundings:
                    self.operand._add_groundings(g)
        return self.operand.neuron.aggregate_bounds(
            None, _utils.negate_bounds(self.get_facts(*groundings)))


class _Quantifier(_Formula):
    r"""Symbolic container for quantifiers

    **Parameters**

        kwds : dict
            fully_grounded : bool
                specifies if a full upward inference can be done on a
                quantifier due to all the groundings being present inside it.
                This applies to the lower bound of a `ForAll` and upper bound
                of an `Exists`

    **Attributes**
        fully_grounded : bool
        unique_var_slots : tuple of int
            returns the slot index of each unique variable

    """
    def __init__(self, *args, **kwds):
        super().__init__(args[-1], arity=1, propositional=True, **kwds)
        self.fully_grounded = kwds.get('fully_grounded', False)

        # dimensions to quantify over
        self._update_variables(args[:-1])
        self.unique_var_slots = tuple(
            self.var_remap[0].index(v) for v in self.unique_vars)

        self._grounding_set = set()
        self._set_activation()

    def upward(self, **kwds) -> Union[torch.Tensor, None]:
        r"""Returns sum of bounds tightening from UPWARD inference"""
        n_groundings = len(self._grounding_set)
        input_bounds = self._upward_bounds(self.operands[0])
        if input_bounds is None:
            return

        if len(self._grounding_set) > n_groundings:
            self._set_activation()
        result = self.neuron.aggregate_bounds(
            None,
            self.func(input_bounds.permute([1, 0])),
            bound=(('upper' if isinstance(self, ForAll) else 'lower')
                   if not self.fully_grounded else None))
        return result

    def _set_activation(self) -> None:
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
        operator = 'And' if isinstance(self, ForAll) else (
            'Or' if isinstance(self, Exists) else None)
        self.neuron = _NeuralActivation()(
            self.propositional, len(self._grounding_set), self.world,
            neuron={
                'weights_learning': False,
                'bias_learning': False}
            )
        self.func = self.neuron.function(
            operator, direction=Direction.UPWARD)

    @property
    def has_free_variables(self) -> bool:
        r"""Returns True if the quantifier contains free variables"""
        return {*self.variables} != {*self.var_remap[0]}

    @property
    def true_groundings(self) -> Set[Union[str, Tuple[str, ...]]]:
        r"""Returns set of groundings that are True"""
        return set(tuple(str(g.partial_grounding[slot])
                         for g in self._grounding_set
                         if self.operands[0].state(g) is Fact.TRUE
                         for slot in self.unique_var_slots))

    def _upward_bounds(self, operand: _Formula
                       ) -> Union[torch.Tensor, None]:
        r"""set Quantifer grounding table and return operand tensor"""
        operand_grounding_set = set(operand.grounding_table)
        if len(operand_grounding_set) == 0:
            return

        self._grounding_set = set([
            grounding
            for grounding in operand_grounding_set
            if _gm.is_grounding_in_bindings(self, 0, grounding)])
        result = operand.get_facts(*self._grounding_set)
        return result

    @property
    def groundings(self) -> Set[Union[str, Tuple[str, ...]]]:
        return set(map(_Grounding.eval, self._grounding_set))

    def _add_facts(self, facts: Union[Tuple[float, float], Fact, Set]) -> None:
        super()._add_facts(facts)
        self._set_activation()


class ForAll(_Quantifier):
    r"""Symbolic universal quantifier

    **Example**

    No free variables, Quantifier reduces to a proposition:

    ```python
    model.add_formulae(ForAll(p, d, birthdate(p, d)))
    ```

    Free variables, Quantifier reduces to a predicate:
        Currently not implemented

    """
    def __init__(self, *args, **kwds):
        kwds.setdefault('name', 'All_' + (args[-1][0].name if isinstance(
            args[-1], Tuple) else args[-1].name))
        super().__init__(*args, **kwds)

    def downward(self) -> torch.Tensor:
        r"""Returns sum of bounds tightening from DOWNWARD inference"""
        operand = self.operands[0]
        current_bounds = self.get_facts()
        groundings = operand.grounding_table.keys()
        result = operand.neuron.aggregate_bounds(
            [operand.grounding_table.get(g) for g in groundings],
            current_bounds)
        return result


class Exists(_Quantifier):
    r"""Symbolic existential quantifier

    **Example**

    No free variables, Quantifier reduces to a proposition:

    ```python
    model.add_formulae(Exists(p, d, birthdate(p, d)))
    ```

     Free variables, Quantifier reduces to a predicate:
        Currently not implemented

    """
    def __init__(self, *args, **kwds):
        kwds.setdefault('name', 'Some_' + (args[-1][0].name if isinstance(
            args[-1], Tuple) else args[-1].name))
        super().__init__(*args, **kwds)


class _NodeActivation:
    def __call__(self, propositional: bool, world: World, **kwds):
        return getattr(import_module(
            'lnn.neural.activations.node'), '_NodeActivation')(
            propositional, world, **kwds)


class _NeuralActivation:
    r"""Switch class, to choose a method from the correct activation class"""
    def __init__(self, type=None):
        self.neuron_type = type if type else (
            NeuralActivationClass.LukasiewiczTransparent)
        self.module = import_module(
            f"lnn.neural.methods.{self.neuron_type.name.lower()}")

    def __call__(self, propositional: bool, arity: int, world: World, **kwds):
        return getattr(self.module, self.neuron_type.name)(
            propositional, arity, world, **kwds)


class NeuralActivationClass(AutoName):
    r"""Enumeration of tnorms for neural computations"""
    Lukasiewicz = auto()
    LukasiewiczTransparent = auto()
