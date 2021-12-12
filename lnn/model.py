##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from . import _utils, _exceptions
from .symbolic.axioms import lifted_axioms
from .constants import Fact, World, Direction
from .symbolic.logic import Proposition, Predicate, _Formula

import torch
import random
import warnings
import networkx as nx
from tqdm import tqdm
from itertools import chain
from typing import Union, Dict, Tuple


class Model:
    r"""Creates a container for logical reasoning and neural learning.

    Models offer contextual segmentation between free-variable formulae and
        the instantiation of facts that may apply to those formulae under a
        specific instance.
    Models are dynamic, often constructed as empty containers to later be
        populated by rules and facts - this allows LNNs to operate under
        dynamic environments where the rules themselves are allowed to grow as
        new information emerges.
    In general a model acts as a canvas, that includes only the formulae
        specified. The model will include a

    **Parameters**

        name : str, optional
            name of contextual model

    **Attributes**

    ```raw
    graph : nx.DiGraph
        directed graph that connects nodes
    nodes : dict
        keys are names given to the formulae at instantiation and values are
        the object containers as placed inside the model
    ```

    **Warning**

    > Assumes that every node in the model has a unique name

    **Example**

    ```python
    # define the predicates
    Smokes, Asthma, Cough = map(Predicate, ['Smokes', 'Asthma', 'Cough'])
    Friends = Predicate('Friends', arity=2)

    # define the connectives/quantifiers
    x, y = map(Variable, ['x', 'y'])
    Smokers_have_friends = And(Smokes(x), Friends(x, y))
    Asmatic_smokers_cough = (
        Exists(x,
               Implies(And(Smokes(x), Asthma(x)), Cough(x)), world=AXIOM))
    Smokers_befriend_smokers = (
        ForAll(
            x, y,
            Implies(Smokers_have_friends(x, y), Smokes(y)), world=AXIOM))

    # add root formulae to model
    model = Model()
    model.add_formulae(
        Asmatic_smokers_cough,
        Smokers_befriend_smokers)

    # add data to the model
    model.add_facts({
        Smokes.name: {
            'Person_1': TRUE,
            'Person_2': UNKNOWN,
            'Person_3': UNKNOWN},
        Friends.name: {
            ('Person_1', 'Person_2'): TRUE,
            ('Person_2', 'Person_3'): UNKNOWN}})

    # reason over the model
    model.infer()

    # verify the model outputs
    model.print()
    ```

    """
    def __init__(self, name: str = 'Model'):
        self.graph = nx.DiGraph()
        self.nodes = dict()
        self.name = name

    def __getitem__(self, key: str):
        r"""model['node_name']"""
        if key in self.nodes:
            return self.nodes[key]

    def __setitem__(self, name: str, formula: _Formula):
        r"""Alias for `model.add_formulae`

        Automatically renames graph nodes, instantiated nodes need not be named
        ```python
        model['P1'] = Predicate()
        ```
        """
        self.add_formulae(formula)
        _utils.dict_rekey(self.nodes, formula.name, name)
        self.nodes[name].rename(name)
        if name in self.__dict__:
            warnings.warn(f'{name} already exists as a model variable the '
                          f'existing object {repr(self.__dict__[name])} will '
                          'be overtten')
            self.__dict__.update({name: formula})

    def __contains__(self, key: str):
        return key in self.nodes

    def add_formulae(self, *formulae: _Formula, world: World = World.OPEN):
        r"""Extend the model to include additional formulae

        Only root level formulae explicitly need to be added to the model.
        Unless otherwise specified, each root formula follows the open world
        assumption.

        **Example**

        Any formula given without cloning will be operated on directly
        To be used when working with only 1 model,
        i.e., directly referencing the model nodes

        ```python
        model.add_formulae(Predicate('P1'))

        ```

        or directly modifying the formula in the user space:

        ```python
        model = Model()
        P1 = Predicate('P1')
        model.add_formulae(P1)

        ```

        the former requires modifications to be accessed via `model['P1']`
        while the latter will store any changes to made by the model,
        directly in `P1`

        """
        self._add_rules(*formulae, world=world)

    def add_propositions(self, *names: str, **kwds):
        ret = []
        for name in names:
            self[name] = Proposition(**kwds)
            ret.append(self[name])
        return ret[0] if len(ret) == 1 else ret

    def add_predicates(self, arity: int, *names: str, **kwds):
        ret = []
        for name in names:
            self[name] = Predicate(arity=arity, **kwds)
            ret.append(self[name])
        return ret[0] if len(ret) == 1 else ret

    def _add_rules(self, *formulae: _Formula, world: World = World.OPEN):
        for f in formulae:
            self.graph.add_node(f)
            self.graph.add_edges_from(f.edge_list)
        self.nodes.update({node.name: node for node in self.graph.nodes})
        if world is not World.OPEN:
            [self[f.name]._set_world(world) for f in formulae]

    def add_facts(self,
                  facts: Dict[str,
                              Union[Union[Fact, Tuple[float, float]],
                                    Dict[Union[str, Tuple[str, ...]],
                                         Union[Fact, Tuple[float, float]]]]]):
        r"""Append facts to the model

        Assumes that the formulae that require facts have already been inserted
        into the model, see
        [add_formulae](https://ibm.github.io/LNN/lnn/LNN.html#lnn.Model.add_formulae)  # noqa: E501
        for more details

        **Parameters**

            # propositional
            facts : dict
                key : str
                    This is the unique node name stored in the model
                    can be reference either by the associated string in the
                    model-scope or extracting the node`.name` stored in the
                    user-scope
                value :  Fact or Bounds
                    Facts may be the flags for classical bounds or bounds can
                    be directly set as a list of two floats, representing lower
                    and upper bounds

            # first-order logic
            facts : dict
                key : str
                    This is the unique node name stored in the model
                    can be reference either by the associated string in the
                    model-scope or extracting the node`.name` stored in the
                    user-scope
                value :  dict
                    key : str or tuple-of-str
                        This inner key represents the first-order grounding or
                        the propositionalised/instantiated binding that applies
                        the the free variable. It represents a single row
                        within the bounds table
                    value :  Fact or Bounds
                        This inner value represents the facts, given either as
                        a flag for classical bounds or directly as a list of
                        two floats, representing the lower and upper bounds.
                        This fact is set as the associated truth for the
                        grounding key above.

        **Example**

        ```python
        # propositional
        P = Proposition('Person')
        model.add_facts({'Person': Facts.TRUE})
        ```

        ```python
        # first-order logic
        P = Predicate('Person')
        B = Predicate('Birthdate', arity=2)
        model.add_facts(
            {'Person': {
                'Barack Obama': TRUE,
                'Bo': FALSE},
             B.name: {
                 ('Barack Obama', '04 August 1961'): TRUE,
                 ('Bo', '09 October 2008'): TRUE}
            })
        ```

        """
        for formula, fact in facts.items():
            _exceptions.AssertFormulaInModel(self, formula)
            if self[formula].propositional:
                _exceptions.AssertBounds(fact)
            else:
                _exceptions.AssertFOLFacts(fact)
            self[formula]._add_facts(fact)

    def add_labels(
            self,
            labels: Union[
                Dict[str, Union[Tuple[float, float], Fact]],
                Dict[str, Dict[Union[str, Tuple[str, ...]],
                               Union[Tuple[float, float], Fact]]]]):
        r"""Append labels to the model

        Adding labels to formulae in the model follows the same dictionary
        input API as
        [adding facts](https://ibm.github.io/LNN/lnn/LNN.html#lnn.Model.add_facts).

        """
        for formula, label in labels.items():
            _exceptions.AssertFormulaInModel(self, formula)
            if self[formula].propositional:
                _exceptions.AssertBounds(label)
            else:
                _exceptions.AssertFOLFacts(label)
            self[formula]._add_labels(label)

    def _traverse_execute(self,
                          func: str,
                          direction: Direction = Direction.UPWARD,
                          source: _Formula = None,
                          **kwds):
        r"""Traverse over the model and execute a node operation

        Traverses through graph from `source` in the given `direction`
            and execute `func` at each node

        """
        _exceptions.AssertValidDirection(direction)
        nodes = None
        if direction is Direction.UPWARD:
            nodes = list(nx.dfs_postorder_nodes(self.graph, source))
        elif direction is Direction.DOWNWARD:
            nodes = list(reversed(list(nx.dfs_postorder_nodes(
                self.graph, source))))

        coalesce = torch.tensor(0.)
        for node in nodes:
            val = getattr(node, func)(**kwds) if hasattr(node, func) else None
            coalesce = coalesce + val if val is not None else coalesce + 0.
        return coalesce

    def lifted_preprocessing(self, n: Union[float, int]):
        print(f'\n{"*" * 75}\n{"":<20} Lifted Reasoning Preprocessing')
        axioms = lifted_axioms()
        subformulae = list()
        for node in self.nodes:
            if self[node].world is World.AXIOM:
                subformulae.append(self[node])
        if len(subformulae) == 0:
            return
        for _ in range(int(n)):
            axiom, k = random.choice(list(axioms.items()))
            nodes = random.choices(subformulae, k=k)
            result = axiom(nodes)
            if result and result.name not in self.nodes:
                self.add_formulae(result)
                subformulae.append(self[str(result)])
                print(f'Added {str(result)}')
        print(f'{"*" * 75}')

    def infer(self,
              direction: Direction = None,
              source: _Formula = None,
              max_steps: int = None,
              **kwds):
        r"""Reasons over all possible inferences until convergence

        **Return**

            steps : int
                The number of steps taken to converge
            facts_inferred : Tensor
                Sum of bounds tightening from inference updates

        **Parameters**

            direction : Direction
                Can be specified as either UPWARD or DOWNWARD inference, a
                single pass of that direction will be applied
                If unspecified, None, defaults to the LNN naive inference
                strategy of doing upward and downward inference until
                convergence
            source : node
                Specifies starting node for
                [depth-first search traversal](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes.html#networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes)  # noqa: E501
                The node can be extracted using __getitem__ from the node name
            max_steps: int
                Limits the inference to a specified number of passes of the
                naive traversal strategy
            kwds
                lifted : bool or float
                    Computes lifted inference preprocessing to expand the
                    knowledge by randomly searching for axioms that can be
                    applied to the network. If True, defaults to 1e3 random
                    nodes.

        """
        lifted = kwds.get('lifted')
        if lifted:
            self.lifted_preprocessing(1e3 if lifted is True else lifted)
        direction = ([Direction.UPWARD, Direction.DOWNWARD]
                     if not direction else [direction])
        converged = False
        steps = 0
        facts_inferred = torch.tensor(0)
        while not converged:
            bounds_diff = 0.
            for d in direction:
                bounds_diff = bounds_diff + self._traverse_execute(
                    d.value.lower(), d, source, **kwds)
            converged = True if direction in (
                [[Direction.UPWARD], [Direction.DOWNWARD]]) else (
                    bounds_diff <= 1e-7)
            facts_inferred = facts_inferred + bounds_diff
            steps = steps + 1
            if max_steps is not None and steps >= max_steps:
                break
        return steps, facts_inferred

    def forward(self, *args, **kwds):
        return self.infer(*args, **kwds)

    reason = inference = forward

    def upward(self, **kwds):
        return self.infer(Direction.UPWARD, **kwds)

    def downward(self, **kwds):
        return self.infer(Direction.DOWNWARD, **kwds)

    def train(self, **kwds):
        r"""Train the model

        Reasons across the model until convergence using the standard inference
        strategy - equivalent to running a NN in the forward direction.
        At the end of each reasoning pass losses are calculated according to a
        predefined or custom loss and model parameters are updated.
        An epoch constitutes all computation until parameters take a step.

        **Parameters**
            kwds
                losses: list or dict
                    predefined losses include
                     ['contradiction', 'uncertainty', 'logical', 'supervised']
                    If given in dict form, coefficients of each loss can be
                    specified as a float value. The value can alternatively
                    specify additional parameters for each loss calculation
                    using a dict

        **Returns**

            epoch + 1: int
            total_loss: tuple
                running_loss: list
                    sum of loss at the end of each epoch
                loss_history: Tensor
                    individual loss components as specified by `losses` kwd

        **Example**

        ```python
        # construct the model from formulae
        model = Model()
        p1, p2 = map(Predicate, ['P1', 'P2'])
        x = Variable('x')
        model['AB'] = And(p1(x), p2(x))

        # add data to the model
        model.add_facts({
            p1.name: {
                '0': TRUE,
                '1': TRUE,
                '2': FALSE,
                '3': FALSE
            },
            p2.name: {
                '0': TRUE,
                '1': FALSE,
                '2': TRUE,
                '3': FALSE,
            }
        })

        # add supervisory targets
        model.add_labels({
            'AB': {
                '0': TRUE,
                '1': FALSE,
                '2': TRUE,
                '3': FALSE,
            }
        })

        # train the model and output results
        model.train(losses=['supervised'])
        model.print(params=True)
        ```

        """
        optimizer = kwds.get(
            'optimizer',
            torch.optim.Adam(
                kwds.get('parameters', self.parameters()),
                lr=kwds.get('learning_rate', 5e-2)))
        running_loss, loss_history, inference_history = [], [], []
        for epoch in tqdm(
                range(int(kwds.get('epochs', 3e2))),
                desc='training epoch', disable=not kwds.get('pbar', False)):
            optimizer.zero_grad()
            if epoch > 0:
                self.reset_bounds()
            self.increment_param_history(kwds.get('parameter_history'))
            _, facts_inferred = self.infer(**kwds)
            loss_fn = self.loss_fn(kwds.get('losses'))
            loss = sum(loss_fn)
            if not loss.grad_fn:
                raise RuntimeError(
                    'graph loss found no gradient... '
                    'check learning flags, loss functions, labels '
                    'or switch to reasoning without learning')
            loss.backward()
            optimizer.step()
            self._project_params()
            running_loss.append(loss.item())
            loss_history.append(
                [L.clone().detach().tolist() for L in loss_fn])
            inference_history.append(facts_inferred.item())
            if loss <= 1e-5 and kwds.get('stop_at_convergence', True):
                break
        self.increment_param_history(kwds.get('parameter_history'))
        return (running_loss, loss_history), inference_history

    def parameters(self):
        result = list(chain.from_iterable(
            [self[n].parameters() for n in self.nodes]))
        return result

    def parameters_grouped_by_neuron(self):
        result = list()
        for n in self.nodes:
            param_group = dict()
            param_group['params'] = list()
            param_group['param_names'] = list()
            for name, param in self[n].named_parameters():
                param_group['params'].append(param)
                param_group['param_names'].append(name)
            param_group['neuron_type'] = self[n].__class__.__name__
            result.append(param_group)
        return result

    def named_parameters(self):
        result = dict()
        for n in self.nodes:
            result.update({
                f'{n}.{name}': param
                for name, param in self[n].named_parameters()})
        return result

    def fit(self, **kwds):
        """Alias for train"""
        return self.train(**kwds)

    learn = fit

    def loss_fn(self, losses):
        loss_names = ['contradiction', 'uncertainty', 'logical', 'supervised',
                      'custom']
        if losses is None:
            raise Exception(
                'no loss function given, '
                f'expected losses from the following {loss_names}')
        elif isinstance(losses, list):
            losses = {c: None for c in losses}
        result = list()
        for loss in losses:
            if loss in loss_names:
                if loss == 'custom':
                    if not isinstance(losses[loss], dict):
                        raise TypeError(
                            'custom losses expected as a dict with keys as '
                            'name of the loss and values as function '
                            'definitions')
                    for loss_fn in losses[loss].values():
                        coalesce = torch.tensor(0.)
                        for node in list(nx.dfs_postorder_nodes(self.graph)):
                            coalesce = coalesce + loss_fn(node)
                        result.append(coalesce)
                else:
                    kwds = losses[loss] if (
                        isinstance(losses[loss], dict)) else (
                        {'coeff': losses[loss]})
                    result.append(self._traverse_execute(
                        f'{loss}_loss', **kwds))
        return result

    def print(self,
              header_len: int = 50,
              roundoff: int = 5,
              params: bool = False,
              grads: bool = False,
              ):
        n = header_len + 25
        print('\n' + '*' * n + f'\n{"":<{n/2 - 5}}LNN {self.name}\n')
        self._traverse_execute('print',
                               Direction.DOWNWARD,
                               header_len=header_len,
                               roundoff=roundoff,
                               params=params,
                               grads=grads)
        print('*' * n)

    def flush(self):
        self._traverse_execute('flush')

    def reset_bounds(self):
        self._traverse_execute('reset_bounds')

    def _project_params(self):
        self._traverse_execute('project_params')

    def increment_param_history(self, parameter_history):
        if parameter_history:
            self._traverse_execute(
                'increment_param_history',
                parameter_history=parameter_history)
