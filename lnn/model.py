##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

import itertools as itls
from collections.abc import Iterable
from typing import Union, Dict, Tuple, List

from . import viz
from . import _exceptions, _utils
from .constants import Fact, World, Direction, Loss
from .symbolic.logic import Proposition, Predicate, Formula

import torch
import random
import logging
import datetime
import networkx as nx
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt

_utils.logger_setup(flush=True)


class Model(nn.Module):
    r"""Creates a container for logical reasoning and neural learning.

    Models define a theory or a collection of formulae, with additional reasoning and
        learning functionality that can be applied to each formula in the model.
        In contrast to standard FOL where the existence of a formula symbol assumes a
        `True` truth value, the data associated with LNN formulae can take on any
        classical truth (Fact) or belief bounds (a range real-values).

    Models are also dynamic, instantiated as empty containers which are later populated
        with knowledge rules and data. This additionally allows LNNs to operate in
        dynamic environments whereby the knowledge acquired may grow as new
        information becomes available.

    Parameters
    ------------
    name : str, optional
        Name of contextual model, defaults to "Model"

    Attributes
    ----------
    graph : nx.DiGraph
        Directed graph that connects nodes, pointing from operator to operand nodes.
    nodes : dict
        Each formula is keyed by a formula_number, with the value as the formula object.
    query : Formula
        A formula node that is set as the current query - allows the model to be used in QA/theorem proving whereby inference is governed towards solving the query.

    Examples
    --------
    ```python
    # define the predicates
    x, y = Variables("x", "y")
    Smokes, Asthma, Cough = Predicates("Smokes", "Asthma", "Cough")
    Friends = Predicate("Friends", arity=2)

    # define the connectives/quantifiers
    Smokers_have_friends = And(Smokes(x), Friends(x, y))
    Asthmatic_smokers_cough = (
        Exists(x, Implies(And(Smokes(x), Asthma(x)), Cough(x))))
    Smokers_befriend_smokers = (
        Forall(x, y, Implies(Smokers_have_friends(x, y), Smokes(y))))

    # add root formulae to model
    model = Model()
    model.add_knowledge(
        Asthmatic_smokers_cough,
        Smokers_befriend_smokers)

    # add data to the model
    model.add_data({
        Smokes: {
            "Person_1": Fact.TRUE,
            "Person_2": Fact.UNKNOWN,
            "Person_3": Fact.UNKNOWN},
        Friends: {
            ("Person_1", "Person_2"): Fact.TRUE,
            ("Person_2", "Person_3"): Fact.UNKNOWN}})

    # reason over the model
    model.infer()

    # verify the model outputs
    model.print()
    ```

    """

    def __init__(
        self,
        knowledge: Union[Formula, Iterable[Formula]] = None,
        data: Dict = None,
        name: str = "Model",
    ):
        super(Model, self).__init__()
        self.graph = nx.DiGraph()
        self.nodes = dict()
        self.node_names = dict()
        self.node_structures = dict()
        self.num_formulae = 0
        self.name = name
        self.query = None
        self._converge = None
        if knowledge:
            if isinstance(knowledge, Iterable):
                self.add_knowledge(*knowledge)
            else:
                self.add_knowledge(knowledge)
        if data:
            self.add_data(data)
        logging.info(f" {name} {datetime.datetime.now()} ".join(["*" * 22] * 2))

    def __getitem__(
        self, formula: Union[Formula, int]
    ) -> Union[Formula, List[Formula]]:
        r"""Returns a formula object from the model.

        If the formula is in the model, return the formula
            - for backward compatibility
        if multiple formula exists in the model with the same structure,
            return a list of all the relevant nodes

        """
        if isinstance(formula, int):
            return self.nodes[formula]
        if formula.formula_number is not None and formula.formula_number in self.nodes:
            return self.nodes[formula.formula_number]
        if formula.structure in self.node_structures:
            result = self.node_structures[formula.structure]
            return (
                result
                if len(self.node_structures[formula.structure]) > 1
                else result[0]
            )

    def __contains__(self, formula: Formula):
        if formula.formula_number and formula.formula_number in self.nodes:
            return True
        return formula.structure in self.node_structures

    def set_query(self, formula: Formula, world=World.OPEN, converge=False):
        r"""Inserts a query node into the model and maintains a handle on the node.

        Parameters
        ----------
        formula : Formula
            Name of contextual model
        world : World
            Default behavior of the formula. If unspecified, assumes open world assumption.

        Notes
        -----
        The query formula will be added to the model and will not be removed, even if a new query is defined using this function.

        """
        self.add_knowledge(formula, world=world)
        self.query = formula
        self._converge = converge

    def infer_query(self, *args, **kwds) -> Tuple[Tuple[int, int], torch.Tensor]:
        r"""Reasons only over the stored query.

        Is the same as calling [model.infer](#lnn.Model.infer) but setting the source
        node as [model.query](#lnn.Model.set_query)."""
        if self.query:
            return self.infer(*args, **kwds, source=self.query)

    def add_formulae(self, *args, **kwds):
        raise NameError(f"`add_formulae` is deprecated, use `add_knowledge` instead")

    def add_knowledge(self, *formulae: Formula, world: World = None):
        r"""Extend the model to include additional formulae.

        Only root level formulae explicitly need to be added to the model.
        Unless otherwise specified, each root formula follows the open world
        assumption.

        Examples
        --------
        ```python
        P, Q = Predicates("P1", "Q")
        model.add_knowledge(P, Q)

        ```
        creates the predicate and inserts into the model

        or

        ```python
        model = Model()
        P1 = Predicate("P1")
        P2 = Predicate("P2", 2)
        P3 = Predicate("P3", 3)
        model.add_knowledge(
            And(P1(x), P2(x, y)),
            Implies(P2(x, y), P3(x, y, z))
        )

        ```

        inserts the formulae roots into the model and appropriately includes
        all subformulae also into the scope of the model.

        Any formulae that directly require inquiry should first be created in
        the user scope and thereafter inserted into the model for reference
        after reasoning/learning

        e.g.

        ```python
        model = Model()
        P1 = Predicate("P1")
        P2 = Predicate("P2", 2)
        P3 = Predicate("P3", 3)
        my_and = And(P1(x), P2(x, y))
        model.add_knowledge(
            my_and,
            Implies(P2(x, y), P3(x, y, z))
        )

        ...
        model.infer()
        ...

        my_and.state()

        ```

        """
        self._add_knowledge(*formulae, world=world)

    def add_propositions(self, *names: str, **kwds):
        ret = []
        for name in names:
            P = Proposition(name, **kwds)
            self.add_knowledge(P)
            ret.append(P)
        return ret[0] if len(ret) == 1 else ret

    def add_predicates(self, arity: int, *names: str, **kwds):
        ret = []
        for name in names:
            P = Predicate(name, arity=arity, **kwds)
            self.add_knowledge(P)
            ret.append(P)
        return ret[0] if len(ret) == 1 else ret

    def replace_graph_edge(
        self, old_edge: (Formula, Formula), new_edge: (Formula, Formula)
    ):
        self.graph.remove_edge(*old_edge)
        self.graph.add_edge(*new_edge)

    def _add_knowledge(self, *formulae: Formula, world: World = None):
        for idx, f in enumerate(formulae):
            _exceptions.AssertFormula(f)
            self.graph.add_node(f)
            self.graph.add_edges_from(f.edge_list)
            self.num_formulae = f.set_formula_number(self.num_formulae) + 1
        for node in self.graph.nodes:
            if node.structure in self.node_structures:
                if node not in self.node_structures[node.structure]:
                    self.node_structures[node.structure].append(node)
            else:
                self.node_structures.update({node.structure: [node]})
            if node.name in self.node_names:
                if node not in self.node_names[node.name]:
                    self.node_names[node.name].append(node)
            else:
                self.node_names.update({node.name: [node]})
            self.nodes[node.formula_number] = node

        if world:
            for f in formulae:
                f.reset_world(world)

    def add_facts(self, *args, **kwds):
        raise NameError(f"`add_facts` is deprecated, use `add_data` instead")

    def add_data(
        self,
        data: Dict[
            Formula,
            Union[
                Union[bool, Fact, float, Tuple[float, float]],
                Dict[
                    Union[str, Tuple[str, ...]],
                    Union[bool, Fact, float, Tuple[float, float]],
                ],
            ],
        ],
    ):
        r"""Add data to select formulae in the model, in the form of classical facts or belief bounds.

        Data given is a Fact or belief bounds assumes a propositional formula.
        Data given in a dict assumes a first-order logic formula,
            keyed by the grounding and a value given as a Fact or belief bounds.

        Parameters
        ----------
        data : a dict of Fact, belief bounds or dict
            The dict is keyed by the formula for which data is to be added, with the truths as the value. For propositional formulae, truths are given as either Facts or belief bounds. These beliefs can be given as a bool, float or a float-range, i.e. a tuple of 2 floats. For first-order logic formula, inputs truths are given as a dict. This is further keyed by the grounding (a str for unary formlae or tuple of strings of larger arities), with values also as Facts or bounds on beliefs.

        Examples
        --------
        ```python
        # propositional
        P = Proposition("Person")
        model.add_data({
            P: Fact.TRUE
        })
        ```
        ```python
        # first-order logic
        Person = Predicate("Person")
        BD = Predicate("Birthdate", 2)
        model.add_data({
            Person: {
                "Barack Obama": Fact.TRUE,
                "Bo": (.1, .4)
            },
            BD: {
                ("Barack Obama", "04 August 1961"): Fact.TRUE,
                ("Bo", "09 October 2008"): (.6, .75)
            }
        })
        ```

        Warning
        -------
        Assumes that the formulae have already been inserted into the model, see [add_knowledge](https://ibm.github.io/LNN/lnn/LNN.html#lnn.Model.add_knowledge) for more details.

        """
        for formula, fact in data.items():
            if not isinstance(formula, Formula):
                raise TypeError(
                    "formula expected of type Formula, received "
                    f"{formula.__class__.__name__}"
                )
            _exceptions.AssertFormulaInModel(self, formula)
            if formula.propositional:
                _exceptions.AssertBounds(fact)
            else:
                _exceptions.AssertFOLFacts(fact)
            formula.add_data(fact)

    def add_labels(
        self,
        labels: Dict[
            Formula,
            Union[
                Union[Fact, Tuple[float, float]],
                Dict[Union[str, Tuple[str, ...]], Union[Fact, Tuple[float, float]]],
            ],
        ],
    ):
        r"""Add labels to select formulae in the model, in the form of classical facts or belief bounds.

        Labels given is a Fact or belief bounds assumes a propositional formula.
        Labels given in a dict assumes a first-order logic formula,
            keyed by the grounding and a value given as a Fact or belief bounds.

        Parameters
        ----------
        labels : a dict of Fact, belief bounds or dict
            The dict is keyed by the formula for which data is to be added, with the truths as the value. For propositional formulae, truths are given as either Facts or belief bounds (a tuple of 2 floats). For first-order logic formula, inputs truths are given as a dict. This is further keyed by the grounding (a str for unary formlae or tuple of strings of larger arities), with values also as Facts or bounds on beliefs.

        Examples
        --------
        ```python
        # propositional
        P = Proposition("Person")
        model.add_labels({
            P: Fact.TRUE
        })
        ```
        ```python
        # first-order logic
        Person = Predicate("Person")
        BD = Predicate("Birthdate", 2)
        model.add_labels({
            Person: {
                "Barack Obama": Fact.TRUE,
                "Bo": (.1, .4)
            },
            BD: {
                ("Barack Obama", "04 August 1961"): Fact.TRUE,
                ("Bo", "09 October 2008"): (.6, .75)
            }
        })
        ```

        Warning
        -------
        Assumes that the formulae have already been inserted into the model, see [add_knowledge](https://ibm.github.io/LNN/lnn/LNN.html#lnn.Model.add_knowledge) for more details.

        """
        for formula, label in labels.items():
            _exceptions.AssertFormulaInModel(self, formula)
            if formula.propositional:
                _exceptions.AssertBounds(label)
            else:
                _exceptions.AssertFOLFacts(label)
            formula.add_labels(label)

    def _traverse_execute(
        self,
        func: str,
        direction: Direction = Direction.UPWARD,
        source: Formula = None,
        **kwds,
    ):
        r"""Traverse over the subgraph and execute an operation per node starting from
        source.

        Traverses through graph from `source` in the given `direction`
            and execute `func` at each node

        """
        _exceptions.AssertValidDirection(direction)
        nodes = None
        if direction is Direction.UPWARD:
            nodes = list(nx.dfs_postorder_nodes(self.graph, source))
        elif direction is Direction.DOWNWARD:
            nodes = list(reversed(list(nx.dfs_postorder_nodes(self.graph, source))))

        coalesce = torch.tensor(0.0)
        for node in nodes:
            val = getattr(node, func)(**kwds) if hasattr(node, func) else None
            coalesce = coalesce + val if val is not None else coalesce
        if coalesce and func in [d.value.lower() for d in Direction]:
            logging.info(f"{direction.value} INFERENCE RESULT:{coalesce}")
        return coalesce

    def infer(
        self,
        direction: Direction = None,
        source: Formula = None,
        max_steps: int = None,
        **kwds,
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        r"""Reasons over all possible inferences until convergence

        Parameters
        ----------
        direction : {Direction.UPWARD, Direction.DOWNWARD}, optional
            Can be specified as either UPWARD or DOWNWARD inference, a single pass of that direction will be applied. If unspecified, defaults to the LNN naive inference strategy of doing inference until convergence.
        source : node, optional
            Specifies starting node for [depth-first search traversal](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes.html#networkx.algorithms.traversal.depth_first_search.dfs_postorder_nodes). Specifying a node here will compute reasoning (until convergence) on the subgraph, with the specified source is the root of the subgraph.
        max_steps: int, optional
            Limits the inference to a specified number of passes of the naive traversal strategy. If unspecified, the steps will not be limited, i.e. inference will take place until convergence.

        Returns
        -------
        (steps, facts_inferred) : Tuple[tuple of 2 ints, torch.Tensor]

        """

        return self._infer(
            direction=direction,
            source=source,
            max_steps=max_steps,
            **kwds,
        )

    def _infer(
        self,
        direction: Direction = None,
        source: Formula = None,
        max_steps: int = None,
        **kwds,
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        r"""Implementation of model inference."""
        direction = (
            [Direction.UPWARD, Direction.DOWNWARD] if not direction else [direction]
        )
        converged = False
        additional_axioms, steps, facts_inferred = 0, 0, 0

        while not converged:
            if self.query and self.query.is_classically_resolved and not self._converge:
                logging.info("=" * 22)
                logging.info(
                    f"QUERY PROVED AS {self.query.world_state(True)} for "
                    f"'{self.query.name}'"
                )
                break
            logging.info("-" * 22)
            logging.info(f"REASONING STEP:{steps}")
            bounds_diff = 0.0
            for d in direction:
                bounds_diff += self._traverse_execute(
                    d.value.lower(), d, source, **kwds
                )
            converged_bounds = (
                True
                if direction in ([[Direction.UPWARD], [Direction.DOWNWARD]])
                else bounds_diff <= 1e-7
            )
            if converged_bounds:
                converged = True
                logging.info("NO UPDATES AVAILABLE, TRYING A NEW AXIOM")
            facts_inferred += bounds_diff
            steps += 1
            if max_steps and steps >= max_steps:
                break
        logging.info("=" * 22)
        logging.info(
            f"INFERENCE CONVERGED WITH {facts_inferred} BOUNDS "
            f"UPDATES IN {steps} REASONING STEPS "
        )
        logging.info("*" * 78)
        return steps, facts_inferred

    def upward(self, **kwds):
        r"""Performs upward inference for each node in the model from leaf to root."""
        return self.infer(Direction.UPWARD, **kwds)

    def downward(self, **kwds):
        r"""Performs downward inference for each node in the model from root to leaf."""
        return self.infer(Direction.DOWNWARD, **kwds)

    def train(self, losses: Union[Loss, List[Loss], Dict[List[Loss], float]], **kwds):
        r"""Train the model.

        Reasons across the model until convergence using the standard inference
        strategy - equivalent to running a NN in the forward direction.
        At the end of each reasoning pass losses are calculated according to a
        predefined or custom loss and model parameters are updated.
        An epoch constitutes all computation until parameters take a step.

        Parameters
        ------------
        losses: Loss, list or dict of losses
            Predefined losses expected from the fixed Loss constants. If given in dict form, coefficients of each loss can be specified as a float value. The value can alternatively specify additional parameters for each loss calculation using a dict.
        optimizer : pytorch optimizer, optional
            Custom optimizers should be instantiated with the model parameters using `model.parameters()`. If unspecified, defaults to [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam).
        learning_rate : float, optional
            If unspecified, defaults to 5e-2.
        epochs : float, optional
            Number of training epochs. If unspecified, trains for 3e2 epochs.
        pbar : bool, optional
            Prints out a tqdm training progress bar. If unspecified, does not print out.

        Returns
        -------
        (epochs, total_loss) : Tuple[int, Tuple[List, Tensor]]
            A tuple of variables are returned. The `epochs` is number of epochs trained before stopped/converged + 1. The `total_loss` returns a tuple of 2 values: first is the `running_loss` as a list for the sum of loss at the end of each epoch; then the `loss_history`, which is a Tensor of individual loss components as specified by the `losses` argument.

        Examples
        --------
        ```python
        # construct the model from formulae
        model = Model()
        p1, p2 = Predicates("P1", "P2")
        x = Variable("x")
        AB = And(p1(x), p2(x))
        model.add_knowledge(AB)

        # add data to the model
        model.add_data({
            p1: {
                "0": Fact.TRUE,
                "1": Fact.TRUE,
                '2': Fact.FALSE,
                '3': Fact.FALSE
            },
            p2: {
                '0': Fact.TRUE,
                '1': Fact.FALSE,
                '2': Fact.TRUE,
                '3': Fact.FALSE,
            }
        })

        # add supervisory targets
        model.add_labels({
            AB: {
                '0': Fact.TRUE,
                '1': Fact.FALSE,
                '2': Fact.TRUE,
                '3': Fact.FALSE,
            }
        })

        # train the model and output results
        model.train(losses=Loss.SUPERVISED)
        model.print(params=True)
        ```

        """
        optimizer = kwds.get(
            "optimizer",
            torch.optim.Adam(
                kwds.get("parameters", self.parameters()),
                lr=kwds.get("learning_rate", 5e-2),
            ),
        )
        running_loss, loss_history, inference_history = [], [], []
        for epoch in tqdm(
            range(int(kwds.get("epochs", 3e2))),
            desc="training epoch",
            disable=not kwds.get("pbar", False),
        ):
            optimizer.zero_grad()
            if epoch > 0:
                logging.info(" PARAMETER STEP ".join(["#" * 31] * 2))
                self.reset_bounds()
            self.increment_param_history(kwds.get("parameter_history"))
            _, facts_inferred = self.infer(**kwds)
            loss_fn = self.loss_fn(losses)
            loss = sum(loss_fn)
            if not loss.grad_fn:
                break
            if loss and len(loss_fn) > 1:
                logging.info(f"TOTAL LOSS: {loss}")
            loss.backward()
            optimizer.step()
            self._project_params()
            running_loss.append(loss.item())
            loss_history.append([L.clone().detach().tolist() for L in loss_fn])
            inference_history.append(facts_inferred.item())
            if loss <= 1e-7 and kwds.get("stop_at_convergence", True):
                break
        self.reset_bounds()
        self.infer(**kwds)
        self.increment_param_history(kwds.get("parameter_history"))
        return (running_loss, loss_history), inference_history

    def parameters(self):
        result = list(
            itls.chain.from_iterable([n.parameters() for n in self.nodes.values()])
        )
        return result

    def parameters_grouped_by_neuron(self):
        result = list()
        for n in self.nodes.values():
            param_group = dict()
            param_group["params"] = list()
            param_group["param_names"] = list()
            for name, param in n.named_parameters():
                param_group["params"].append(param)
                param_group["param_names"].append(name)
            param_group["neuron_type"] = n.__class__.__name__
            result.append(param_group)
        return result

    def named_parameters(self):
        result = dict()
        for n in self.nodes.values():
            result.update(
                {f"{n}.{name}": param for name, param in n.named_parameters()}
            )
        return result

    def loss_fn(self, losses):
        if losses is None:
            raise Exception(
                "no loss function given, "
                f"expected losses from the following {[l.name for l in Loss]}"
            )
        elif isinstance(losses, Loss):
            losses = {losses: None}
        elif isinstance(losses, list):
            losses = {c: None for c in losses}
        result = list()
        for loss in losses:
            _exceptions.AssertLossType(loss)
            if loss == Loss.CUSTOM:
                if not isinstance(losses[loss], dict):
                    raise TypeError(
                        "custom losses expected as a dict with keys as "
                        "name of the loss and values as function "
                        "definitions"
                    )
                for loss_fn in losses[loss].values():
                    coalesce = torch.tensor(0.0)
                    for node in list(nx.dfs_postorder_nodes(self.graph)):
                        coalesce = coalesce + loss_fn(node)
                    result.append(coalesce)
            else:
                kwds = (
                    losses[loss]
                    if (isinstance(losses[loss], dict))
                    else ({"coeff": losses[loss]})
                )
                result.append(
                    self._traverse_execute(f"_{loss.value.lower()}_loss", **kwds)
                )
            if result[-1]:
                logging.info(f"{loss.value.upper()} LOSS {result[-1]}")
        return result

    def print(
        self,
        source: Formula = None,
        header_len: int = 50,
        roundoff: int = 5,
        params: bool = False,
        grads: bool = False,
        numbering: bool = False,
    ):
        n = header_len + 25
        print("\n" + "*" * n + f'\n{"":<{n / 2 - 5}}LNN {self.name}\n')
        self._traverse_execute(
            "print",
            Direction.DOWNWARD,
            source=source,
            header_len=header_len,
            roundoff=roundoff,
            params=params,
            grads=grads,
            numbering=numbering,
        )
        print("*" * n)

    def plot_graph(
        self, formula_number: bool = False, edge_variables: bool = False, **kwds
    ):
        options = {
            "with_labels": False,
            "arrows": False,
            "edge_color": "#d0e2ff",
            "node_color": "#ffffff",
            "node_size": 16,
            "font_size": 9,
        }
        options.update(kwds)
        pos = viz.get_pos(self)
        nx.draw(self.graph, pos, **options)
        nx.draw_networkx_labels(
            self.graph,
            pos,
            dict(
                [
                    (
                        (node, node.formula_number)
                        if formula_number
                        else (
                            (node, node.connective_str)
                            if hasattr(node, "connective_str")
                            else (node, node.name)
                        )
                    )
                    for node in self.graph
                ]
            ),
        )
        if edge_variables:
            labels = {
                edge: _utils.list_to_str(
                    edge[0].operand_map[edge[0].operands.index(edge[1])]
                )
                for edge in self.graph.edges
                if isinstance(edge[1], Predicate)
            }
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                labels,
            )
        plt.show()

    def flush(self):
        self._traverse_execute("flush")

    def reset_bounds(self):
        self._traverse_execute("reset_bounds")

    def _project_params(self):
        self._traverse_execute("project_params")

    def increment_param_history(self, parameter_history):
        if parameter_history:
            self._traverse_execute(
                "increment_param_history", parameter_history=parameter_history
            )

    def has_contradiction(self):
        return (
            True
            if any([node.is_contradiction() for node in self.nodes.values()])
            else False
        )

    @property
    def shape(self):
        groundings = sum(
            [
                1 if node.propositional else len(node.groundings)
                for node in self.nodes.values()
            ]
        )
        return [len(self.nodes), groundings]
