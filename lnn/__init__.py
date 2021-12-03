##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from .model import Model
from .symbolic import (Proposition, Predicate, And, Or,
                       Implies, Bidirectional, Not, ForAll, Exists, Variable,
                       NeuralActivationClass)
from .utils import (truth_table, truth_table_dict, predicate_truth_table,
                    plot_graph, plot_loss, plot_params, fact_to_bool,
                    bool_to_fact)
from .constants import Fact, World, Direction, Join

# constants
UPWARD = Direction.UPWARD
DOWNWARD = Direction.DOWNWARD
TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN
CONTRADICTION = Fact.CONTRADICTION
AXIOM = World.AXIOM
OPEN = World.OPEN
CLOSED = World.CLOSED
Lukasiewicz = NeuralActivationClass.Lukasiewicz
LukasiewiczTransparent = NeuralActivationClass.LukasiewiczTransparent
OUTER = Join.OUTER
INNER = Join.INNER
OUTER_PRUNED = Join.OUTER_PRUNED

__all__ = [
    'Model',
    'Proposition', 'Predicate',
    'And', 'Or', 'Implies', 'Bidirectional', 'Not',
    'ForAll', 'Exists', 'Variable',

    'truth_table', 'truth_table_dict', 'predicate_truth_table',
    'fact_to_bool', 'bool_to_fact',
    'plot_graph', 'plot_loss', 'plot_params',

    'Direction',
    'UPWARD', 'DOWNWARD',

    'Fact', 'World',
    'OPEN', 'CLOSED', 'AXIOM',
    'TRUE', 'FALSE', 'UNKNOWN', 'CONTRADICTION',

    'Lukasiewicz', 'LukasiewiczTransparent',

    'Join',
    'OUTER', 'INNER', 'OUTER_PRUNED'
]
