##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Implies, Proposition, World, Direction, Fact, Loss

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test_true_operator():
    """decrease bias for contradictory facts

    given a True Implies, for (True, False) inputs
        all inputs are telling the operator that it is incorrect
        - computation should recognise that the contradiction comes from
        the assignment of the truth values and not the inputs
        - the input weights should therefore remain the same and the bias
        should be corrected to fix the user's mistake
    under this configuration, the parent operand defaults to the expected
        behavior of an implication
        - subtract input information from True to become False
        - both LHS and RHS are incapable of subtracting information, therefore
        the operator is incorrect
    """
    model = Model()
    LHS = Proposition("LHS")
    RHS = Proposition("RHS")
    AB = Implies(LHS, RHS, world=World.AXIOM)
    model.add_knowledge(AB)
    model.add_data({LHS: TRUE, RHS: FALSE})
    model.train(direction=Direction.UPWARD, losses=[Loss.CONTRADICTION])
    model.print(params=True)
    bias = AB.params("bias")
    bounds = AB.state()
    assert bias <= 1e-5, f"expected bias to be downweighted <= 0., received {bias}"
    assert bounds is TRUE, f"expected operator bounds to remain True, received {bounds}"


def test_false_operator_1():
    """decrease weights for contradictory facts

    given a False Implies, for (False, Unknown) inputs
    expects LHS innput to be downweighted
    """
    model = Model()
    activation = {"alpha": 1 - 1e-5}
    LHS = Proposition("LHS")
    RHS = Proposition("RHS")
    AB = Implies(LHS, RHS, world=World.CLOSED, activation=activation)
    model.add_knowledge(AB)
    model.add_data({LHS: FALSE, RHS: UNKNOWN})
    model.train(direction=Direction.UPWARD, losses=[Loss.CONTRADICTION])
    weights = AB.params("weights")
    bounds = AB.state()
    assert (
        weights[0] <= 1 / 2
    ), f"expected input LHS to be downweighted <= .5, received {weights[0]}"
    assert bounds is FALSE, f"expected bounds AB to remain False, received {bounds}"


def test_false_operator_2():
    """decrease weights for contradictory facts

    given a False Implies, for (False, True) inputs
    expects both inputs to be downweighted
    under this configuration, the parent operand defaults to the expected
        behavior of an implication
        - add input information from False to become True
        - both LHS and RHS add information, and are therefore incorrect
    """
    model = Model()
    LHS = Proposition("LHS")
    RHS = Proposition("RHS")
    AB = Implies(LHS, RHS, world=World.CLOSED)
    model.add_knowledge(AB)
    model.add_data({LHS: FALSE, RHS: TRUE})
    model.train(direction=Direction.UPWARD, losses=[Loss.CONTRADICTION])

    weights = AB.params("weights")
    bounds = AB.state()
    assert all(
        [w <= 1 / 2 for w in weights]
    ), f"expected both inputs to be downweighted <= .5, received {weights}"
    assert bounds is FALSE, f"expected bounds AB to remain False, received {bounds}"


def test_false_operator_3():
    """decrease weights for contradictory facts

    given a False Implies, for (True, True) inputs
    The LHS agrees with the operator that the operator bounds could be False,
    both the operator and the LHS therefore conclude that the RHS is the source
    of the contradiction - downweight RHS
    """
    model = Model()
    activation = {"alpha": 1 - 1e-5}
    LHS = Proposition("LHS")
    RHS = Proposition("RHS")
    AB = Implies(LHS, RHS, world=World.FALSE, activation=activation)
    model.add_knowledge(AB)
    model.add_data({LHS: TRUE, RHS: TRUE})
    model.train(direction=Direction.UPWARD, losses=[Loss.CONTRADICTION])
    weights = AB.params("weights")
    bounds = AB.state()
    assert (
        weights[1] <= 1 / 2
    ), f"expected input RHS to be downweighted <= .5, received {weights[1]}"
    assert bounds is FALSE, f"expected bounds AB to remain False, received {bounds}"


if __name__ == "__main__":
    test_true_operator()
    test_false_operator_1()
    test_false_operator_2()
    test_false_operator_3()
    print("success")
