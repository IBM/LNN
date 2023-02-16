##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Variable, Fact, Forall, Exists, World

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test_1():
    """Quantifier with bounded variables, upward on predicate
    UNKNOWN result when not fully grounded
    """
    x = Variable("x")
    model = Model()
    A, S = model.add_predicates(1, "A", "S")
    All = Forall(x, A(x), world=World.OPEN)
    Some = Exists(x, S(x))

    model.add_knowledge(All, Some)
    model.add_data(
        {
            A: {"0": TRUE, "1": TRUE, "2": TRUE},
            S: {"0": FALSE, "1": FALSE, "2": FALSE},
        }
    )

    model.upward()
    model.print()
    predictions = [All.state(), Some.state()]
    assert predictions[0] is UNKNOWN, (
        f"Forall expected as UNKNOWN, received {predictions[0]}"
        "cannot learn to be TRUE unless fully grounded"
    )
    assert predictions[1] is UNKNOWN, (
        f"Exists expected as UNKNOWN, received {predictions[1]}"
        "cannot learn to be FALSE unless fully grounded"
    )


def test_2():
    """Quantifier with bounded variables, upward on predicate
    Single predicate truth updates quantifier truth
    """
    x = Variable("x")
    model = Model()
    A, S = model.add_predicates(1, "A", "S")
    All = Forall(x, A(x), world=World.OPEN)
    Some = Exists(x, S(x))
    model.add_knowledge(All, Some)
    model.add_data(
        {
            A: {"0": TRUE, "1": TRUE, "2": FALSE},
            S: {"0": FALSE, "1": FALSE, "2": TRUE},
        }
    )

    model.upward()
    assert Some.state() is TRUE, f"Forall expected as TRUE, received {Some.state()}"
    assert All.state() is FALSE, f"Exists expected as FALSE, received {All.state()}"


if __name__ == "__main__":
    test_1()
    test_2()
