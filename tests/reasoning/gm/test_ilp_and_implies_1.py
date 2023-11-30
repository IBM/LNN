##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, World, And, Variable, Direction, Implies, Fact, Predicates

TRUE = Fact.TRUE


def test_1():
    """
    And(isFather(x,z),isFather(z,y))
    Using expanded outer joins
    Closed World Assumption on isFather
    """

    GT = dict(
        [
            (("a", "b", "b"), Fact.FALSE),
            (("a", "b", "c"), Fact.TRUE),
            (("b", "c", "b"), Fact.FALSE),
            (("b", "a", "b"), Fact.FALSE),
            (("b", "c", "c"), Fact.FALSE),
            (("b", "b", "c"), Fact.FALSE),
            (("a", "a", "b"), Fact.FALSE),
        ]
    )

    # background data (features)
    B = ["isFather", [("a", "b"), ("b", "c")]]

    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    vars1 = (x, z)
    vars2 = (z, y)

    model = Model()
    b = Predicates(B[0], arity=2, world=World.FALSE, model=model)
    model.add_data({b: {pair: TRUE for pair in B[1]}})
    rule = And(b(*vars1), b(*vars2))

    model.infer(direction=Direction.UPWARD)

    assert all([rule.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
    assert len(rule.groundings) == len(GT), "FAILED 😔"


def test_2():
    """
    Implies(isGrandFather(x,y),And(isFather(x,z),isFather(z,y))
    Using expanded outer joins
    Closed World Assumption on isFather and isGrandFather
    """
    GT = dict(
        [
            (("a", "b", "b"), Fact.TRUE),
            (("a", "b", "a"), Fact.TRUE),
            (("a", "c", "a"), Fact.UNKNOWN),
            (("a", "c", "b"), Fact.TRUE),
            (("b", "b", "c"), Fact.TRUE),
            (("b", "c", "c"), Fact.TRUE),
            (("a", "c", "c"), Fact.UNKNOWN),
            (("b", "b", "a"), Fact.TRUE),
            (("b", "c", "b"), Fact.TRUE),
        ],
    )

    # background data (features)
    B = ["isFather", [("a", "b"), ("b", "c")]]

    # positive (target) labels for isGrandFather(x,y)
    P1 = ["isGrandFather", [("a", "c")]]

    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    vars1 = (x, z)
    vars2 = (z, y)

    model = Model()
    b, p = Predicates(B[0], P1[0], arity=2, world=World.FALSE, model=model)
    model.add_data(
        {
            b: {pair: TRUE for pair in B[1]},
            p: {pair: TRUE for pair in P1[1]},
        }
    )
    rule = Implies(p(x, y), And(b(*vars1), b(*vars2)))

    model.infer(direction=Direction.UPWARD)

    assert all([rule.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
    assert len(rule.groundings) == len(GT), "FAILED 😔"


if __name__ == "__main__":
    test_1()
    test_2()
