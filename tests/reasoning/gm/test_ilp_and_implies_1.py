##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, World, And, Variable, Direction, Implies, Fact, Join

TRUE = Fact.TRUE


def test_1():
    """
    And(isFather(x,z),isFather(z,y))
    Using expanded outer joins
    Closed World Assumption on isFather
    """

    GT_i = {("a", "b", "c"): Fact.TRUE}
    GT_o = dict(
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
    GT_op = dict(
        [
            (("a", "b", "b"), Fact.TRUE),
            (("a", "b", "c"), Fact.TRUE),
            (("b", "c", "b"), Fact.TRUE),
            (("b", "a", "b"), Fact.TRUE),
            (("b", "c", "c"), Fact.TRUE),
            (("b", "b", "c"), Fact.TRUE),
            (("a", "a", "b"), Fact.TRUE),
        ]
    )

    for join, GT in zip([Join.INNER, Join.OUTER], [GT_i, GT_o, GT_op]):

        # background data (features)
        B = ["isFather", [("a", "b"), ("b", "c")]]

        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        vars1 = (x, z)
        vars2 = (z, y)

        model = Model()
        b = model.add_predicates(2, B[0], world=World.FALSE)
        model.add_data({b: {pair: TRUE for pair in B[1]}})
        rule = And(b(*vars1), b(*vars2), join=join)
        model.add_knowledge(rule)

        model.infer(direction=Direction.UPWARD)

        assert all([rule.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
        assert len(rule.groundings) == len(GT), "FAILED 😔"


def test_2():
    """
    Implies(isGrandFather(x,y),And(isFather(x,z),isFather(z,y))
    Using expanded outer joins
    Closed World Assumption on isFather and isGrandFather
    """
    GT_i = {("a", "c", "b"): Fact.TRUE}
    GT_o = dict(
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

    for join, GT in zip([Join.INNER, Join.OUTER], [GT_i, GT_o]):
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
        b, p = model.add_predicates(2, B[0], P1[0], world=World.FALSE)
        model.add_data(
            {
                b: {pair: TRUE for pair in B[1]},
                p: {pair: TRUE for pair in P1[1]},
            }
        )
        rule = Implies(p(x, y), And(b(*vars1), b(*vars2), join=join), join=join)
        model.add_knowledge(rule)

        model.infer(direction=Direction.UPWARD)

        assert all([rule.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
        assert len(rule.groundings) == len(GT), "FAILED 😔"


if __name__ == "__main__":
    test_1()
    test_2()
    print("success")
