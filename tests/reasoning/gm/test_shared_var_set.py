##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, Variable, And, Join, Implies, ForAll, Model, Fact, World


def test_1():
    """Simple theorem proving example
    Unary predicates with overlapping variable set.
    """

    x = Variable("x")
    square = Predicate(name="square")
    rectangle = Predicate(name="rectangle")

    square_rect = ForAll(
        x,
        Implies(square(x), rectangle(x), name="square-rect", join=Join.OUTER),
        name="all-square-rect",
        join=Join.OUTER,
        world=World.AXIOM,
    )

    model = Model()
    model.add_formulae(square, rectangle, square_rect)
    model.add_facts({"square": {"c": Fact.TRUE, "k": Fact.TRUE}})

    model.upward()

    assert len(rectangle.groundings) == 2, "FAILED 😔"


def test_2():
    """
    Binary and unary predicates with an overlapping variable subset.
    :return:
    """

    x, y = map(Variable, ["x", "y"])
    model = Model()  # Instantiate a model.

    enemy = model["enemy"] = Predicate(arity=2, name="enemy")
    hostile = model["hostile"] = Predicate(name="hostile")

    model["america-enemies"] = ForAll(
        x,
        Implies(
            enemy(x, (y, "America")), hostile(x), name="enemy->hostile", join=Join.OUTER
        ),
        name="america-enemies",
        join=Join.OUTER,
        world=World.AXIOM,
    )

    # Add facts to model.
    model.add_facts({"enemy": {("Nono", "America"): Fact.TRUE}})

    model.upward()
    assert len(hostile.groundings) == 1, "FAILED 😔"


def test_3():
    """
    Tenary and binary predicates with an overlapping variable subset.
    :return:
    """

    x, y, z = map(Variable, ["x", "y", "z"])
    model = Model()  # Instantiate a model.

    f1 = Predicate(name="F1", arity=3)
    f2 = Predicate(name="F2", arity=2)

    rule = And(f1(x, y, z), f2(x, y), join=Join.OUTER)

    model.add_formulae(f1, f2, rule)
    model.add_facts({"F1": {("x1", "y1", "z1"): Fact.TRUE}})

    model.upward()

    assert len(f2.groundings) == 1, "FAILED 😔"


def test_4():
    """
    Tenary predicate (x,y,z) and 3 unary predicates (x),(y),(z) requiring
    product join and overlapping variable subset.
    :return:
    """

    x, y, z = map(Variable, ["x", "y", "z"])
    american = Predicate("american")
    hostile = Predicate("hostile")
    weapon = Predicate("weapon")
    sells = Predicate(arity=3, name="sells")

    model = Model()  # Instantiate a model.
    rule = And(american(x), weapon(y), hostile(z), sells(x, y, z), join=Join.OUTER)
    model.add_formulae(american, hostile, weapon, rule)
    model.add_facts(
        {
            "american": {"West": Fact.TRUE},
            "hostile": {"Nono": Fact.TRUE},
            "weapon": {"m1": Fact.TRUE},
        }
    )

    model.upward()

    assert len(sells.groundings) == 1, "FAILED 😔"


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
    print("success")
