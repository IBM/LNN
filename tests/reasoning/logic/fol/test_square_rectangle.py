##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, Variable, Exists, Implies, ForAll, Model, Fact, World


def test_1():
    """Simple theorem proving example
    Square(c)
    Square(k)

    """

    x = Variable("x")
    square = Predicate(name="square")
    rectangle = Predicate(name="rectangle")
    foursides = Predicate(name="foursides")
    square_rect = ForAll(
        x,
        Implies(square(x), rectangle(x), name="square-rect"),
        name="all-square-rect",
        world=World.AXIOM,
    )
    rect_foursides = ForAll(
        x,
        Implies(rectangle(x), foursides(x), name="rect-foursides"),
        name="all-rect-foursides",
        world=World.AXIOM,
    )
    query = Exists(x, foursides(x), name="foursided_objects")

    model = Model()
    model.add_formulae(square, rectangle, square_rect, rect_foursides, query)
    model.add_facts({"square": {"c": Fact.TRUE, "k": Fact.TRUE}})

    steps, facts_inferred = model.infer()

    # Currently finishes in 4 inference steps
    assert steps == 4, "FAILED 😔"

    GT_o = dict([(("c"), Fact.TRUE), (("k"), Fact.TRUE)])

    assert all(
        [model["foursided_objects"].state(groundings=g) is GT_o[g] for g in GT_o]
    ), "FAILED 😔"


if __name__ == "__main__":
    test_1()
    print("success")
