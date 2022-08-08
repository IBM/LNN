##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Fact, Exists, Variable, Join


def test_1():
    join = Join.OUTER
    x = Variable("x")
    y = Variable("y")

    model = Model()

    director, starring = model.add_predicates(2, "director", "starring")

    facts = {
        director: {("William_Shatner", "The_captains"): Fact.TRUE},
        starring: {
            ("William_Shatner", "The_captains"): Fact.TRUE,
            ("Patrick_Stewart", "The_captains"): Fact.TRUE,
        },
    }
    model.add_data(facts)

    model.set_query(
        Exists(
            x,
            And(
                director(x, y, bind={x: "William_Shatner"}),
                starring(x, y),
                join=join,
            ),
        )
    )
    model.infer()
    predictions = model.query.true_groundings
    for p in predictions:
        assert p[0] == "William_Shatner"


def test_2():
    join = Join.OUTER
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    model = Model()

    director, starring = model.add_predicates(2, "director", "starring")

    model.add_data(
        {
            director: {("William_Shatner", "The_captains"): Fact.TRUE},
            starring: {
                ("William_Shatner", "The_captains"): Fact.TRUE,
                ("Patrick_Stewart", "The_captains"): Fact.TRUE,
            },
        }
    )

    model.set_query(
        Exists(
            z,
            And(
                director(x, y, bind={x: "William_Shatner"}),
                starring(z, y),
                join=join,
            ),
        )
    )

    model.infer()
    predictions = model.query.true_groundings
    for p in predictions:
        assert p[-1] in {"William_Shatner", "Patrick_Stewart"}


if __name__ == "__main__":
    test_1()
    test_2()

    print("success")
