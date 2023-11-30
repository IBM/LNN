##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Fact, Exists, Variable, Predicates


def test_1():
    x = Variable("x")
    y = Variable("y")

    model = Model()

    director, starring = Predicates("director", "starring", arity=2, model=model)

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
                director("William_Shatner", y),
                starring(x, y),
            ),
        )
    )
    model.infer()
    predictions = model.query.true_groundings
    for p in predictions:
        assert p[0] == "The_captains"


def test_2():
    y = Variable("y")
    z = Variable("z")

    model = Model()

    director, starring = Predicates("director", "starring", arity=2, model=model)

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
                director("William_Shatner", y),
                starring(z, y),
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
