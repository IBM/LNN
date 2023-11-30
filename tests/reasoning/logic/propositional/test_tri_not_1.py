##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, Not, Model, Fact, fact_to_bool


TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test_upward():
    """standard upward ,2-input conjunction three-valued truth table"""

    TT = [
        # A, Not(A)
        [TRUE, FALSE],
        [FALSE, TRUE],
        [UNKNOWN, UNKNOWN],
    ]

    # define the rules
    model = Model()
    A = Proposition("A", model=model)
    NotA = Not(A)

    for row in TT:
        # get ground truth
        GT = fact_to_bool(row[1])

        # load model and reason over facts
        facts = {A: row[0]}
        model.add_data(facts)
        NotA.upward()

        # evaluate the conjunction
        prediction = NotA.state(to_bool=True)
        assert (
            prediction is GT
        ), f"Not({row[0]}, {row[1]}) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    # define model rules
    model = Model()
    A = Proposition("A", model=model)
    NotA = Not(A)

    # define model facts
    model.add_data({NotA: TRUE})
    NotA.downward()

    # evaluate
    prediction = A.state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({NotA: FALSE})
    NotA.downward()

    # evaluate
    prediction = A.state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({NotA: UNKNOWN})
    NotA.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is UNKNOWN
    ), f"Expected input A to be Unknown, received {prediction}"
    model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
