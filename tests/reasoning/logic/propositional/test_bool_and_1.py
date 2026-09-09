##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (
    Proposition,
    And,
    Model,
    Fact,
    truth_table,
    fact_to_bool,
    bool_to_fact,
)
import numpy as np

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN
CONTRADICTION = Fact.CONTRADICTION


def test_upward():
    r"""Standard upward, 2-input conjunction boolean truth table."""

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B)
    formulae = [AB]

    for row in truth_table(2):
        # get ground truth
        GT = np.logical_and(*list(map(fact_to_bool, row)))

        # load model and reason over facts
        facts = {A: row[0], B: row[1]}
        model = Model()
        model.add_knowledge(*formulae)
        model.add_data(facts)
        AB.upward()

        # evaluate the conjunction
        prediction = AB.state()
        assert prediction is bool_to_fact(
            GT
        ), f"And({row[0]}, {row[1]}) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    # define model rules
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B)
    model.add_knowledge(AB)

    # define model facts
    model.add_data(
        {
            A: TRUE,
            AB: FALSE,
        }
    )
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is TRUE, f"Expected input A to be TRUE, received {prediction}"
    prediction = B.state()
    assert prediction is FALSE, f"Expected input B to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({AB: TRUE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is TRUE, f"Expected input A to be TRUE, received {prediction}"
    prediction = B.state()
    assert prediction is TRUE, f"Expected input B to be TRUE, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: FALSE, AB: FALSE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    prediction = B.state()
    assert (
        prediction is UNKNOWN
    ), f"Expected input B to be Unknown, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: FALSE, AB: TRUE})
    AB.downward()

    # evaluate
    assert (
        A.state() is CONTRADICTION
    ), f"Expected input B to be Contradiction, received {prediction}"
    assert (
        B.state() is TRUE
    ), f"Expected input B to be Contradiction, received {prediction}"
    model.flush()

    # define model facts
    AB.downward()

    # evaluate
    assert (
        A.state() is UNKNOWN
    ), f"Expected input B to be Unknown, received {prediction}"
    assert (
        B.state() is UNKNOWN
    ), f"Expected input B to be Unknown, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: TRUE})
    AB.downward()

    # evaluate
    assert A.state() is TRUE, f"Expected input B to be Unknown, received {prediction}"
    assert (
        B.state() is UNKNOWN
    ), f"Expected input B to be Unknown, received {prediction}"
    model.flush()

    # define model facts
    model.add_data(
        {
            A: TRUE,
            B: FALSE,
        }
    )
    AB.downward()

    # evaluate
    assert A.state() is TRUE, f"Expected input B to be Unknown, received {prediction}"
    assert B.state() is FALSE, f"Expected input B to be False, received {prediction}"
    model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
