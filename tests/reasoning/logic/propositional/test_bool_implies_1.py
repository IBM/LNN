##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (
    Proposition,
    Implies,
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
    r"""standard upward 2-input implication boolean truth table."""

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Implies(A, B)
    formulae = [AB]

    for row in truth_table(2):
        # get ground truth
        GT = np.logical_or(np.logical_not(fact_to_bool(row[0])), fact_to_bool(row[1]))

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
    AB = Implies(A, B)
    model.add_knowledge(AB)

    # define model facts
    model.add_data({A: TRUE, AB: TRUE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"
    prediction = B.state()
    assert prediction is TRUE, f"Expected input B to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: TRUE, AB: FALSE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"
    prediction = B.state()
    assert prediction is FALSE, f"Expected input B to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: FALSE, AB: TRUE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    prediction = B.state()
    assert (
        prediction is UNKNOWN
    ), f"Expected input B to be Unknown, received {prediction}"

    # define model facts
    model.add_data({A: FALSE, AB: FALSE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is CONTRADICTION
    ), f"Expected input A to be Contradiction, received {prediction}"
    prediction = B.state()
    assert prediction is FALSE, f"Expected input B to be False, received {prediction}"


if __name__ == "__main__":
    test_upward()
    test_downward()
