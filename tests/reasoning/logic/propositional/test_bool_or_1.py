##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (
    Proposition,
    Or,
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
    """standard upward 2-input disjunction boolean truth table"""

    TT = truth_table(2)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Or(A, B)
    formulae = [AB]

    for row in TT:
        # get ground truth
        GT = np.logical_or(*list(map(fact_to_bool, row)))

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
    AB = Or(A, B)
    model.add_knowledge(AB)

    # define model facts
    model.add_data({A: FALSE, AB: TRUE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    prediction = B.state()
    assert prediction is TRUE, f"Expected input B to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({AB: FALSE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    prediction = B.state()
    assert prediction is FALSE, f"Expected input B to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: TRUE, AB: TRUE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert prediction is TRUE, f"Expected input A to be TRUE, received {prediction}"
    prediction = B.state()
    assert (
        prediction is UNKNOWN
    ), f"Expected input B to be UNKNOWN, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: TRUE, AB: FALSE})
    AB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is CONTRADICTION
    ), f"Expected input A to be CONTRADICTION, received {prediction}"
    prediction = B.state()
    assert prediction is FALSE, f"Expected input B to be FALSE, received {prediction}"


if __name__ == "__main__":
    test_upward()
    test_downward()
