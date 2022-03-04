##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import numpy as np
from lnn import (
    Proposition,
    Or,
    Model,
    TRUE,
    FALSE,
    UNKNOWN,
    CONTRADICTION,
    truth_table,
    fact_to_bool,
    bool_to_fact,
)


def test_upward():
    """standard upward 2-input disjunction boolean truth table"""

    TT = truth_table(2)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Or(A, B, name="AB")
    formulae = [AB]

    for row in TT:
        # get ground truth
        GT = np.logical_or(*list(map(fact_to_bool, row)))

        # load model and reason over facts
        facts = {"A": row[0], "B": row[1]}
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model["AB"].upward()

        # evaluate the conjunction
        prediction = model["AB"].state()
        assert prediction is bool_to_fact(
            GT
        ), f"And({row[0]}, {row[1]}) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    # define model rules
    model = Model()
    model["A"] = Proposition("A")
    model["B"] = Proposition("B")
    model["AB"] = Or(model["A"], model["B"])

    # define model facts
    model.add_facts({"A": FALSE, "AB": TRUE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    prediction = model["B"].state()
    assert prediction is TRUE, f"Expected input B to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_facts({"AB": FALSE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    prediction = model["B"].state()
    assert prediction is FALSE, f"Expected input B to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_facts({"A": TRUE, "AB": TRUE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is TRUE, f"Expected input A to be TRUE, received {prediction}"
    prediction = model["B"].state()
    assert (
        prediction is UNKNOWN
    ), f"Expected input B to be UNKNOWN, received {prediction}"
    model.flush()

    # define model facts
    model.add_facts({"A": TRUE, "AB": FALSE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert (
        prediction is CONTRADICTION
    ), f"Expected input A to be CONTRADICTION, received {prediction}"
    prediction = model["B"].state()
    assert prediction is FALSE, f"Expected input B to be FALSE, received {prediction}"


if __name__ == "__main__":
    test_upward()
    test_downward()
    print("success")
