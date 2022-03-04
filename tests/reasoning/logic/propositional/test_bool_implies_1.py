##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import numpy as np
from lnn import (
    Proposition,
    Implies,
    Model,
    CONTRADICTION,
    TRUE,
    FALSE,
    UNKNOWN,
    truth_table,
    fact_to_bool,
    bool_to_fact,
)


def test_upward():
    """standard upward 2-input implication boolean truth table"""

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Implies(A, B, name="AB")
    formulae = [AB]

    for row in truth_table(2):
        # get ground truth
        GT = np.logical_or(np.logical_not(fact_to_bool(row[0])), fact_to_bool(row[1]))

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
    model["AB"] = Implies(model["A"], model["B"])

    # define model facts
    model.add_facts({"A": TRUE, "AB": TRUE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"
    prediction = model["B"].state()
    assert prediction is TRUE, f"Expected input B to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_facts({"A": TRUE, "AB": FALSE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"
    prediction = model["B"].state()
    assert prediction is FALSE, f"Expected input B to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_facts({"A": FALSE, "AB": TRUE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"
    prediction = model["B"].state()
    assert (
        prediction is UNKNOWN
    ), f"Expected input B to be Unknown, received {prediction}"

    # define model facts
    model.add_facts({"A": FALSE, "AB": FALSE})
    model["AB"].downward()

    # evaluate
    prediction = model["A"].state()
    assert (
        prediction is CONTRADICTION
    ), f"Expected input A to be Contradiction, received {prediction}"
    prediction = model["B"].state()
    assert prediction is FALSE, f"Expected input B to be False, received {prediction}"


if __name__ == "__main__":
    test_upward()
    test_downward()
    print("success")
