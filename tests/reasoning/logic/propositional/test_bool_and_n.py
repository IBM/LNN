##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import numpy as np
from lnn import Proposition, And, Model, TRUE, FALSE, UNKNOWN, CONTRADICTION


def test_upward():
    """standard upward n-input conjunction boolean truth table"""

    n = 1000
    propositions = list()
    for i in range(1, n):
        propositions.append(Proposition("p" + str(i)))
    formulae = [And(*propositions, name="And_n")]

    dat = np.random.random(n)
    row = list(map(lambda x: TRUE if x >= 0.5 else FALSE, dat))

    # get ground truth
    GT = TRUE
    for t in row:
        if t is FALSE:
            GT = FALSE

    # load model and reason over facts
    facts = {}
    for i in range(1, n):
        facts["p" + str(i)] = row[i]

    model = Model()
    model.add_formulae(*formulae)
    model.add_facts(facts)
    model["And_n"].upward()

    # evaluate the conjunction
    prediction = model["And_n"].state()
    assert prediction is GT, f"And({row}) expected {GT}, received {prediction}"
    model.flush()

    # Test the case of all True
    for i in range(1, n):
        facts["p" + str(i)] = TRUE

    model = Model()
    model.add_formulae(*formulae)
    model.add_facts(facts)
    model["And_n"].upward()

    # evaluate the conjunction
    prediction = model["And_n"].state()
    assert prediction is TRUE, f"And({row}) expected {GT}, received {prediction}"
    model.flush()

    # Test the case of all False
    for i in range(1, n):
        facts["p" + str(i)] = FALSE

    model = Model()
    model.add_formulae(*formulae)
    model.add_facts(facts)
    model["And_n"].upward()

    # evaluate the conjunction
    prediction = model["And_n"].state()
    assert prediction is FALSE, f"And({row}) expected {GT}, received {prediction}"
    model.flush()


def test_downward():
    # define model rules
    n = 1000
    propositions = list()
    for i in range(1, n):
        propositions.append(Proposition("p" + str(i)))

    model = Model()
    model["A"] = propositions[0]
    for i in range(1, n - 1):
        model["P" + str(i)] = propositions[i]
    model["And_n"] = And(*propositions)

    # define model facts
    model.add_facts(
        {
            "A": TRUE,
            "And_n": TRUE,
        }
    )
    model["And_n"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"

    for i in range(1, n - 1):
        prediction = model["P" + str(i)].state()
        assert prediction is TRUE, f"Expected input to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_facts({"A": FALSE, "And_n": TRUE})
    model["And_n"].downward()

    # evaluate
    prediction = model["A"].state()
    assert (
        prediction is CONTRADICTION
    ), f"Expected input A to be Contradiction, received {prediction}"

    for i in range(1, n - 1):
        prediction = model["P" + str(i)].state()
        assert prediction is TRUE, f"Expected input to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_facts({"A": TRUE, "And_n": FALSE})
    model["And_n"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"

    for i in range(1, n - 1):
        prediction = model["P" + str(i)].state()
        assert (
            prediction is UNKNOWN
        ), f"Expected input to be Unknown, received {prediction}"
    model.flush()

    # Test the case of all True except one
    model.add_facts({"P" + str(i): TRUE for i in range(1, n - 1)})
    model.add_facts({"And_n": FALSE})
    model["And_n"].downward()

    # evaluate
    prediction = model["A"].state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"

    for i in range(1, n - 1):
        prediction = model["P" + str(i)].state()
        assert prediction is TRUE, f"Expected input to be True, received {prediction}"
    model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
    print("success")
