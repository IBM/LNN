##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, And, Model, Fact
import numpy as np

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN
CONTRADICTION = Fact.CONTRADICTION


def test_upward():
    """standard upward n-input conjunction boolean truth table"""

    n = 1000
    props = list()
    for i in range(n):
        props.append(Proposition("p" + str(i)))
    And_n = And(*props)

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
        facts[props[i]] = row[i]

    model = Model()
    model.add_knowledge(And_n)
    model.add_data(facts)
    And_n.upward()

    # evaluate the conjunction
    prediction = And_n.state()
    assert prediction is GT, f"And({row}) expected {GT}, received {prediction}"
    model.flush()

    # Test the case of all True
    for i in range(n):
        facts[props[i]] = TRUE

    model = Model()
    model.add_knowledge(And_n)
    model.add_data(facts)
    And_n.upward()

    # evaluate the conjunction
    prediction = And_n.state()
    assert prediction is TRUE, f"And({row}) expected {GT}, received {prediction}"
    model.flush()

    # Test the case of all False
    for i in range(n):
        facts[props[i]] = FALSE

    model = Model()
    model.add_knowledge(And_n)
    model.add_data(facts)
    And_n.upward()

    # evaluate the conjunction
    prediction = And_n.state()
    assert prediction is FALSE, f"And({row}) expected {GT}, received {prediction}"
    model.flush()


def test_downward():
    # define model rules
    n = 1000
    props = list()
    model = Model()
    for i in range(n):
        props.append(Proposition("P" + str(i)))
    And_n = And(*props)
    model.add_knowledge(And_n)

    # define model facts
    model.add_data(
        {
            And_n: TRUE,
        }
    )
    And_n.downward()

    # evaluate
    for i in range(n):
        prediction = props[i].state()
        assert prediction is TRUE, f"Expected input to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({props[0]: FALSE, And_n: TRUE})
    And_n.downward()

    # evaluate
    prediction = props[0].state()
    assert (
        prediction is CONTRADICTION
    ), f"Expected input P0 to be Contradiction, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert prediction is TRUE, f"Expected input to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({props[0]: TRUE, And_n: FALSE})
    And_n.downward()

    # evaluate
    prediction = props[0].state()
    assert prediction is TRUE, f"Expected input p0 to be True, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert (
            prediction is UNKNOWN
        ), f"Expected input to be Unknown, received {prediction}"
    model.flush()

    # Test the case of all True except one
    model.add_data({props[i]: TRUE for i in range(1, n)})
    model.add_data({And_n: FALSE})
    And_n.downward()

    # evaluate
    prediction = props[0].state()
    assert prediction is FALSE, f"Expected input P0 to be False, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert prediction is TRUE, f"Expected input to be True, received {prediction}"
    model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
    print("success")
