##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, Or, Model, Fact
import numpy as np

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN
CONTRADICTION = Fact.CONTRADICTION


def test_upward():
    """standard upward n-input disjunction boolean truth table"""

    n = 1000
    props = list()
    for i in range(n):
        props.append(Proposition("p" + str(i)))
    Or_n = Or(*props)

    # Test the case of 1 True
    facts = {props[i]: FALSE for i in range(n)}
    facts[props[np.random.randint(n)]] = TRUE

    # get ground truth
    GT = TRUE

    model = Model()
    model.add_knowledge(Or_n)
    model.add_data(facts)
    Or_n.upward()

    # evaluate the conjunction
    prediction = Or_n.state()
    assert (
        prediction is GT
    ), f"Or{tuple(facts.values())} expected {GT}, received {prediction}"
    model.flush()

    # Test the case of all True
    for i in range(n):
        facts[props[i]] = TRUE

    model = Model()
    model.add_knowledge(Or_n)
    model.add_data(facts)
    Or_n.upward()

    # evaluate the conjunction
    prediction = Or_n.state()
    assert (
        prediction is TRUE
    ), f"Or{tuple(facts.values())} expected {GT}, received {prediction}"
    model.flush()

    # Test the case of all False
    for i in range(n):
        facts[props[i]] = FALSE

    model = Model()
    model.add_knowledge(Or_n)
    model.add_data(facts)
    Or_n.upward()

    # evaluate the conjunction
    prediction = Or_n.state()
    assert (
        prediction is FALSE
    ), f"Or{tuple(facts.values())} expected {GT}, received {prediction}"
    model.flush()


def test_downward():
    # define model rules
    n = 1000
    props = list()
    for i in range(n):
        props.append(Proposition("p" + str(i)))

    model = Model()
    Or_n = Or(*props)
    model.add_knowledge(Or_n)

    # define model facts
    model.add_data({props[0]: FALSE, Or_n: FALSE})
    Or_n.downward()

    # evaluate
    prediction = props[0].state()
    assert prediction is FALSE, f"Expected input p0 to be False, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert prediction is FALSE, f"Expected input to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({props[0]: TRUE, Or_n: TRUE})
    Or_n.downward()

    # evaluate
    prediction = props[0].state()
    assert prediction is TRUE, f"Expected input p0 to be True, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert (
            prediction is UNKNOWN
        ), f"Expected input to be Unknown, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({props[0]: FALSE, Or_n: TRUE})
    Or_n.downward()

    # evaluate
    prediction = props[0].state()
    assert prediction is FALSE, f"Expected input A to be False, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert (
            prediction is UNKNOWN
        ), f"Expected input to be Unknown, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({props[0]: TRUE, Or_n: FALSE})
    Or_n.downward()

    # evaluate
    prediction = props[0].state()
    assert (
        prediction is CONTRADICTION
    ), f"Expected input A to be Contradiction, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert prediction is FALSE, f"Expected input to be False, received {prediction}"
    model.flush()

    # test model facts for All False except one
    model.add_data({props[i]: FALSE for i in range(1, n)})
    model.add_data({Or_n: TRUE})
    Or_n.downward()

    # evaluate
    prediction = props[0].state()
    assert prediction is TRUE, f"Expected input A to be True, received {prediction}"

    for i in range(1, n):
        prediction = props[i].state()
        assert prediction is FALSE, f"Expected input to be False, received {prediction}"
    model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
