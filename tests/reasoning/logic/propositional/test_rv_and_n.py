##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Propositions, And, Model, Fact
import numpy as np


def test_upward():
    """standard upward n-input conjunction boolean truth table"""

    # define the model
    model = Model()
    n = 1000
    props = Propositions(*["p" + str(i) for i in range(n)], model=model)
    And_n = And(*props)

    # ground truth facts
    dat = np.linspace(0.001, 1.0, n)
    GT = dat[0]
    for i in range(1, n):
        GT = max(0, GT + dat[i] - 1)
    print("Ground truth", GT)

    # reason over facts
    facts = {}
    for i in range(0, n):
        facts[props[i]] = (dat[i], dat[i])

    model.add_data(facts)
    And_n.upward()

    # evaluate the conjunction
    prediction = And_n.get_data()
    assert prediction[0] == prediction[1], (
        "Lower and upper bounds are not the same, "
        + f"got {prediction[0]}, {prediction[1]}"
    )
    assert round(prediction[0].item(), 4) == round(
        GT, 4
    ), f"And(...) failed, expected {GT}, received {prediction[0].item()}"
    model.flush()


def test_downward():
    # define the model
    model = Model()
    n = 1000
    props = Propositions(*["p" + str(i) for i in range(n)], model=model)
    And_n = And(*props)

    # ground truth
    dat = np.linspace(1.0, 1.0, n)
    GT = dat[0]
    for i in range(1, n):
        GT = max(0, GT + dat[i] - 1)
    print("Ground truth", GT)

    # reason over facts
    facts = {}
    for i in range(0, n):
        facts[props[i]] = (dat[i], dat[i])

    model.add_data(facts)
    And_n.upward()

    # now make one of the inputs unknown
    p0 = props[0].get_data()
    model.add_data({props[0]: Fact.UNKNOWN})
    model.downward()

    # evaluate the conjunction
    prediction = props[0].get_data()
    assert prediction[0] == prediction[1], (
        "Lower and upper bounds are not the same, "
        + f"got {prediction[0]}, {prediction[1]}"
    )
    assert round(prediction[0].item(), 4) == round(
        p0[0].item(), 4
    ), f"And(...) failed, expected {GT}, received {prediction[0].item()}"

    # now make the AND false
    dat = np.linspace(0.0, 1.0, n)

    # reason over facts
    facts = {}
    for i in range(0, n):
        facts[props[i]] = (dat[i], dat[i])

    model.add_data(facts)
    And_n.upward()

    # now make one of the inputs unknown
    model.add_data({props[0]: Fact.UNKNOWN})
    model.downward()

    # evaluate the conjunction
    prediction = props[0].get_data()
    assert (
        prediction[0].item() == 0 and prediction[1].item() == 1
    ), "p0 should be UNKNOWN"


if __name__ == "__main__":
    test_upward()
    test_downward()
