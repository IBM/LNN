##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import numpy as np
from lnn import Proposition, Or, Model, Fact


def test_upward(output=False):
    """standard upward n-input disjunction boolean truth table"""

    n = 1000
    propositions = list()
    for i in range(0, n):
        propositions.append(Proposition("p" + str(i)))
    formulae = [Or(*propositions, name="Or_n")]

    dat = np.linspace(0.0001, 0.001, n)
    GT = dat[0]
    for i in range(1, n):
        GT = min(1, GT + dat[i])

    if output:
        print("Ground truth:", GT)

    # load model and reason over facts
    facts = {}
    for i in range(0, n):
        facts["p" + str(i)] = (dat[i], dat[i])

    model = Model()
    model.add_formulae(*formulae)
    model.add_facts(facts)
    model["Or_n"].upward()

    # evaluate the conjunction
    prediction = model["Or_n"].get_facts()
    assert prediction[0] == prediction[1], (
        "Lower and upper bounds are not the same, "
        + f"got {prediction[0]}, {prediction[1]}"
    )
    assert round(prediction[0].item(), 4) == round(
        GT, 4
    ), f"Or(...) failed, expected {GT}, received {prediction[0].item()}"
    model.flush()


def test_downward(output=False):
    n = 1000
    propositions = list()
    for i in range(0, n):
        propositions.append(Proposition("p" + str(i)))
    formulae = [Or(*propositions, name="Or_n")]

    dat = np.linspace(0.0, 0.0, n)
    GT = dat[0]
    for i in range(1, n):
        GT = min(1, GT + dat[i])

    if output:
        print("Ground truth", GT)

    # load model and reason over facts
    facts = {}
    for i in range(0, n):
        facts["p" + str(i)] = (dat[i], dat[i])

    model = Model()
    model.add_formulae(*formulae)
    model.add_facts(facts)
    model["Or_n"].upward()

    # now make one of the inputs unknown
    p0 = model["p0"].get_facts()
    model.add_facts({"p0": Fact.UNKNOWN})
    model.downward()

    # evaluate
    prediction = model["p0"].get_facts()
    assert prediction[0] == prediction[1], (
        "Lower and upper bounds are not the same, "
        + f"got {prediction[0]}, {prediction[1]}"
    )
    assert round(prediction[0].item(), 4) == round(
        p0[0].item(), 4
    ), f"Or(...) failed, expected {GT}, received {prediction[0].item()}"

    # now make the OR True
    dat = np.linspace(0.0, 1.0, n)

    # load model and reason over facts
    facts = {}
    for i in range(0, n):
        facts["p" + str(i)] = (dat[i], dat[i])

    model = Model()
    model.add_formulae(*formulae)
    model.add_facts(facts)
    model["Or_n"].upward()

    # now make one of the inputs unknown
    p0 = model["p0"].get_facts()
    model.add_facts({"p0": Fact.UNKNOWN})
    model.downward()

    # evaluate
    prediction = model["p0"].get_facts()
    assert (
        prediction[0].item() == 0 and prediction[1].item() == 1
    ), "p0 should be UNKNOWN"


if __name__ == "__main__":
    test_upward(output=True)
    test_downward(output=True)
    print("success")
