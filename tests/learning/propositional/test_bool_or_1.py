##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Or, Proposition, TRUE, FALSE, UPWARD, CLOSED


def test_upward():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    """
    model = Model()
    model["A"] = Proposition("A")
    model["B"] = Proposition("B")
    model["AB"] = Or(
        model["A"],
        model["B"],
        world=CLOSED,
    )
    model.add_facts({"A": TRUE, "B": FALSE})
    model.train(direction=UPWARD, losses={"contradiction": 0.1})

    weights = model["AB"].params("weights")
    bounds = model["A"].state()
    assert (
        weights[0] <= 0.5
    ), f"expected input A to be downweighted <= 0., received {weights[0]}"
    assert bounds is TRUE, f"expected bounds to remain True, received {bounds}"


if __name__ == "__main__":
    test_upward()
    print("success")
