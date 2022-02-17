##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Or, Proposition, World, TRUE, FALSE


def test_1():
    model = Model()
    model["A"] = Proposition("A")
    model["B"] = Proposition("B")
    model["A&B"] = And(model["A"], model["B"], world=World.AXIOM)
    model["A|B"] = Or(model["A"], model["B"])
    model.add_facts({"A": TRUE})
    model.add_facts({"B": FALSE})
    model.train(epochs=11, losses={"contradiction": 1})
    model.print(params=True)

    weights_and = model["A&B"].params("weights")[1]
    weights_or, bias_or = model["A|B"].params("weights", "bias")
    bounds = model["B"].state()
    assert model["A|B"].is_unweighted(), (
        "expected A|B to be unweighted, received " f"w: {weights_or}, b: {bias_or}"
    )
    assert (
        weights_and <= 1 / 2
    ), f"expected input B in A&B to be downweighted, received {weights_and}"
    assert bounds is FALSE, f"expected bounds to remain False, received {bounds}"


if __name__ == "__main__":
    test_1()
    print("success")
