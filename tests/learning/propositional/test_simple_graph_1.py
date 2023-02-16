##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Or, Proposition, World, Fact, Loss

TRUE = Fact.TRUE
FALSE = Fact.FALSE


def test_1():
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    _and = And(A, B, world=World.AXIOM)
    _or = Or(A, B)
    model.add_knowledge(_and, _or)
    model.add_data({A: TRUE, B: FALSE})
    losses = [Loss.CONTRADICTION]
    model.train(losses=losses)
    model.print(params=True)

    weights_and = _and.neuron.weights
    weights_or = _or.neuron.weights
    bias_or = _or.neuron.bias
    bounds = B.state()
    assert _or.is_unweighted(), (
        "expected A|B to be unweighted, received " f"w: {weights_or}, b: {bias_or}"
    )

    assert (
        weights_and[1] <= 1e-3
    ), f"expected input B in A&B to be down-weighted, received {weights_and}"
    assert bounds is FALSE, f"expected bounds to remain False, received {bounds}"


if __name__ == "__main__":
    test_1()
