##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (
    Model,
    Proposition,
    And,
    Or,
    Implies,
    Fact,
    World,
    Loss,
)

TRUE = Fact.TRUE
FALSE = Fact.FALSE


def test_1():
    """test supervised, contradiction and logical loss for all neurons"""
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    _and = And(A, B, world=World.AXIOM, activation={"bias_learning": False})
    _or = Or(A, B, world=World.FALSE, activation={"bias_learning": False})
    _implies = Implies(A, B, world=World.FALSE)
    model.add_knowledge(_and, _or, _implies)
    model.add_data({A: FALSE, B: TRUE})
    model.add_labels(
        {
            _or: FALSE,
            _and: TRUE,
            _implies: FALSE,
        }
    )

    parameter_history = {"weights": True, "bias": True}
    losses = {Loss.CONTRADICTION: 1, Loss.SUPERVISED: 1, Loss.LOGICAL: 2e-2}
    total_loss, _ = model.train(
        epochs=3e2,
        learning_rate=5e-2,
        losses=losses,
        parameter_history=parameter_history,
    )

    model.print(params=True)

    state = _or.state()
    eps = 1e-3
    assert state is FALSE, f"expected A|B to be FALSE, received {state}"
    w = _or.neuron.weights
    assert (
        1 - eps <= w[0] and 0 <= w[1] <= eps
    ), f"expected A|B weights to be in [±1, <=.5], received {w}"

    state = _and.state()
    assert state is TRUE, f"expected A&B to be TRUE, received {state}"
    w = _and.neuron.weights
    assert (
        1 - eps <= w[1] and 0 <= w[0] <= eps
    ), f"expected A&B weights to be [<=.5, ±1], received {w}"

    state = _implies.state()
    assert state is FALSE, f"expected A->B to be FALSE, received {state}"
    w = _implies.neuron.weights
    assert all(
        (0 <= w) + (w <= eps)
    ), f"expected A->B weights to be in [0, .5], received {w}"


if __name__ == "__main__":
    test_1()
