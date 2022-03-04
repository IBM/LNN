##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (
    Model,
    Proposition,
    And,
    Or,
    Implies,
    TRUE,
    FALSE,
    World,
    plot_params,
    plot_loss,
)


def test_1():
    """test supervised, contradiction and logical loss for all neurons"""
    model = Model()
    A = model["A"] = Proposition()
    B = model["B"] = Proposition()
    model["A|B"] = Or(A, B, world=World.FALSE, neuron={"bias_learning": False})
    model["A&B"] = And(A, B, world=World.AXIOM, neuron={"bias_learning": False})
    model["A->B"] = Implies(A, B, world=World.FALSE)

    model.add_facts({"A": FALSE, "B": TRUE})
    model.add_labels(
        {
            "A|B": FALSE,
            "A&B": TRUE,
            "A->B": FALSE,
        }
    )

    parameter_history = {"weights": True, "bias": True}
    losses = {"contradiction": 1, "supervised": 1, "logical": 2e-2}
    total_loss, _ = model.train(
        epochs=3e2,
        learning_rate=5e-2,
        losses=losses,
        parameter_history=parameter_history,
    )

    model.print(params=True)
    plot_params(model)
    plot_loss(total_loss, losses)

    state = model["A|B"].state()
    eps = 1e-3
    assert state is FALSE, f"expected A|B to be FALSE, received {state}"
    w = model["A|B"].neuron.weights
    assert (
        1 - eps <= w[0] and 0 <= w[1] <= eps
    ), f"expected A|B weights to be in [±1, <=.5], received {w}"

    state = model["A&B"].state()
    assert state is TRUE, f"expected A&B to be TRUE, received {state}"
    w = model["A&B"].neuron.weights
    assert (
        1 - eps <= w[1] and 0 <= w[0] <= eps
    ), f"expected A&B weights to be [<=.5, ±1], received {w}"

    state = model["A->B"].state()
    assert state is FALSE, f"expected A->B to be FALSE, received {state}"
    w = model["A->B"].neuron.weights
    assert all(
        (0 <= w) + (w <= eps)
    ), f"expected A->B weights to be in [0, .5], received {w}"


if __name__ == "__main__":
    test_1()
    print("success")
