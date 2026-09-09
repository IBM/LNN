##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (
    Model,
    And,
    Variable,
    Fact,
    World,
    Loss,
    Direction,
)

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test():
    model = Model()

    p1, p2 = model.add_predicates(1, "P1", "P2")

    model.add_data(
        {
            p1: {"0": TRUE, "1": TRUE, "2": TRUE, "3": TRUE},
            p2: {
                "0": TRUE,
                "1": UNKNOWN,
                "2": FALSE,
                "3": FALSE,
            },
        }
    )

    x = Variable("x")
    AB = And(p1(x), p2(x), world=World.AXIOM)
    model.add_knowledge(AB)
    parameter_history = {"weights": True}
    losses = [Loss.CONTRADICTION]
    total_loss, _ = model.train(
        direction=Direction.UPWARD,
        learning_rate=1e-2,
        losses=losses,
        parameter_history=parameter_history,
    )
    model.print(params=True)
    predictions = p1.state().values()
    assert all([fact is TRUE for fact in predictions]), (
        "expected AB Facts to all be TRUE, received bounds "
        f"{[p for p in predictions]}"
    )
    assert (
        AB.neuron.weights[0] > 0.95
    ), f"expected input p1 to remain high, received {AB.neuron.weights[0]}"
    assert (
        AB.neuron.weights[1] <= 1e-5
    ), f"expected input p2 to be down-weighted, received {AB.neuron.weights[1]}"
    return model, total_loss, losses


if __name__ == "__main__":
    model, total_loss, losses = test()
