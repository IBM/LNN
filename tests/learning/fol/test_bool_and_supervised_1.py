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
    AB = And(p1(x), p2(x))
    model.add_knowledge(AB)
    model.add_labels(
        {
            AB: {
                "0": TRUE,
                "1": UNKNOWN,
                "2": TRUE,
                "3": FALSE,
            }
        }
    )
    parameter_history = {"weights": True, "bias": True}
    losses = [Loss.LOGICAL, Loss.SUPERVISED]
    total_loss, _ = model.train(
        direction=Direction.UPWARD, losses=losses, parameter_history=parameter_history
    )
    model.print(params=True)
    predictions = p1.state().values()
    assert all([fact is TRUE for fact in predictions]), (
        "expected AB Facts to all be TRUE, received bounds "
        f"{[p for p in predictions]}"
    )

    return model, total_loss, losses


if __name__ == "__main__":
    my_model, total_loss, losses = test()
