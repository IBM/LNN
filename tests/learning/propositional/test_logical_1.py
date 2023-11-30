##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Proposition, And, Or, Fact, Loss

TRUE = Fact.TRUE
FALSE = Fact.FALSE


def test_1():
    model = Model()
    A = Proposition("A", model=model)
    B = Proposition("B", model=model)
    Or(A, B)
    AB = And(A, B)
    model.add_data({A: TRUE, B: FALSE})
    model.add_labels({AB: TRUE})

    parameter_history = {"weights": AB, "bias": AB}
    losses = [Loss.LOGICAL, Loss.SUPERVISED]
    total_loss, _ = model.train(
        losses=losses, learning_rate=0.1, parameter_history=parameter_history
    )
    model.print(params=True)
    return total_loss, losses, model


if __name__ == "__main__":
    total_loss, losses, model = test_1()
