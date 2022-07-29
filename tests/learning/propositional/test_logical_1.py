##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Proposition, And, Or, TRUE, FALSE, plot_loss, plot_params


def test_1():
    model = Model()
    model["A"] = Proposition()
    model["B"] = Proposition()
    AB = model["A&B"] = And(model["A"], model["B"])
    model["A|B"] = Or(model["A"], model["B"])

    model.add_facts({"A": TRUE})
    model.add_facts({"B": FALSE})
    model.add_labels({AB.name: TRUE})

    parameter_history = {"weights": AB, "bias": AB}
    losses = ["logical", "supervised"]
    total_loss, _ = model.train(
        losses=losses, learning_rate=0.1, parameter_history=parameter_history
    )

    return total_loss, losses, model


if __name__ == "__main__":
    total_loss, losses, model = test_1()
    model.print(params=True)
    plot_loss(total_loss, losses)
    plot_params(model)
    print("success")
