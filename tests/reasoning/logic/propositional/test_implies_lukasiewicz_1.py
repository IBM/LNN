##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Propositions, Implies, Model
import numpy as np


def test():
    """Unittest for upward 2-input real-value implication"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the model
    model = Model()
    A, B = Propositions("A", "B", model=model)
    AB = Implies(A, B)

    for row in range(samples):
        for col in range(samples):
            # inputs
            a, b = x[row][col], y[row][col]

            # ground truth
            GT = 1 - a + b

            # facts per model
            facts = {A: (a, a), B: (b, b)}

            # load data into the model
            model.add_data(facts)

            # evaluate the implication
            AB.upward()

            # test the prediction
            prediction = AB.get_data()[0]
            assert (
                prediction - GT <= 1e-7
            ), f"And({a}, {b}) expected {GT}, received {prediction}"
            model.flush()


if __name__ == "__main__":
    test()
