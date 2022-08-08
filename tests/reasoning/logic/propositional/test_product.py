##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

# flake8: noqa: E501

from lnn import Proposition, And, Or, Implies, Model, NeuralActivation
import numpy as np
import random


def test_robust_and_2():
    """Unittest for robust (meaning lower_bd != upper_bd) upward 2-input real-value conjunction"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [AB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a_upper, b_upper = x[row][col], y[row][col]
            a_lower = random.uniform(0, a_upper)
            b_lower = random.uniform(0, b_upper)

            # ground truth
            GT_lower = a_lower * b_lower
            GT_upper = a_upper * b_upper

            # facts per model
            facts = {
                A: (a_lower, a_upper),
                B: (b_lower, b_upper),
            }  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AB.upward()

            # test the prediction
            prediction_lower = AB.get_data()[0]
            prediction_upper = AB.get_data()[1]
            assert prediction_lower - GT_lower <= 1e-5, (
                f"And({a_lower}, {b_lower}) expected {GT_lower}, "
                f"received {prediction_lower}"
            )
            assert prediction_upper - GT_upper <= 1e-5, (
                f"And({a_upper}, {b_upper}) expected {GT_upper}, "
                f"received {prediction_upper}"
            )
            model.flush()


def test_robust_and_3():
    """Unittest for robust (meaning lower_bd != upper_bd) upward 3-input real-value conjunction"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y, z = np.meshgrid(steps, steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    C = Proposition("C")
    ABC = And(A, B, C, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [ABC]

    for row in range(samples):
        for col in range(samples):
            for ht in range(samples):
                # inputs
                a_upper, b_upper, c_upper = (
                    x[row][col][ht],
                    y[row][col][ht],
                    z[row][col][ht],
                )
                a_lower = random.uniform(0, a_upper)
                b_lower = random.uniform(0, b_upper)
                c_lower = random.uniform(0, c_upper)

                # ground truth
                GT_lower = a_lower * b_lower * c_lower
                GT_upper = a_upper * b_upper * c_upper

                # facts per model
                facts = {
                    A: (a_lower, a_upper),
                    B: (b_lower, b_upper),
                    C: (c_lower, c_upper),
                }  # syntax for adding beliefs

                # load data into a new model
                model = Model()
                model.add_knowledge(
                    *formulae
                )  # unpacks the array formulae and hands one at a time to add_knowledge
                model.add_data(facts)

                # evaluate the conjunction
                ABC.upward()

                # test the prediction
                prediction_lower = ABC.get_data()[0]
                prediction_upper = ABC.get_data()[1]
                assert prediction_lower - GT_lower <= 1e-6, (
                    f"And({a_lower}, {b_lower}, {c_lower}) expected {GT_lower}, "
                    f"received {prediction_lower}"
                )
                assert prediction_upper - GT_upper <= 1e-6, (
                    f"And({a_upper}, {b_upper}, {c_upper}) expected {GT_upper}, "
                    f"received {prediction_upper}"
                )
                model.flush()


def test_robust_and_downward():
    # define model rules
    """Unittest for robust (meaning lower_bd != upper_bd) downward 2-input real-value conjunction"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [AB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a_upper, ab_upper = x[row][col], y[row][col]
            a_lower = random.uniform(0, a_upper)
            ab_lower = random.uniform(0, ab_upper)
            b_upper = a_upper
            b_lower = random.uniform(0, b_upper)

            # ground truth
            GT_lower_A = a_lower
            GT_upper_A = a_upper
            GT_lower_B = b_lower
            GT_upper_B = b_upper
            if a_upper > 0:
                GT_lower_B = max(b_lower, ab_lower / a_upper)
            if a_lower > 0:
                GT_upper_B = min(b_upper, ab_upper / a_lower)
            if b_upper > 0:
                GT_lower_A = max(a_lower, ab_lower / b_upper)
            if b_lower > 0:
                GT_upper_A = min(a_upper, ab_upper / b_lower)

            # facts per model
            facts = {
                A: (a_lower, a_upper),
                B: (b_lower, b_upper),
                AB: (ab_lower, ab_upper),
            }  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AB.downward()

            # test the prediction
            prediction_lower_A = A.get_data()[0]
            prediction_lower_B = B.get_data()[0]
            prediction_upper_A = A.get_data()[1]
            prediction_upper_B = B.get_data()[1]

            assert prediction_lower_B - GT_lower_B <= 1e-5, (
                "Given (ab_lower,a_upper, b_lower) = "
                f"({ab_lower}, {a_upper}, {b_lower}) expected B_lower = {GT_lower_B}, "
                f"received {prediction_lower_B}"
            )
            assert prediction_upper_B - GT_upper_B <= 1e-5, (
                f"Given (ab_upper,a_lower = ({ab_upper}, {a_lower}) expected B_upper = "
                f"{GT_upper_B}, received {prediction_upper_B}"
            )
            assert prediction_lower_A - GT_lower_A <= 1e-5, (
                "Given (ab_lower,b_upper, a_lower) = "
                f"({ab_lower}, {b_upper}, {a_lower}) expected A_lower = {GT_lower_A}, "
                f"received {prediction_lower_A}"
            )
            assert prediction_upper_A - GT_upper_A <= 1e-5, (
                f"Given (ab_upper,b_lower) = ({ab_upper}, {b_lower}) "
                f"expected A_upper = {GT_upper_A}, received {prediction_upper_A}"
            )
            model.flush()


def test_robust_or_2():
    """Unittest for upward 2-input real-value disjunction"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AvB = Or(A, B, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [AvB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a_upper, b_upper = x[row][col], y[row][col]
            a_lower = random.uniform(0, a_upper)
            b_lower = random.uniform(0, b_upper)

            # ground truth
            GT_lower = a_lower + b_lower - a_lower * b_lower
            GT_upper = a_upper + b_upper - a_upper * b_upper

            # facts per model
            facts = {
                A: (a_lower, a_upper),
                B: (b_lower, b_upper),
            }  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AvB.upward()

            # test the prediction
            prediction_lower = AvB.get_data()[0]
            prediction_upper = AvB.get_data()[1]
            assert prediction_lower - GT_lower <= 1e-5, (
                f"Or({a_lower}, {b_lower}) expected {GT_lower}, "
                f"received {prediction_lower}"
            )
            assert prediction_upper - GT_upper <= 1e-5, (
                f"Or({a_upper}, {b_upper}) expected {GT_upper}, "
                f"received {prediction_upper}"
            )
            model.flush()


def test_robust_or_3():
    """Unittest for upward 3-input real-value disjunction"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y, z = np.meshgrid(steps, steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    C = Proposition("C")
    AvBvC = Or(A, B, C, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [AvBvC]

    for row in range(samples):
        for col in range(samples):
            for ht in range(samples):
                # inputs
                a_upper, b_upper, c_upper = (
                    x[row][col][ht],
                    y[row][col][ht],
                    z[row][col][ht],
                )
                a_lower = random.uniform(0, a_upper)
                b_lower = random.uniform(0, b_upper)
                c_lower = random.uniform(0, c_upper)

                # ground truth
                GT_lower = (
                    a_lower
                    + b_lower
                    + c_lower
                    - a_lower * b_lower
                    - a_lower * c_lower
                    - b_lower * c_lower
                    + a_lower * b_lower * c_lower
                )
                GT_upper = (
                    a_upper
                    + b_upper
                    + c_upper
                    - a_upper * b_upper
                    - a_upper * c_upper
                    - b_upper * c_upper
                    + a_upper * b_upper * c_upper
                )

                # facts per model
                facts = {
                    A: (a_lower, a_upper),
                    B: (b_lower, b_upper),
                    C: (c_lower, c_upper),
                }  # syntax for adding beliefs

                # load data into a new model
                model = Model()
                model.add_knowledge(
                    *formulae
                )  # unpacks the array formulae and hands one at a time to add_knowledge
                model.add_data(facts)

                # evaluate the conjunction
                AvBvC.upward()

                # test the prediction
                prediction_lower = AvBvC.get_data()[0]
                prediction_upper = AvBvC.get_data()[1]
                assert prediction_lower - GT_lower <= 1e-5, (
                    f"Or({a_lower}, {b_lower}, {c_lower}) expected {GT_lower}, "
                    f"received {prediction_lower}"
                )
                assert prediction_upper - GT_upper <= 1e-5, (
                    f"Or({a_upper}, {b_upper}, {c_upper}) expected {GT_upper}, "
                    f"received {prediction_upper}"
                )
                model.flush()


def test_robust_or_downward():
    # define model rules
    """Unittest for robust (meaning lower_bd != upper_bd) downward 2-input real-value disjunction"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AvB = Or(A, B, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [AvB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a_upper, avb_upper = x[row][col], y[row][col]
            a_lower = random.uniform(0, a_upper)
            avb_lower = random.uniform(0, avb_upper)
            b_upper = a_upper
            b_lower = random.uniform(0, b_upper)

            # ground truth
            GT_upper_A = a_upper
            GT_upper_B = b_upper
            GT_lower_A = a_lower
            GT_lower_B = b_lower
            if a_upper < 1:
                GT_lower_B = max(b_lower, (avb_lower - a_upper) / (1 - a_upper))
            if b_upper < 1:
                GT_lower_A = max(a_lower, (avb_lower - b_upper) / (1 - b_upper))

            if a_lower < 1:
                GT_upper_B = min(b_upper, max(0, (avb_upper - a_lower) / (1 - a_lower)))
            if b_lower < 1:
                GT_upper_A = min(a_upper, max(0, (avb_upper - b_lower) / (1 - b_lower)))

            # facts per model
            facts = {
                A: (a_lower, a_upper),
                B: (b_lower, b_upper),
                AvB: (avb_lower, avb_upper),
            }  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AvB.downward()

            # test the prediction
            prediction_lower_A = A.get_data()[0]
            prediction_lower_B = B.get_data()[0]
            prediction_upper_A = A.get_data()[1]
            prediction_upper_B = B.get_data()[1]

            assert prediction_lower_B - GT_lower_B <= 1e-6, (
                f"Given (avb_lower,a_upper) = ({avb_lower}, {a_upper}) "
                f"expected B_lower = {GT_lower_B}, received {prediction_lower_B}"
            )
            assert prediction_upper_B - GT_upper_B <= 1e-6, (
                f"Given (avb_upper,a_lower) = ({avb_upper}, {a_lower}) "
                f"expected B_upper = {GT_upper_B}, received {prediction_upper_B}"
            )
            assert prediction_lower_A - GT_lower_A <= 1e-6, (
                f"Given (avb_lower,b_upper) = ({avb_lower}, {b_upper}) "
                f"expected A_lower = {GT_lower_A}, received {prediction_lower_A}"
            )
            assert prediction_upper_A - GT_upper_A <= 1e-6, (
                f"Given (avb_upper,b_lower) = ({avb_upper}, {b_lower}) "
                f"expected A_upper = {GT_upper_A}, received {prediction_upper_A}"
            )
            model.flush()


def test_robust_implies():
    """Unittest for robust upward real-value implication"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AimplB = Implies(A, B, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [AimplB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a_upper, b_upper = x[row][col], y[row][col]
            a_lower = random.uniform(0, a_upper)
            b_lower = random.uniform(0, b_upper)

            # ground truth
            GT_lower = 0
            if a_upper > 0:
                GT_lower = min(1, b_lower / a_upper)
            else:
                GT_lower = 1
            GT_upper = 1
            if a_lower > 0:
                GT_upper = min(1, b_upper / a_lower)

            # facts per model
            facts = {
                A: (a_lower, a_upper),
                B: (b_lower, b_upper),
            }  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AimplB.upward()

            # test the prediction
            prediction_lower = AimplB.get_data()[0]
            prediction_upper = AimplB.get_data()[1]
            assert prediction_lower - GT_lower <= 1e-5, (
                f"Implies(a_upper, b_lower) for Implies({a_upper}, {b_lower}) "
                f"expected {GT_lower}, received {prediction_lower}"
            )
            assert prediction_upper - GT_upper <= 1e-5, (
                f"Implies(a_lower, b_upper) for Implies({a_lower}, {b_upper}) "
                f"expected {GT_upper}, received {prediction_upper}"
            )
            model.flush()


def test_robust_implies_downward():
    # define model rules
    """Unittest for robust (meaning lower_bd != upper_bd) downward real-value implication"""

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AimplB = Implies(A, B, activation={"type": NeuralActivation.Product})

    # rules per model
    formulae = [AimplB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a_upper, aimplb_upper = x[row][col], y[row][col]
            a_lower = random.uniform(0, a_upper)
            aimplb_lower = random.uniform(0, aimplb_upper)
            b_upper = a_upper
            b_lower = random.uniform(0, b_upper)

            # ground truth; GTs for A & B are not symmetrical!
            GT_lower_B = max(b_lower, a_lower * aimplb_lower)
            GT_upper_B = b_upper
            if aimplb_upper < 1:
                GT_upper_B = min(b_upper, a_upper * aimplb_upper)

            GT_lower_A = a_lower
            GT_upper_A = a_upper
            if aimplb_upper > 0 and aimplb_upper < 1:
                GT_lower_A = max(a_lower, min(1, b_lower / aimplb_upper))
            if aimplb_lower > 0:
                GT_upper_A = max(a_upper, min(1, b_upper / aimplb_lower))

            # facts per model
            facts = {
                A: (a_lower, a_upper),
                B: (b_lower, b_upper),
                AimplB: (aimplb_lower, aimplb_upper),
            }  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AimplB.downward()

            # test the prediction
            prediction_lower_A = A.get_data()[0]
            prediction_lower_B = B.get_data()[0]
            prediction_upper_A = A.get_data()[1]
            prediction_upper_B = B.get_data()[1]

            assert prediction_lower_B - GT_lower_B <= 1e-6, (
                "Given (aimplb_lower,a_lower, b_lower) = "
                f"({aimplb_lower}, {a_lower}, {b_lower}) expected B_lower = "
                f"{GT_lower_B}, received {prediction_lower_B}"
            )
            assert prediction_upper_B - GT_upper_B <= 1e-6, (
                "Given (aimplb_upper,a_upper, b_upper) = "
                f"({aimplb_upper}, {a_upper}, {b_upper}) expected B_upper = "
                f"{GT_upper_B}, received {prediction_upper_B}"
            )
            assert prediction_lower_A - GT_lower_A <= 1e-6, (
                f"Given (aimplb_upper,b_lower,a_lower) = "
                f"({aimplb_upper}, {b_lower}, {a_lower}) expected A_lower = "
                f"{GT_lower_A}, received {prediction_lower_A}"
            )
            assert prediction_upper_A - GT_upper_A <= 1e-6, (
                f"Given (aimplb_lower,b_upper,a_upper) = "
                f"({aimplb_lower}, {b_upper}, {a_upper}) expected A_upper = "
                f"{GT_upper_A}, received {prediction_upper_A}"
            )
            model.flush()


if __name__ == "__main__":
    test_robust_and_2()  # PASSED
    test_robust_and_3()  # PASSED
    test_robust_or_2()  # PASSED
    test_robust_or_3()  # PASSED
    test_robust_implies()  # PASSED
    test_robust_and_downward()  # PASSED
    test_robust_or_downward()  # PASSED
    test_robust_implies_downward()  # PASSED
