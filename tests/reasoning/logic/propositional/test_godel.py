##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, And, Or, Implies, Model, NeuralActivation, Fact
import numpy as np


def test_and_2():
    """Unittest for upward 2-input real-value conjunction"""

    print("Testing 2-input real-valued conjunction....")

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, activation={"type": NeuralActivation.Godel})

    # rules per model
    formulae = [AB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a, b = x[row][col], y[row][col]

            # ground truth
            GT = min(a, b)  # (GT = ground truth) must be changed to the Godel T-Norm

            # facts per model
            facts = {A: (a, a), B: (b, b)}  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AB.upward()

            # test the prediction
            prediction = AB.get_data()[0]
            assert (
                prediction - GT <= 1e-7
            ), f"And({a}, {b}) expected {GT}, received {prediction}"
            model.flush()
    print("Success!")


def test_and_3():
    """Unittest for upward 3-input real-value conjunction"""
    print("Testing 3-input real-valued conjunction....")

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y, z = np.meshgrid(steps, steps, steps)

    # define the rules
    A = Proposition("A")
    # A.add_data(Fact.UNKNOWN)
    B = Proposition("B")
    # B.add_data(Fact.UNKNOWN)
    C = Proposition("C")
    # C.add_data(Fact.UNKNOWN)
    ABC = And(A, B, C, activation={"type": NeuralActivation.Godel})

    # rules per model
    formulae = [ABC]

    for row in range(samples):
        for col in range(samples):
            for ht in range(samples):
                # inputs
                a, b, c = x[row][col][ht], y[row][col][ht], z[row][col][ht]

                # ground truth
                GT = min(
                    a, b, c
                )  # (GT = ground truth) must be changed to the Godel T-Norm

                # facts per model
                facts = {A: (a, a), B: (b, b), C: (c, c)}  # syntax for adding beliefs

                # load data into a new model
                model = Model()
                model.add_knowledge(
                    *formulae
                )  # unpacks the array formulae and hands one at a time to add_knowledge
                model.add_data(facts)

            # evaluate the conjunction
            ABC.upward()

            # test the prediction
            prediction = ABC.get_data()[0]
            assert (
                prediction - GT <= 1e-7
            ), f"And({a}, {b}, {c}) expected {GT}, received {prediction}"
            model.flush()
    print("Success!")


def test_and_downward():
    # define model rules
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, activation={"type": NeuralActivation.Godel})
    model.add_knowledge(AB)

    # define model facts
    model.add_data(
        {
            A: Fact.TRUE,
            AB: Fact.FALSE,
        }
    )
    AB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is Fact.TRUE
    ), f"Expected input A to be TRUE, received {prediction}"
    prediction = B.state()
    assert (
        prediction is Fact.FALSE
    ), f"Expected input B to be False, received {prediction}"
    model.flush()


def test_or_downward():
    # define model rules
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    AvB = Or(A, B)
    model.add_knowledge(AvB)

    # define model facts
    model.add_data({A: Fact.FALSE, AvB: Fact.TRUE})
    AvB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is Fact.FALSE
    ), f"Expected input A to be False, received {prediction}"
    prediction = B.state()
    assert (
        prediction is Fact.TRUE
    ), f"Expected input B to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({AvB: Fact.FALSE})
    AvB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is Fact.FALSE
    ), f"Expected input A to be False, received {prediction}"
    prediction = B.state()
    assert (
        prediction is Fact.FALSE
    ), f"Expected input B to be False, received {prediction}"
    model.flush()


def test_or_2():
    """Unittest for upward 2-input real-value disjunction"""
    print("Testing 2-input real-valued disjunction....")

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AvB = Or(A, B, activation={"type": NeuralActivation.Godel})

    # rules per model
    formulae = [AvB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a, b = x[row][col], y[row][col]

            # ground truth
            GT = max(a, b)  # (GT = ground truth) must be changed to the Godel T-Norm

            # facts per model
            facts = {A: (a, a), B: (b, b)}  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AvB.upward()

            # test the prediction
            prediction = AvB.get_data()[0]
            assert (
                prediction - GT <= 1e-7
            ), f"Or({a}, {b}) expected {GT}, received {prediction}"
            model.flush()
    print("success")


def test_or_3():
    """Unittest for upward 3-input real-value disjunction"""
    print("Testing 3-input real-valued disjunction....")

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y, z = np.meshgrid(steps, steps, steps)

    # define the rules
    A = Proposition("A")
    # A.add_data(Fact.UNKNOWN)
    B = Proposition("B")
    # B.add_data(Fact.UNKNOWN)
    C = Proposition("C")
    # C.add_data(Fact.UNKNOWN)
    AvBvC = Or(A, B, C, activation={"type": NeuralActivation.Godel})

    # rules per model
    formulae = [AvBvC]

    for row in range(samples):
        for col in range(samples):
            for ht in range(samples):
                # inputs
                a, b, c = x[row][col][ht], y[row][col][ht], z[row][col][ht]

                # ground truth
                GT = max(
                    a, b, c
                )  # (GT = ground truth) must be changed to the Godel T-Norm

                # facts per model
                facts = {A: (a, a), B: (b, b), C: (c, c)}  # syntax for adding beliefs

                # load data into a new model
                model = Model()
                model.add_knowledge(
                    *formulae
                )  # unpacks the array formulae and hands one at a time to add_knowledge
                model.add_data(facts)

            # evaluate the conjunction
            AvBvC.upward()

            # test the prediction
            prediction = AvBvC.get_data()[0]
            assert (
                prediction - GT <= 1e-7
            ), f"Or({a}, {b}, {c}) expected {GT}, received {prediction}"
            model.flush()
    print("Success!")


def test_implies():
    """Unittest for upward real-valued implication"""
    print("Testing (2-input) real-valued implication....")

    samples = 21
    steps = np.linspace(0, 1, samples)
    x, y = np.meshgrid(steps, steps)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AimplB = Implies(A, B, activation={"type": NeuralActivation.Godel})

    # rules per model
    formulae = [AimplB]

    for row in range(samples):
        for col in range(samples):
            # inputs
            a, b = x[row][col], y[row][col]

            # ground truth
            GT = max(
                1 - a, b
            )  # (GT = ground truth) must be changed to the Godel T-Norm

            # facts per model
            facts = {A: (a, a), B: (b, b)}  # syntax for adding beliefs

            # load data into a new model
            model = Model()
            model.add_knowledge(
                *formulae
            )  # unpacks the array formulae and hands one at a time to add_knowledge
            model.add_data(facts)

            # evaluate the conjunction
            AimplB.upward()

            # test the prediction
            prediction = AimplB.get_data()[0]
            assert (
                prediction - GT <= 1e-7
            ), f"Implies({a}, {b}) expected {GT}, received {prediction}"
            model.flush()
    print("success")


def test_implies_downward():
    # define model rules
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    AimplB = Implies(A, B, activation={"type": NeuralActivation.Godel})
    model.add_knowledge(AimplB)

    # define model facts
    model.add_data({A: Fact.TRUE, AimplB: Fact.TRUE})
    AimplB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is Fact.TRUE
    ), f"Expected input A to be True, received {prediction}"
    prediction = B.state()
    assert (
        prediction is Fact.TRUE
    ), f"Expected input B to be True, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: Fact.TRUE, AimplB: Fact.FALSE})
    AimplB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is Fact.TRUE
    ), f"Expected input A to be True, received {prediction}"
    prediction = B.state()
    assert (
        prediction is Fact.FALSE
    ), f"Expected input B to be False, received {prediction}"
    model.flush()

    # define model facts
    model.add_data({A: Fact.FALSE, AimplB: Fact.TRUE})
    AimplB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is Fact.FALSE
    ), f"Expected input A to be False, received {prediction}"
    prediction = B.state()
    assert (
        prediction is Fact.UNKNOWN
    ), f"Expected input B to be Unknown, received {prediction}"

    # define model facts
    model.add_data({A: Fact.FALSE, AimplB: Fact.FALSE})
    AimplB.downward()

    # evaluate
    prediction = A.state()
    assert (
        prediction is Fact.CONTRADICTION
    ), f"Expected input A to be Contradiction, received {prediction}"
    prediction = B.state()
    assert (
        prediction is Fact.FALSE
    ), f"Expected input B to be False, received {prediction}"


if __name__ == "__main__":
    test_and_2()
    test_and_3()
    test_or_2()
    test_or_3()
    test_implies()
    test_and_downward()
    test_or_downward()
    test_implies_downward()
