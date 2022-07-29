##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, And, Model, TRUE, FALSE, UNKNOWN


def test_upward():
    """standard upward ,2-input conjunction three-valued truth table"""

    TT = [
        # A, B, And(A, B)
        [TRUE, TRUE, TRUE],
        [FALSE, TRUE, FALSE],
        [FALSE, FALSE, FALSE],
        [UNKNOWN, TRUE, UNKNOWN],
        [UNKNOWN, UNKNOWN, UNKNOWN],
        [UNKNOWN, FALSE, FALSE],
    ]

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, name="AB")
    formulae = [AB]

    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {"A": row[0], "B": row[1]}
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model["AB"].upward()

        # evaluate the conjunction
        prediction = model["AB"].state()
        assert (
            prediction is GT
        ), f"And({row[0]}, {row[1]}) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    TT = [
        # B, And(A, B), A
        [TRUE, TRUE, TRUE],
        [TRUE, FALSE, FALSE],
        [TRUE, UNKNOWN, UNKNOWN],
        [FALSE, TRUE, TRUE],  # contradiction at B [downward]
        [FALSE, FALSE, UNKNOWN],
        [FALSE, UNKNOWN, UNKNOWN],  # contradiction at And [upward]
    ]

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, name="AB")
    formulae = [AB]

    for i, row in enumerate(TT):
        # load model and reason over facts
        facts = {"B": row[0], "AB": row[1]}
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model["AB"].downward(index=0)

        # evaluate the conjunction
        prediction = model["A"].state()
        assert (
            prediction is row[2]
        ), f"{i} And{row} Expected {row[2]}, received {prediction}"

        prediction = model["B"].state()
        assert (
            prediction is row[0]
        ), f"{i} And{row} Expected {row[0]}, received {prediction}"
        model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
    print("success")
