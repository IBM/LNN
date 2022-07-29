##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, Implies, Model, TRUE, FALSE, UNKNOWN


def test_upward():
    """standard upward ,2-input implies three-valued truth table"""

    # Kleene and Priest logics
    TT = [
        # A, B, Implies(A, B)
        [TRUE, TRUE, TRUE],
        [FALSE, TRUE, TRUE],
        [FALSE, FALSE, TRUE],
        [UNKNOWN, TRUE, TRUE],
        [UNKNOWN, UNKNOWN, UNKNOWN],
        [UNKNOWN, FALSE, UNKNOWN],
    ]

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Implies(A, B, name="AB")
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
            prediction == GT
        ), f"And({row[0]}, {row[1]}) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    TT = [
        # B, Implies(A, B), A
        [TRUE, TRUE, UNKNOWN],
        [TRUE, FALSE, TRUE],  # contradition at B [downward]
        [TRUE, UNKNOWN, UNKNOWN],  # True at And [upward]
        [FALSE, TRUE, FALSE],
        [FALSE, FALSE, TRUE],
        [FALSE, UNKNOWN, UNKNOWN],
    ]

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Implies(A, B, name="AB")
    formulae = [AB]

    for i, row in enumerate(TT):
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {"B": row[0], "AB": row[1]}
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model["AB"].downward(index=0)

        # evaluate the conjunction
        prediction = model["A"].state()
        assert prediction is GT, f"{i}: Expected {GT}, received {prediction}"
        model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
    print("success")
