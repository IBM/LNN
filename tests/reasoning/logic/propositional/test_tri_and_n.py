##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, And, Model, TRUE, FALSE, UNKNOWN


def test_upward():
    """standard upward ,n-input conjunction three-valued truth table"""

    TT = [
        # A, B, And(A, B, B, B, ...)
        [TRUE, TRUE, TRUE],
        [FALSE, TRUE, FALSE],
        [FALSE, FALSE, FALSE],
        [UNKNOWN, TRUE, UNKNOWN],
        [UNKNOWN, UNKNOWN, UNKNOWN],
        [UNKNOWN, FALSE, FALSE],
    ]

    # define the rules
    n = 1000
    propositions = list()
    for i in range(1, n):
        propositions.append(Proposition("p" + str(i)))
    formulae = [And(*propositions, name="And_n")]

    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {}
        facts["p1"] = row[0]
        for i in range(2, n):
            facts["p" + str(i)] = row[1]
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model["And_n"].upward()

        # evaluate the conjunction
        prediction = model["And_n"].state()
        assert (
            prediction is GT
        ), f"And({row[0]}, {row[1]}...) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    TT = [
        # B, And(A, B, B, ...), A
        [TRUE, TRUE, TRUE],
        [TRUE, FALSE, FALSE],
        [TRUE, UNKNOWN, UNKNOWN],
        [FALSE, TRUE, TRUE],  # contradiction at B [downward]
        [FALSE, FALSE, UNKNOWN],
        [FALSE, UNKNOWN, UNKNOWN],  # contradiction at And [upward]
    ]

    # define the rules
    n = 1000
    propositions = list()
    for i in range(1, n):
        propositions.append(Proposition("p" + str(i)))
    formulae = [And(*propositions, name="And_n")]

    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {}
        facts["And_n"] = row[1]
        for i in range(2, n):
            facts["p" + str(i)] = row[0]
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model["And_n"].downward(index=0)

        # evaluate the conjunction
        prediction = model["p1"].state()
        assert prediction is GT, (
            f"And(A, {row[0]}, ...)={row[1]} expected"
            + " A={GT}, received {prediction}"
        )
        model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
    print("success")
