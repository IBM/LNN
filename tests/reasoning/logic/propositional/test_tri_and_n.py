##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Propositions, And, Model, Fact

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


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
    model = Model()
    n = 1000
    props = Propositions(*["p" + str(i) for i in range(n)], model=model)
    And_n = And(*props)

    facts = {}
    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts[props[0]] = row[0]
        for i in range(1, n):
            facts[props[i]] = row[1]
        model.add_data(facts)
        And_n.upward()

        # evaluate the conjunction
        prediction = And_n.state()
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
    model = Model()
    n = 3
    props = Propositions(*["p" + str(i) for i in range(n)], model=model)
    And_n = And(*props)

    facts = {}
    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts[And_n] = row[1]
        for i in range(1, n):
            facts[props[i]] = row[0]
        model.add_data(facts)
        And_n.downward(index=0)

        # evaluate the conjunction input A
        prediction = props[0].state()
        assert (
            prediction is GT
        ), f"And(A, {row[0]}, ...)={row[1]} expected A={GT}, received {prediction}"
        model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
