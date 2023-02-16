##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, Or, Model, Fact

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test_upward():
    """standard upward ,n-input disjunction three-valued truth table"""

    TT = [
        # A, B, Or(A, B, ...)
        [TRUE, TRUE, TRUE],
        [FALSE, TRUE, TRUE],
        [FALSE, FALSE, FALSE],
        [UNKNOWN, TRUE, TRUE],
        [UNKNOWN, UNKNOWN, UNKNOWN],
        [UNKNOWN, FALSE, UNKNOWN],
    ]

    # define the rules
    n = 1000
    props = list()
    for i in range(n):
        props.append(Proposition("p" + str(i)))
    Or_n = Or(*props)

    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {props[0]: row[0]}
        for i in range(1, n):
            facts[props[i]] = row[1]
        model = Model()
        model.add_knowledge(Or_n)
        model.add_data(facts)
        Or_n.upward()

        # evaluate the conjunction
        prediction = Or_n.state()
        assert (
            prediction is GT
        ), f"Or({row[0]}, {row[1]}...) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    TT = [
        # B, Or(A, B, ...), A
        [FALSE, FALSE, FALSE],
        [FALSE, UNKNOWN, UNKNOWN],
        [FALSE, TRUE, TRUE],
        [TRUE, FALSE, FALSE],  # contradiction at B
        [TRUE, UNKNOWN, UNKNOWN],  # contradiction at Or()
        [TRUE, TRUE, UNKNOWN],
    ]

    # define the rules
    n = 1000
    props = list()
    for i in range(n):
        props.append(Proposition("p" + str(i)))
    Or_n = Or(*props)

    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {Or_n: row[1]}
        for i in range(1, n):
            facts[props[i]] = row[0]
        model = Model()
        model.add_knowledge(Or_n)
        model.add_data(facts)
        Or_n.downward(index=0)

        # evaluate the conjunction
        prediction = props[0].state()
        assert prediction is GT, (
            f"Or(A, {row[0]}, ...)={row[1]} expected A={GT}, " + "received {prediction}"
        )
        model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
