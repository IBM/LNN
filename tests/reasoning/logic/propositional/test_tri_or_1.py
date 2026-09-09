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
    """standard upward ,2-input disjunction three-valued truth table"""

    TT = [
        # A, B, Or(A, B)
        [TRUE, TRUE, TRUE],
        [FALSE, TRUE, TRUE],
        [FALSE, FALSE, FALSE],
        [UNKNOWN, TRUE, TRUE],
        [UNKNOWN, UNKNOWN, UNKNOWN],
        [UNKNOWN, FALSE, UNKNOWN],
    ]

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Or(A, B)
    formulae = [AB]

    for i, row in enumerate(TT):
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {A: row[0], B: row[1]}
        model = Model()
        model.add_knowledge(*formulae)
        model.add_data(facts)
        AB.upward()

        # evaluate the conjunction
        prediction = AB.state()
        assert (
            prediction is GT
        ), f"{i} Or({row[0]}, {row[1]}) expected {GT}, received {prediction}"
        model.flush()


def test_downward():
    TT = [
        # B, Or(A, B), A
        [FALSE, FALSE, FALSE],
        [FALSE, UNKNOWN, UNKNOWN],
        [FALSE, TRUE, TRUE],
        [TRUE, FALSE, FALSE],  # contradiction at B
        [TRUE, UNKNOWN, UNKNOWN],  # contradiction at Or()
        [TRUE, TRUE, UNKNOWN],
    ]

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    AB = Or(A, B)
    formulae = [AB]

    for i, row in enumerate(TT):
        # load model and reason over facts
        facts = {B: row[0], AB: row[1]}
        model = Model()
        model.add_knowledge(*formulae)
        model.add_data(facts)
        AB.downward(index=0)

        # evaluate the conjunction
        prediction = A.state()
        assert prediction is row[2], (
            f"{i}: Or(A, {row[0]}) = {row[1]}, Expected A={row[2]}, "
            f"received {prediction}"
        )

        # evaluate the conjunction
        prediction = B.state()
        assert prediction is row[0], (
            f"{i}: Or(A, {row[0]}) = {row[1]}, Expected B={row[0]}, "
            f"received {prediction}"
        )
        model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
