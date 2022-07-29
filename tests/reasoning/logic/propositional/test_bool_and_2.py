##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from functools import reduce

import numpy as np
from lnn import Proposition, And, Model, truth_table, fact_to_bool, bool_to_fact


def test():
    """Unittest for upward 3-input boolean truth table"""

    TT = truth_table(3)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    C = Proposition("C")
    A_B_C = And(A, B, C, name="A_B_C")

    formulae = [A_B_C]

    for row in TT:
        # ground truth
        GT = reduce(np.logical_and, map(fact_to_bool, row))

        # facts per model
        facts = {"A": row[0], "B": row[1], "C": row[2]}

        # load data into a new model
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)

        # evaluate the conjunction
        model["A_B_C"].upward()

        # test the prediction
        prediction = model["A_B_C"].state()
        assert prediction is bool_to_fact(
            GT
        ), f"And{row} expected {bool_to_fact(GT)}, received {prediction}"
        model.flush()
    print("success")


if __name__ == "__main__":
    test()
