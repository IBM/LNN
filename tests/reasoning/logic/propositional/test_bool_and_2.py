##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, And, Model, truth_table, fact_to_bool, bool_to_fact
import numpy as np
from functools import reduce


def test():
    """Unittest for upward 3-input boolean truth table"""

    TT = truth_table(3)

    # define the rules
    A = Proposition("A")
    B = Proposition("B")
    C = Proposition("C")
    A_B_C = And(A, B, C)

    formulae = [A_B_C]

    for row in TT:
        # ground truth
        GT = reduce(np.logical_and, map(fact_to_bool, row))

        # facts per model
        facts = {A: row[0], B: row[1], C: row[2]}

        # load data into a new model
        model = Model()
        model.add_knowledge(*formulae)
        model.add_data(facts)

        # evaluate the conjunction
        A_B_C.upward()

        # test the prediction
        prediction = A_B_C.state()
        assert prediction is bool_to_fact(
            GT
        ), f"And{row} expected {bool_to_fact(GT)}, received {prediction}"
        model.flush()


if __name__ == "__main__":
    test()
