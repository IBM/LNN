##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (
    Predicates,
    And,
    Model,
    Variable,
    truth_table,
    fact_to_bool,
    bool_to_fact,
)
from functools import reduce
import numpy as np


def test():
    """Unary FOL upward 2-input conjunction boolean truth table"""

    TT = truth_table(2)

    for row in TT:
        # get ground truth
        GT = reduce(np.logical_and, map(fact_to_bool, row))

        # load model and reason over facts
        model = Model()
        x = Variable("x")
        A, B = Predicates("A", "B", model=model)
        AB = And(A(x), B(x))

        # set model facts
        model.add_data({A: {"0": row[0]}, B: {"0": row[1]}})

        # evaluate the conjunction
        AB.upward()
        prediction = AB.state("0")
        assert prediction is bool_to_fact(
            GT
        ), f"And({row[0]}, {row[1]}) expected {GT}, received {prediction}"


if __name__ == "__main__":
    test()
