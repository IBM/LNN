##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, And, Model, Variable, truth_table, fact_to_bool, bool_to_fact
from functools import reduce
import numpy as np


def test():
    """FOL upward n-input conjunction boolean truth table

    10 predicates input to 1 conjunction, with unary groundings (x)
    collapses the truth table rows to groundings, instead of separate models
    """

    n_preds = 10
    TT = truth_table(n_preds)

    # get ground truth
    GT = [reduce(np.logical_and, map(fact_to_bool, row)) for row in TT]

    # load model and reason over facts
    model = Model()
    x = Variable("x")

    preds = []
    for pred in range(n_preds):
        preds.append(Predicate(f"P{pred}"))
    AB = And(*[preds[p](x) for p in range(n_preds)])
    model.add_knowledge(AB)

    # set model facts
    for idx, pred in enumerate(preds):
        model.add_data({pred: {f"{row}": truth[idx] for row, truth in enumerate(TT)}})

    # evaluate the conjunction
    AB.upward()

    for row in range(len(TT)):
        prediction = AB.state(str(row))
        assert prediction is bool_to_fact(
            GT[row]
        ), f"And({TT[:,row]}) expected {GT}, received {prediction}"
    print("success")


if __name__ == "__main__":
    test()
