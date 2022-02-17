##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from functools import reduce

import numpy as np
from lnn import Predicate, And, Model, Variable, truth_table, fact_to_bool, bool_to_fact


def test():
    """FOL upward n-input conjunction boolean truth table

    10 predicates input to 1 conjunction, with binary groundings (x, y)
    collapses the truth table rows to groundings, instead of separate models
    """

    n_preds = 10
    TT = truth_table(n_preds)

    # get ground truth
    GT = [reduce(np.logical_and, map(fact_to_bool, row)) for row in TT]

    # load model and reason over facts
    model = Model()
    x, y = map(Variable, ("x", "y"))

    for pred in range(n_preds):
        model[f"P{pred}"] = Predicate(arity=2)
    model["AB"] = And(*[model[f"P{pred}"](x, y) for pred in range(n_preds)])

    # set model facts
    for pred in range(n_preds):
        model.add_facts(
            {
                f"P{pred}": {
                    (f"{row}", f"{row}"): truth[pred] for row, truth in enumerate(TT)
                }
            }
        )

    # inference
    model["AB"].upward()

    # evaluate the conjunction
    for row in range(len(TT)):
        prediction = model["AB"].state((str(row), str(row)))
        assert prediction is bool_to_fact(
            GT[row]
        ), f"And({TT[:, row]}) expected {GT}, received {prediction}"
    print("success")


if __name__ == "__main__":
    test()
