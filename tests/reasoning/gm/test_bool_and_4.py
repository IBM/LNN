##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, And, Model, Variable, truth_table, fact_to_bool, bool_to_fact
from functools import reduce
import numpy as np


def test():
    n_preds = 2
    n_vars = 1000
    TT = truth_table(n_preds)

    # get ground truth
    GT = [reduce(np.logical_and, fact_to_bool(*row)) for row in TT]

    # load model and reason over facts
    model = Model()
    var_labels = tuple(f"x{i}" for i in range(0, n_vars))
    variables = list(map(Variable, var_labels))

    preds = []
    for pred in range(n_preds):
        preds.append(Predicate(f"P{pred}", arity=n_vars))
    AB = And(*[pred(*variables) for pred in preds])
    model.add_knowledge(AB)

    # set model facts
    for idx, pred in enumerate(preds):
        test_case = {(f"{row}",) * n_vars: truth[idx] for row, truth in enumerate(TT)}
        model.add_data({pred: test_case})

    # inference
    AB.upward()

    # evaluate the conjunction
    for row in range(len(TT)):
        state = (str(row),) * n_vars
        prediction = AB.state(state)
        assert prediction is bool_to_fact(
            GT[row]
        ), f"And({TT[row]}) expected {GT}, received {prediction}"


if __name__ == "__main__":
    test()
