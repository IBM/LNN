##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (Predicate, And, Model, Variable,
                 truth_table, fact_to_bool, bool_to_fact)
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
    var_labels = tuple(f'x{i}' for i in range(0, n_vars))
    variables = list(map(Variable, var_labels))

    for pred in range(n_preds):
        model[f'P{pred}'] = Predicate(arity=n_vars)

    preds = [model[f'P{pred}'](*variables) for pred in range(n_preds)]
    model['AB'] = And(*preds)

    # set model facts
    for pred in range(n_preds):

        test_case = {(f'{row}',) * n_vars: truth[pred]
                     for row, truth in enumerate(TT)}
        model.add_facts({f'P{pred}': test_case})

    # inference
    model['AB'].upward()

    # evaluate the conjunction
    for row in range(len(TT)):
        state = (str(row),) * n_vars
        prediction = model['AB'].state(state)
        assert prediction is bool_to_fact(GT[row]), (
            f'And({TT[row]}) expected {GT}, received {prediction}')
    print('success')


if __name__ == "__main__":
    test()
