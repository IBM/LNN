##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from functools import reduce

import numpy as np
from lnn import (Predicate, And, Model, Variable,
                 truth_table, fact_to_bool, bool_to_fact)


def test():
    """Unary FOL upward 2-input conjunction boolean truth table"""

    TT = truth_table(2)

    for row in TT:
        # get ground truth
        GT = reduce(np.logical_and, map(fact_to_bool, row))

        # load model and reason over facts
        x = Variable('x')
        model = Model()
        model['A'] = Predicate()
        model['B'] = Predicate()
        model['AB'] = And(model['A'](x), model['B'](x))

        # set model facts
        model.add_facts({
            'A': {'0': row[0]},
            'B': {'0': row[1]}
        })

        # evaluate the conjunction
        model['AB'].upward()
        prediction = model['AB'].state('0')
        assert prediction is bool_to_fact(GT), (
            f'And({row[0]}, {row[1]}) expected {GT}, received {prediction}')
    print('success')


if __name__ == "__main__":
    test()
