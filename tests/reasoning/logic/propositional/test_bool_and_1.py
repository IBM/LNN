##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import numpy as np
from lnn import (Proposition, And, Model, TRUE, FALSE, UNKNOWN, CONTRADICTION,
                 truth_table, fact_to_bool, bool_to_fact)


def test_upward():
    """standard upward, 2-input conjunction boolean truth table"""

    # define the rules
    A = Proposition('A')
    B = Proposition('B')
    AB = And(A, B, name='AB')
    formulae = [AB]

    for row in truth_table(2):
        # get ground truth
        GT = np.logical_and(*list(map(fact_to_bool, row)))

        # load model and reason over facts
        facts = {'A': row[0], 'B': row[1]}
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model['AB'].upward()

        # evaluate the conjunction
        prediction = model['AB'].state()
        assert prediction is bool_to_fact(GT), (
            f'And({row[0]}, {row[1]}) expected {GT}, received {prediction}')
        model.flush()


def test_downward():
    # define model rules
    model = Model()
    A = model['A'] = Proposition('A')
    B = model['B'] = Proposition('B')
    model['AB'] = And(A, B)

    # define model facts
    model.add_facts({
        'A': TRUE,
        'AB': FALSE,
    })
    model['AB'].downward()

    # evaluate
    prediction = model['A'].state()
    assert prediction is TRUE, (
        f'Expected input A to be TRUE, received {prediction}')
    prediction = model['B'].state()
    assert prediction is FALSE, (
        f'Expected input B to be False, received {prediction}')
    model.flush()

    # define model facts
    model.add_facts({'AB': TRUE})
    model['AB'].downward()

    # evaluate
    prediction = model['A'].state()
    assert prediction is TRUE, (
        f'Expected input A to be TRUE, received {prediction}')
    prediction = model['B'].state()
    assert prediction is TRUE, (
        f'Expected input B to be TRUE, received {prediction}')
    model.flush()

    # define model facts
    model.add_facts({
        'A': FALSE,
        'AB': FALSE
    })
    model['AB'].downward()

    # evaluate
    prediction = model['A'].state()
    assert prediction is FALSE, (
        f'Expected input A to be False, received {prediction}')
    prediction = model['B'].state()
    assert prediction is UNKNOWN, (
        f'Expected input B to be Unknown, received {prediction}')
    model.flush()

    # define model facts
    model.add_facts({
        'A': FALSE,
        'AB': TRUE
    })
    model['AB'].downward()

    # evaluate
    assert model['A'].state() is CONTRADICTION, (
        f'Expected input B to be Contradiction, received {prediction}')
    assert model['B'].state() is TRUE, (
        f'Expected input B to be Contradiction, received {prediction}')
    model.flush()

    # define model facts
    model['AB'].downward()

    # evaluate
    assert model['A'].state() is UNKNOWN, (
        f'Expected input B to be Unknown, received {prediction}')
    assert model['B'].state() is UNKNOWN, (
        f'Expected input B to be Unknown, received {prediction}')
    model.flush()

    # define model facts
    model.add_facts({'A': TRUE})
    model['AB'].downward()

    # evaluate
    assert model['A'].state() is TRUE, (
        f'Expected input B to be Unknown, received {prediction}')
    assert model['B'].state() is UNKNOWN, (
        f'Expected input B to be Unknown, received {prediction}')
    model.flush()

    # define model facts
    model.add_facts({
        'A': TRUE,
        'B': FALSE,
    })
    model['AB'].downward()

    # evaluate
    assert model['A'].state() is TRUE, (
        f'Expected input B to be Unknown, received {prediction}')
    assert model['B'].state() is FALSE, (
        f'Expected input B to be False, received {prediction}')
    model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
    print('success')
