##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, Or, Model, TRUE, FALSE, UNKNOWN


def test_upward():
    """standard upward ,n-input disjunction three-valued truth table"""

    TT = [
        # A, B, Or(A, B, ...)
        [TRUE, TRUE, TRUE],
        [FALSE, TRUE, TRUE],
        [FALSE, FALSE, FALSE],
        [UNKNOWN, TRUE, TRUE],
        [UNKNOWN, UNKNOWN, UNKNOWN],
        [UNKNOWN, FALSE, UNKNOWN]
    ]

    # define the rules
    n = 1000
    propositions = list()
    for i in range(1, n):
        propositions.append(Proposition('p' + str(i)))
    formulae = [Or(*propositions, name='Or_n')]

    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {'p1': row[0]}
        for i in range(2, n):
            facts['p' + str(i)] = row[1]
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model['Or_n'].upward()

        # evaluate the conjunction
        prediction = model['Or_n'].state()
        assert prediction is GT, (
            f'Or({row[0]}, {row[1]}...) expected {GT}, received {prediction}')
        model.flush()


def test_downward():
    TT = [
        # B, Or(A, B, ...), A
        [FALSE, FALSE, FALSE],
        [FALSE, UNKNOWN, UNKNOWN],
        [FALSE, TRUE, TRUE],
        [TRUE, FALSE, FALSE],      # contradiction at B
        [TRUE, UNKNOWN, UNKNOWN],  # contradiction at Or()
        [TRUE, TRUE, UNKNOWN],
    ]

    # define the rules
    n = 1000
    propositions = list()
    for i in range(1, n):
        propositions.append(Proposition('p' + str(i)))
    formulae = [Or(*propositions, name='Or_n')]

    for row in TT:
        # get ground truth
        GT = row[2]

        # load model and reason over facts
        facts = {'Or_n': row[1]}
        for i in range(2, n):
            facts['p' + str(i)] = row[0]
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model['Or_n'].downward(index=0)

        # evaluate the conjunction
        prediction = model['p1'].state()
        assert prediction is GT, (
            f'Or(A, {row[0]}, ...)={row[1]} expected A={GT}, ' +
            'received {prediction}')
        model.flush()


if __name__ == "__main__":
    test_upward()
    test_downward()
    print('success')
