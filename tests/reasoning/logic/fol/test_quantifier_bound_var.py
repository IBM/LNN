##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (Model, Variable, TRUE, FALSE, UNKNOWN, ForAll, Exists)


def test_1():
    """Quantifier with bounded variables, upward on predicate
    UNKNOWN result when not fully grounded
    """
    x = Variable('x')
    model = Model()
    A, S = model.add_predicates(1, 'A', 'S')
    All = model['All'] = ForAll(x, A(x))
    Some = model['Some'] = Exists(x, S(x))

    model.add_facts({
        'A': {
            '0': TRUE,
            '1': TRUE,
            '2': TRUE},
        'S': {
            '0': FALSE,
            '1': FALSE,
            '2': FALSE}
        })

    model.upward()
    predictions = [All.state(), Some.state()]
    assert predictions[0] is UNKNOWN, (
        f'ForAll expected as UNKNOWN, received {predictions[0]}'
        'cannot learn to be TRUE unless fully grounded')
    assert predictions[1] is UNKNOWN, (
        f'Exists expected as UNKNOWN, received {predictions[1]}'
        'cannot learn to be FALSE unless fully grounded')


def test_2():
    """Quantifier with bounded variables, upward on predicate
    Single predicate truth updates quantifier truth
    """
    x = Variable('x')
    model = Model()
    A, S = model.add_predicates(1, 'A', 'S')
    All = model['All'] = ForAll(x, A(x))
    Some = model['Some'] = Exists(x, S(x))

    model.add_facts({
        'A': {
            '0': TRUE,
            '1': TRUE,
            '2': FALSE},
        'S': {
            '0': FALSE,
            '1': FALSE,
            '2': TRUE}
        })

    model.upward()
    assert Some.state() is TRUE, (
        f'ForAll expected as TRUE, received {Some.state()}')
    assert All.state() is FALSE, (
        f'Exists expected as FALSE, received {All.state()}')


if __name__ == "__main__":
    test_1()
    test_2()
    print('success')
