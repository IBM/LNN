##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Variable, TRUE, FALSE, UNKNOWN, ForAll, Exists


def test_1():
    """Quantifier with free variables, upward on predicate"""
    p, c = map(Variable, ('dbo:Person', 'dbo:City'))
    model = Model()
    mayor = model.add_predicates(2, 'dbo:Mayor')

    # List all the mayors of Boston
    Some = model['Some'] = Exists(
        p, mayor(p, c))

    GT = ['dbr:Marty_Walsh_(politician)', 'dbr:Lori_Lightfoot']

    model.add_facts({
        mayor.name: {
            ('dbr:Kim_Janey', 'dbr:Boston'): UNKNOWN,
            (GT[0], 'dbr:Boston'): TRUE,
            ('dbr:Tishaura_Jones', 'dbr:St._Louis'): UNKNOWN,
            (GT[1], 'dbr:Chicago'): TRUE}
        })

    model.upward()
    assert Some.true_groundings == set(GT), (
        f'expected True groundings to be {GT}, received {Some.true_groundings}'
    )


def test_2():
    """Quantifier with free variables, upward on predicate
    UNKNOWN result when not fully grounded
    """
    p, c = map(Variable, ('dbo:Person', 'dbo:City'))
    model = Model()
    mayor = model.add_predicates(2, 'dbo:Mayor')

    # List all the mayors of Boston
    Some = model['Some'] = Exists(
        p, mayor(p, (c, ['dbr:Chicago', 'dbr:Boston'])))

    GT_truth = ['dbr:Marty_Walsh_(politician)']
    GT_bindings = [('dbr:Lori_Lightfoot', 'dbr:Chicago'),
                   ('dbr:Kim_Janey', 'dbr:Boston'),
                   (GT_truth[0], 'dbr:Boston')]

    model.add_facts({
        mayor.name: {
            GT_bindings[0]: UNKNOWN,
            GT_bindings[1]: UNKNOWN,
            ('dbr:Tishaura_Jones', 'dbr:St._Louis'): UNKNOWN,
            GT_bindings[2]: TRUE,
        }})

    model.upward()
    assert Some.groundings == set(GT_bindings), (
        f'expected groundings to be bound to GT bindings {GT_bindings}, '
        f'received {Some.groundings}')
    assert Some.true_groundings == set(GT_truth), (
        f'expected True groundings to be {GT_truth}, '
        f'received {Some.true_groundings}')


def test_3():
    """Quantifier with free variables, upward on predicate
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
