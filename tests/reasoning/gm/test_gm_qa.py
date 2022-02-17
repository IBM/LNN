##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Predicate, And, Fact, Exists, Variable, Join


def test_1():
    join = Join.OUTER
    x = Variable('x')
    y = Variable('y')

    model = Model()

    model['director'] = Predicate('director', arity=2)
    model['starring'] = Predicate('starring', arity=2)

    facts = {'director': {('William_Shatner', 'The_captains'): Fact.TRUE},
             'starring': {('William_Shatner', 'The_captains'): Fact.TRUE,
                          ('Patrick_Stewart', 'The_captains'): Fact.TRUE}
             }
    model.add_facts(facts)

    query = Exists(x, And(model['director']((x, 'William_Shatner'), y),
                          model['starring'](x, y),
                          join=join),
                   name='Shatner-stars')
    model.add_formulae(query)

    model.infer()

    assert query.true_groundings == {'William_Shatner'}


def test_2():
    join = Join.OUTER
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')

    model = Model()

    model['director'] = Predicate('director', arity=2)
    model['starring'] = Predicate('starring', arity=2)

    model.add_facts({'director': {('William_Shatner',
                                   'The_captains'): Fact.TRUE},
                     'starring': {('William_Shatner',
                                   'The_captains'): Fact.TRUE,
                                  ('Patrick_Stewart',
                                   'The_captains'): Fact.TRUE}
                     })

    query = Exists(z, And(model['director']((x, 'William_Shatner'), y),
                          model['starring'](z, y),
                          join=join),
                   name='Shatner-stars')
    model.add_formulae(query)

    model.infer()

    assert query.true_groundings == {'William_Shatner', 'Patrick_Stewart'}


if __name__ == '__main__':
    test_1()
    test_2()

    print('success')
