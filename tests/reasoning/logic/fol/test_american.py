##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (Predicate, Variable, Join, And,
                 Exists, Implies, ForAll, Model, Fact, World)


def test_1():
    """The 'American' theorem proving example

    """

    x, y, z, w = map(Variable, ['x', 'y', 'z', 'w'])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    owns = model['owns'] = Predicate(arity=2, name='owns')
    missile = model['missile'] = Predicate('missile')
    american = model['american'] = Predicate('american')
    enemy = model['enemy'] = Predicate(arity=2, name='enemy')
    hostile = model['hostile'] = Predicate('hostile')
    criminal = model['criminal'] = Predicate('criminal')
    weapon = model['weapon'] = Predicate('weapon')
    sells = model['sells'] = Predicate(arity=3, name='sells')

    # Define and add the background knowledge to  the model.
    model['america-enemies'] = ForAll(x, Implies(enemy(x, (y, 'America')),
                                                 hostile(x),
                                                 name='enemy->hostile',
                                                 join=Join.OUTER),
                                      name='america-enemies', join=Join.OUTER,
                                      world=World.AXIOM)
    model['crime'] = ForAll(x, y, z, Implies(And(american(x), weapon(y),
                                                 sells(x, y, z), hostile(z),
                                                 name='american-to-hostile',
                                                 join=Join.OUTER),
                                             criminal(x),
                                             name='implies-criminal',
                                             join=Join.OUTER), name='crime',
                            join=Join.OUTER, world=World.AXIOM)
    model['nono_missiles_byWest'] = ForAll(x, Implies(And(missile(x),
                                                          owns((y, 'Nono'), x),
                                                          name='nono-missile',
                                                          join=Join.OUTER),
                                                      sells((z, 'West'), x,
                                                            (y, 'Nono')),
                                                      name='West-to-Nono',
                                                      join=Join.OUTER),
                                           name='nono_missiles_byWest',
                                           join=Join.OUTER, world=World.AXIOM)
    model['missiles-are-weapons'] = ForAll(x, Implies(missile(x), weapon(x),
                                                      name='missile-weapon',
                                                      join=Join.OUTER),
                                           name='missiles-are-weapons',
                                           join=Join.OUTER, world=World.AXIOM)

    # Define queries
    model['query'] = Exists(x, criminal(x), name='who-is-criminal',
                            join=Join.OUTER)

    # Add facts to model.
    model.add_facts({'owns': {('Nono', 'M1'): Fact.TRUE},
                     'missile': {'M1': Fact.TRUE},
                     'american': {'West': Fact.TRUE},
                     'enemy': {('Nono', 'America'): Fact.TRUE},
                     })

    steps, facts_inferred = model.infer()

    # Currently finishes in 5 inference steps
    assert steps == 5, "FAILED 😔"

    GT_o = dict([
        (('West'), Fact.TRUE)])

    assert all([model['query'].state(groundings=g) is GT_o[g]
                for g in GT_o]), "FAILED 😔"


if __name__ == "__main__":
    test_1()
    print('success')
