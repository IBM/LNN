##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Variable, Predicate, AXIOM, Implies, Bidirectional, \
    TRUE, CLOSED

model = Model()

# define the rules/knowledge
x, y = map(Variable, ('x', 'y'))

Smokes = Predicate('Smokes')
Cancer = Predicate('Cancer', world=CLOSED)
Friends = Predicate('Friends', arity=2)

S_Implies_C = model['Smokers have Cancer'] = (
    Implies(
        Smokes(x),
        Cancer(x),
        world=AXIOM)
    )
Friends_Smoke = model['Smokers befriend Smokers'] = (
    Implies(
        Friends(x, y),
        Bidirectional(Smokes(x), Smokes(y)),
        world=AXIOM)
    )

# training data
model.add_facts({
    'Friends': {
        ('Anna', 'Bob'): TRUE,
        ('Bob', 'Anna'): TRUE,
        ('Anna', 'Edward'): TRUE,
        ('Edward', 'Anna'): TRUE,
        ('Anna', 'Frank'): TRUE,
        ('Frank', 'Anna'): TRUE,
        ('Bob', 'Chris'): TRUE,
        ('Chris', 'Bob'): TRUE,
        ('Chris', 'Daniel'): TRUE,
        ('Daniel', 'Chris'): TRUE,
        ('Edward', 'Frank'): TRUE,
        ('Frank', 'Edward'): TRUE,
        ('Gary', 'Helen'): TRUE,
        ('Helen', 'Gary'): TRUE,
        ('Gary', 'Anna'): TRUE,
        ('Anna', 'Gary'): TRUE
    },
    'Smokes': {
        'Anna': TRUE,
        'Edward': TRUE,
        'Frank': TRUE,
        'Gary': TRUE
    },
    'Cancer': {
        'Anna': TRUE,
        'Edward': TRUE,
    }
})

model.infer()  # should be .train()
model.print()

# testing data
model.flush()
model.add_facts({
    'Friends': {
        ('Ivan', 'John'): TRUE,
        ('John', 'Ivan'): TRUE,
        ('Katherine', 'Lars'): TRUE,
        ('Lars', 'Katherine'): TRUE,
        ('Michael', 'Nick'): TRUE,
        ('Nick', 'Michael'): TRUE,
        ('Ivan', 'Michael'): TRUE,
        ('Michael', 'Ivan'): TRUE
    },
    'Smokes': {
        'Ivan': TRUE,
        'Nick': TRUE
    }
})
model.infer()
model.print()
