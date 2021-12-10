##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, And, Model, Variable, World, TRUE

model = Model()
x, y, z = map(Variable, ('x', 'y', 'z'))
model['P'] = Predicate()
model['B'] = Predicate()
model['And'] = And(model['P'](x), model['B'](x), world=World.AXIOM)


model['P'].add_facts({
    'Jon': TRUE,
    'Bob': TRUE})
model['B'].add_facts({
    ('James', 'Date1'): 0.4,
    ('Bob', 'Date2'): TRUE,
    ('Jon', 'Date1'): TRUE})
# print(model.infer())
model.print()
