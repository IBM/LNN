##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (Model, Predicate, Variable, Implies, And, plot_graph,
                 AXIOM, CLOSED)


model = Model()

# define the rules/knowledge
teacher, student, course, company = map(
    Variable, ('teacher', 'student', 'course', 'company'))

Teaches = model['Teaches'] = Predicate(arity=2, world=AXIOM)
Takes = model['Takes'] = Predicate(arity=2, world=AXIOM)
JobOffers = model['JobOffers'] = Predicate(arity=2, world=CLOSED)

model['Teaches_Take_JobOffers'] = Implies(
    And(Teaches(teacher, course), Takes(student, course)),
    JobOffers(student, company),
    world=AXIOM)

model.infer(lifted=True)
model.print()
plot_graph(model)
