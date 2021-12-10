##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, TRUE, Variable, Implies, Exists, ForAll, AXIOM

# Define model
model = Model()

# define rules
s, r, f = model.add_predicates(1, 'square', 'rectangle', '4 sides')

x = Variable('x')
axiom_1 = model['axiom_1'] = ForAll(Implies(s(x), r(x)), world=AXIOM)
axiom_2 = model['axiom_2'] = ForAll(Implies(r(x), f(x)), world=AXIOM)
query = model['query'] = Exists(x, f(x))

# define facts
model.add_facts({
    s.name: {'c': TRUE,
             'k': TRUE
             }
})
#
#                 ForAll ~.2
# s(x)     w=.8    ->      w=1     r(x)
# 'c':(.2, .3)    'c':(.7, .7)    'k':U
# 'k':T           'k':T           'c':U

# model.infer()
# plot_graph(model)
model.print()


