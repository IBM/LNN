##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, And, Model, Variable, World, Fact

TRUE = Fact.TRUE
FALSE = Fact.FALSE

model = Model()
x, y, z = map(Variable, ("x", "y", "z"))
P = Predicate()
B = Predicate()
_And = And(P(x), B(x), world=World.AXIOM)

P.add_data({"Jon": TRUE, "Bob": TRUE})
B.add_data({("James", "Date1"): 0.4, ("Bob", "Date2"): TRUE, ("Jon", "Date1"): TRUE})
model.print()
model.add_knowledge(_And)
