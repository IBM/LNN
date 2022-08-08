##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Fact, World

TRUE = Fact.TRUE


def test():
    model = Model()

    Smokes, Friends = model.add_propositions("Smokes", "Friends")
    # check that more propositions can be added
    Colleagues, Gender = model.add_propositions("Colleagues", "GenderAlike")
    assert Colleagues.name == "Colleagues"
    assert Gender.name == "GenderAlike"

    formula = Smokes.And(Colleagues, Gender).Implies(Friends, world=World.AXIOM)

    facts = {Smokes: TRUE, Colleagues: TRUE, Gender: TRUE}

    model.add_knowledge(formula)
    model.add_data(facts)
    model.infer()
    model.print()
    assert Friends.state() == TRUE, "Not friends :-("
