##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Fact, World, Propositions

TRUE = Fact.TRUE


def test():
    model = Model()

    Smokes, Friends = Propositions("Smokes", "Friends", model=model)
    # check that more propositions can be added
    Colleagues, Gender = Propositions("Colleagues", "GenderAlike", model=model)
    assert Colleagues.name == "Colleagues"
    assert Gender.name == "GenderAlike"
    assert Colleagues in model

    Smokes.And(Colleagues, Gender).Implies(Friends, world=World.AXIOM)

    facts = {Smokes: TRUE, Colleagues: TRUE, Gender: TRUE}

    model.add_data(facts)
    model.infer()
    model.print()
    assert Friends.state() == TRUE, "Not friends :-("
