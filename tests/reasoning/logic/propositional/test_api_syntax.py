##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, TRUE, AXIOM


def test():
    model = Model()

    smokes,  friends = model.add_propositions("Smokes", "Friends")
    # check that more propositions can be added
    colleagues, gender = model.add_propositions("Colleagues", "GenderAlike")
    assert colleagues.name == "Colleagues"
    assert gender.name == "GenderAlike"

    formula = smokes.And(colleagues, gender).Implies(
        friends, world=AXIOM)

    facts = {
      'Smokes': TRUE,
      'Colleagues': TRUE,
      'GenderAlike': TRUE
    }

    model.add_formulae(formula)
    model.add_facts(facts)
    model.infer()
    model.print()
    assert model['Friends'].state() == TRUE, "Not friends :-("
