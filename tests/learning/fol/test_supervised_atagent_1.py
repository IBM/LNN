##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import itertools
from lnn import (Model, World, And, Bidirectional, ForAll, Variable,
                 TRUE, UPWARD)


def test():
    # background data (features)
    B = {'at-agent': [('cabinet',), ('wardrobe',), ('hanger',),
                      ('ladderback chair',)],
         'in': [('cabinet', 'shoe'), ('wardrobe', 'shirt'),
                ('hanger', 'coat'), ('ladderback chair', 'tuna'), ]}

    # positive (target) labels
    P = {'take': [('shoe',), ('shirt',), ('coat',), ('tuna',)]}

    # Predicates:
    # ['at-agent(x)', 'at-agent(z)', 'in(x,z)', 'in(z,x)']

    # Subrule template:
    # take(x):- at-agent(*) ∧ in(*, *) [x, z]

    # Learned rule:
    # take(x):- at-agent(z) ∧ in(z, x)

    model = Model()
    at_agent = model.add_predicates(1, 'at-agent', world=World.FALSE)
    in_ = model.add_predicates(2, 'in', world=World.FALSE)
    take = model.add_predicates(1, 'take', world=World.FALSE)

    model.add_facts({
        'at-agent': {c: TRUE for c in B['at-agent']},
        'in': {pair: TRUE for pair in B['in']},
        'take': {c: TRUE for c in P['take']}})

    x = Variable('x')
    z = Variable('z')

    variables = [x, z]
    singles = [(x,), (z,)]
    pairs = itertools.permutations(variables, 2)  # e.g. [(x, y), ...]
    pairs = list(pairs)

    neuron = {
        'bias_learning': False,
        'weights_learning': False}

    subrules = []
    for single in singles:
        for pair in pairs:
            k = str(len(subrules))
            model[k] = ForAll(Bidirectional(And(at_agent(*single),
                                                in_(*pair), neuron=neuron),
                                            take(x), neuron=neuron),
                              fully_grounded=True)
            subrules.append(model[k])

    rule = model['rule'] = And(*subrules, world=World.AXIOM,
                               neuron={'bias_learning': False})

    parameter_history = {'weights': True, 'bias': True}
    losses = {'contradiction': 1}

    total_loss, _ = model.train(
        direction=UPWARD,
        losses=losses,
        parameter_history=parameter_history
    )

    chosen_idx = []
    weighted_idx = (rule.neuron.weights == 1).nonzero(
        as_tuple=True)[0]
    for idx in weighted_idx:
        subrule_forall = rule.operands[idx]
        if subrule_forall.neuron.bounds_table[0][0] == 1:
            subrule_bidirectional = subrule_forall.operands[0]
            subrule_implication = subrule_bidirectional.Imp1
            subrule_and = subrule_implication.operands[0]
            subrule_and_vars = subrule_and.binding_str
            chosen_idx.append(subrule_and_vars)

    num_chosen = len(chosen_idx)
    assert num_chosen == 1, 'expected 1 got ' + str(num_chosen)
    assert chosen_idx[0] == ['z', 'z, x'], (
            'expected ' + str(['z', 'z, x']) +
            ' got ' + str(chosen_idx[0]))


if __name__ == "__main__":
    test()
    print('success')
