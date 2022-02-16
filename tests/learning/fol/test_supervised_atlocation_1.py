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
    B = {'atlocation': [('hat', 'rack'), ('moccasins', 'cabinet'),
                        ('tissue', 'basket')],
         'related_to': [('shoe cabinet', 'cabinet'), ('hat rack', 'rack'),
                        ('top hat', 'hat'), ('blue moccasins', 'moccasins'),
                        ('used tissue', 'tissue'),
                        ('wastepaper basket', 'basket')]}

    # positive (target) labels
    P = {'put': [('top hat', 'hat rack'), ('blue moccasins', 'shoe cabinet'),
                 ('used tissue', 'wastepaper basket')]}

    # Predicates:
    # ['atlocation( x,y)', 'atlocation( x,u)', 'atlocation( x,v)',
    #  'atlocation( y,x)', 'atlocation( y,u)', 'atlocation( y,v)',
    #  'atlocation( u,x)', 'atlocation( u,y)', 'atlocation( u,v)',
    #  'atlocation( v,x)', 'atlocation( v,y)', 'related_to( x,y)',
    #  'related_to( x,u)', 'related_to( x,v)', 'related_to( y,x)',
    #  'related_to( y,u)', 'related_to( y,v)', 'related_to( u,x)',
    #  'related_to( u,y)', 'related_to( u,v)', 'related_to( v,x)',
    #  'related_to( v,y)']

    # Learned rule:
    # put(x,y):- related_to(x,u) ∧ atlocation(u,v) ∧ related_to(y,v)

    valid_rules = [['x, u', 'u, v', 'y, v'],
                   ['x, v', 'v, u', 'y, u'],
                   ['y, u', 'v, u', 'x, v'],
                   ['y, v', 'u, v', 'x, u']]

    # Subrule template:
    # put(x,y):- related_to(*,*) ∧ atlocation(*,*) ∧ related_to(*,*) [x,y,u,v]

    model = Model()
    atlocation = model.add_predicates(2, 'atlocation', world=World.FALSE)
    related_to = model.add_predicates(2, 'related_to', world=World.FALSE)
    put = model.add_predicates(2, 'put', world=World.FALSE)

    model.add_facts({
        'atlocation': {pair: TRUE for pair in B['atlocation']},
        'related_to': {pair: TRUE for pair in B['related_to']},
        'put': {pair: TRUE for pair in P['put']}})

    x = Variable('x')
    y = Variable('y')
    u = Variable('u')
    v = Variable('v')

    variables = [x, y, u, v]
    pairs = itertools.permutations(variables, 2)  # e.g. [(x, y), ...]
    pairs = list(pairs)

    # e.g. [((x, y), (x, z)), ...]
    # Order of predicates in body matters, so permutations are needed.
    rule_vars = itertools.permutations(pairs, 3)
    rule_vars = list(rule_vars)

    '''
    rule_vars = [((x, u), (u, v), (y, v)), ((x, y), (x, u), (x, v)),
                 ((x, y), (x, u), (y, x)), ((x, v), (v, u), (y, u)),
                 ((u, y), (u, v), (v, u)), ((y, v), (u, y), (v, u)),
                 ((x, u), (u, x), (v, u)), ((x, u), (x, v), (y, u)),
                 ((x, u), (u, v), (v, x)), ((x, y), (y, x), (v, x))]
    '''
    neuron = {
        'bias_learning': False,
        'weights_learning': False}

    subrules = []
    for i, preds_vars in enumerate(rule_vars):
        vars1, vars2, vars3 = preds_vars
        model[str(i)] = ForAll(Bidirectional(And(related_to(*vars1),
                                                 atlocation(*vars2),
                                                 related_to(*vars3),
                                                 neuron=neuron),
                                             put(x, y),
                                             neuron=neuron),
                               fully_grounded=True)
        subrules.append(model[str(i)])

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
    rule_idx = []
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
            rule_idx.append(idx)

    num_chosen = len(chosen_idx)
    assert num_chosen == 4, 'expected 4 got ' + str(num_chosen)

    assert all([r in valid_rules for r in chosen_idx]), ('some rule(s)' +
                                                         'are not found')


if __name__ == "__main__":
    test()
    print('success')
