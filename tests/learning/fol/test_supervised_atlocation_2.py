##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import itertools
from lnn import (Model, World, And, Bidirectional, ForAll, Variable,
                 TRUE, UPWARD, UNKNOWN, Implies)


def prune_rules(B, P, variables, rule_vars):
    """
    This function performs inference on background facts using rule defined
    by 'rule_vars' and return the number of predicted facts that agree
    with examples in P.
    :param B:  Background facts.
    :param P:  Positive examples.
    :param variables: Set of variables.
    :param rule_vars: Variable configuration for candidate rule.
    :return: Sum of facts correctly predicted as True.
    """

    neuron = {
        'bias_learning': False,
        'weights_learning': False}
    model = Model()
    atlocation = model.add_predicates(2, 'atlocation', world=World.FALSE)
    related_to = model.add_predicates(2, 'related_to', world=World.FALSE)
    put = model.add_predicates(2, 'put', world=World.FALSE)

    model.add_facts({
        'atlocation': {pair: TRUE for pair in B['atlocation']},
        'related_to': {pair: TRUE for pair in B['related_to']},
        'put': {pair: UNKNOWN for pair in P['put']}})

    # Set ground truth for target predicate
    GT = {}
    for grounding in P['put']:
        GT[grounding] = TRUE

    x, y, u, v = variables
    vars1, vars2, vars3 = rule_vars
    model['test_rule'] = ForAll(Implies(And(related_to(*vars1),
                                            atlocation(*vars2),
                                            related_to(*vars3),
                                            neuron=neuron),
                                        put(x, y),
                                        neuron=neuron),
                                world=World.AXIOM)

    model.infer()

    return sum([model['put'].state(groundings=g) is GT[g] for g in GT])


def test():
    # background data (features)
    B = {'atlocation': [('towel', 'towel rail'), ('cap', 'hat rack'),
                        ('flour', 'shelf'), ('handsoap', 'sink'),
                        ('sugar', 'shelf'), ('shoes', 'cabinet'),
                        ('peanut oil', 'shelf'), ('bathrobe', 'hook'),
                        ('white cap', 'hat rack')],
         'related_to': [('white cap', 'white cap'), ('hat rack', 'hat rack'),
                        ('shelf', 'shelf'), ('handsoap', 'handsoap'),
                        ('bathrobe', 'bathrobe'), ('flour', 'flour'),
                        ('towel', 'towel'), ('towel rail', 'towel rail'),
                        ('peanut oil', 'peanut oil'), ('wall hook', 'hook'),
                        ('sugar', 'sugar'), ('sink', 'sink'),
                        ('climbing shoes', 'shoes'), ('brown cap', 'cap'),
                        ('white cap', 'cap'), ('shoe cabinet', 'cabinet')]}

    # (noisy) positive (target) labels
    P = {'put': [('towel', 'towel rail'), ('flour', 'shelf'),
                 ('handsoap', 'sink'), ('sugar', 'shelf'),
                 ('peanut oil', 'shelf'), ('bathrobe', 'wall hook'),
                 ('brown cap', 'hat rack'), ('white cap', 'hat rack'),
                 ('flour', 'folding chair'), ('sugar', 'folding chair'),
                 ('peanut oil', 'folding chair')]}

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

    neuron = {
        'bias_learning': False,
        'weights_learning': False}

    subrules = []
    valid_sub_rules = []
    head_vars = (x, y)
    for i, preds_vars in enumerate(rule_vars):
        vars1, vars2, vars3 = preds_vars
        body_vars = set(vars1)
        body_vars = set.union(body_vars, set(vars2))
        body_vars = set.union(body_vars, set(vars3))
        # Filter rules who have head variables not appearing in body.
        if all([var in body_vars for var in head_vars]):
            valid_sub_rules.append(preds_vars)
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
    best_value = 0
    weighted_idx = (rule.neuron.weights == 1).nonzero(
        as_tuple=True)[0]
    for idx in weighted_idx:
        subrule_forall = rule.operands[idx]
        if subrule_forall.neuron.bounds_table[0][0] == 1:
            rule_value = prune_rules(B, P, variables, valid_sub_rules[idx])
            # Pick rule with highest agreement with facts.
            if rule_value > best_value:
                best_value = rule_value
                subrule_bidirectional = subrule_forall.operands[0]
                subrule_implication = subrule_bidirectional.Imp1
                subrule_and = subrule_implication.operands[0]
                subrule_and_vars = subrule_and.binding_str
                chosen_idx = subrule_and_vars

    assert chosen_idx in valid_rules, 'Incorrect rule.'


if __name__ == "__main__":
    test()
    print('success')
