##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import itertools
import random
from lnn import (Model, World, And, Bidirectional, ForAll, Variable,
                 TRUE, FALSE, UPWARD)


def test_1():
    """
    In this test all positive examples are provided.
    :return:
    """

    # background data (features)

    B = ['isFather', [('a', 'b'), ('b', 'c'), ('d', 'e'),
                      ('f', 'g'), ('e', 'h'), ('g', 'a'),
                      ('d', 'f'), ('a', 'j'), ('g', 'i')]]

    # positive (target) labels for isGrandFather(x,y)

    P1 = ['isGrandFather', [('a', 'c'), ('g', 'b'), ('f', 'a'),
                            ('d', 'g'), ('d', 'h'), ('g', 'j'),
                            ('f', 'i')]]

    # positive (target) labels for isSibling(x,y)
    P2 = ['isSibling', [('j', 'b'), ('b', 'j'), ('a', 'i'),
                        ('i', 'a'), ('f', 'e'), ('e', 'f')]]

    # The rule templates are
    #   isGrandFather(x,y) -> isFather(*,*) ∧ isFather(*,*) [x,y,z]
    #   isSibling(x,y) <-> (isFather(*,*) ∧ isFather(*,*)) [x,y,z]

    # The rules learned are:
    #   isGrandFather(x,y):- isFather(x,z) ∧ isFather(z,y)
    #   isGrandFather(x,y) <-> (isFather(x,z) ∧ isFather(z,y))

    #   isSibling(x,y):- isFather(z,x) ∧ isFather(z,y)
    #   isSibling(x,y) <-> (isFather(z,x) ∧ isFather(z,y))

    for target in [(P1, ['x, z', 'z, y']),
                   (P2, ['z, x', 'z, y'])]:
        model = Model()
        b, p = model.add_predicates(2, B[0], target[0][0], world=World.FALSE)
        model.add_facts({
            b.name: {pair: TRUE for pair in B[1]},
            p.name: {pair: TRUE for pair in target[0][1]}})

        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        variables = [x, y, z]
        pairs = itertools.permutations(variables, 2)  # e.g. [(x, y), ...]
        pairs = list(pairs)

        # e.g. [((x, y), (x, z)), ...]
        rule_vars = itertools.combinations(pairs, 2)
        rule_vars = list(rule_vars)

        neuron = {
            'bias_learning': False,
            'weights_learning': False}

        subrules = []
        for i, preds_vars in enumerate(rule_vars):
            vars1, vars2 = preds_vars
            model[str(i)] = ForAll(Bidirectional(And(b(*vars1), b(*vars2),
                                                     neuron=neuron),
                                                 p(x, y),
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
        assert chosen_idx[0] == target[1], (
                'expected ' + str(target[1]) +
                ' got ' + str(chosen_idx[0]))


def test_2():
    """
    In this test all positive examples are provided and the rest are assumed
    negative.
    :return:
    """

    # Constants
    C = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}
    # Binary grounding pairs
    groundings = itertools.permutations(C, 2)
    groundings = list(groundings)

    # background data (features)

    B = ['isFather', [('a', 'b'), ('b', 'c'), ('d', 'e'),
                      ('f', 'g'), ('e', 'h'), ('g', 'a'),
                      ('d', 'f'), ('a', 'j'), ('g', 'i')]]

    # positive (target) labels for isGrandFather(x,y)

    P1 = ['isGrandFather', [('a', 'c'), ('g', 'b'), ('f', 'a'),
                            ('d', 'g'), ('d', 'h'), ('g', 'j'),
                            ('f', 'i')]]

    # positive (target) labels for isSibling(x,y)
    P2 = ['isSibling', [('j', 'b'), ('b', 'j'), ('a', 'i'),
                        ('i', 'a'), ('f', 'e'), ('e', 'f')]]

    # The rule templates are
    #   isGrandFather(x,y) -> isFather(*,*) ∧ isFather(*,*) [x,y,z]
    #   isSibling(x,y) <-> (isFather(*,*) ∧ isFather(*,*)) [x,y,z]

    # The rules learned are:
    #   isGrandFather(x,y):- isFather(x,z) ∧ isFather(z,y)
    #   isGrandFather(x,y) <-> (isFather(x,z) ∧ isFather(z,y))

    #   isSibling(x,y):- isFather(z,x) ∧ isFather(z,y)
    #   isSibling(x,y) <-> (isFather(z,x) ∧ isFather(z,y))

    for target in [(P1, ['x, z', 'z, y']),
                   (P2, ['z, x', 'z, y'])]:
        model = Model()
        b, p = model.add_predicates(2, B[0], target[0][0], world=World.FALSE)
        model.add_facts({
            b.name: {pair: TRUE for pair in B[1]},
            p.name: {pair: TRUE for pair in target[0][1]}})

        # Negative facts
        model.add_facts({
            p.name: {pair: FALSE for pair in groundings
                     if pair not in target[0][1]}})

        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        variables = [x, y, z]
        pairs = itertools.permutations(variables, 2)  # e.g. [(x, y), ...]
        pairs = list(pairs)

        # e.g. [((x, y), (x, z)), ...]
        rule_vars = itertools.combinations(pairs, 2)
        rule_vars = list(rule_vars)

        neuron = {
            'bias_learning': False,
            'weights_learning': False}

        subrules = []
        for i, preds_vars in enumerate(rule_vars):
            vars1, vars2 = preds_vars
            model[str(i)] = ForAll(Bidirectional(And(b(*vars1), b(*vars2),
                                                     neuron=neuron),
                                                 p(x, y),
                                                 neuron=neuron),
                                   fully_grounded=True)
            subrules.append(model[str(i)])

        rule = model['rule'] = And(*subrules, world=World.AXIOM)

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
        assert chosen_idx[0] == target[1], (
                'expected ' + str(target[1]) +
                ' got ' + str(chosen_idx[0]) +
                ' from ' + str(chosen_idx))


def test_3():
    """
    In this test all positive examples are provided and the rest are assumed
    negative.
    :return:
    """

    # Constants
    C = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}
    # Binary grounding pairs
    groundings = itertools.permutations(C, 2)
    groundings = list(groundings)

    # background data (features)

    B = ['isFather', [('a', 'b'), ('b', 'c'), ('d', 'e'),
                      ('f', 'g'), ('e', 'h'), ('g', 'a'),
                      ('d', 'f'), ('a', 'j'), ('g', 'i')]]

    # positive (target) labels for isGrandFather(x,y)

    P1 = ['isGrandFather', [('a', 'c'), ('g', 'b'), ('f', 'a'),
                            ('d', 'g'), ('d', 'h'), ('g', 'j'),
                            ('f', 'i')]]

    # positive (target) labels for isSibling(x,y)
    P2 = ['isSibling', [('j', 'b'), ('b', 'j'), ('a', 'i'),
                        ('i', 'a'), ('f', 'e'), ('e', 'f')]]

    # The rule templates are
    #   isGrandFather(x,y) -> isFather(*,*) ∧ isFather(*,*) [x,y,z]
    #   isSibling(x,y) <-> (isFather(*,*) ∧ isFather(*,*)) [x,y,z]

    # The rules learned are:
    #   isGrandFather(x,y):- isFather(x,z) ∧ isFather(z,y)
    #   isGrandFather(x,y) <-> (isFather(x,z) ∧ isFather(z,y))

    #   isSibling(x,y):- isFather(z,x) ∧ isFather(z,y)
    #   isSibling(x,y) <-> (isFather(z,x) ∧ isFather(z,y))

    for target in [(P1, ['x, z', 'z, y']),
                   (P2, ['z, x', 'z, y'])]:
        model = Model()
        b, p = model.add_predicates(2, B[0], target[0][0], world=World.FALSE)
        model.add_facts({
            b.name: {pair: TRUE for pair in B[1][0:-2]},
            p.name: {pair: TRUE for pair in target[0][1]}})

        # Negative facts
        neg_facts = [pair for pair in groundings if pair not in target[0][1]]
        model.add_facts({
            p.name: {pair: FALSE for pair in random.choices(neg_facts, k=7)}})

        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        variables = [x, y, z]
        pairs = itertools.permutations(variables, 2)  # e.g. [(x, y), ...]
        pairs = list(pairs)

        # e.g. [((x, y), (x, z)), ...]
        # For this specific example, the order of predicates in the body
        # does not matter. In general we should use permutations because
        # order matters.
        rule_vars = itertools.combinations(pairs, 2)
        rule_vars = list(rule_vars)

        neuron = {
            'bias_learning': False,
            'weights_learning': False}

        subrules = []
        for i, preds_vars in enumerate(rule_vars):
            vars1, vars2 = preds_vars
            model[str(i)] = ForAll(Bidirectional(And(b(*vars1), b(*vars2),
                                                     neuron=neuron),
                                                 p(x, y),
                                                 neuron=neuron),
                                   fully_grounded=True)
            subrules.append(model[str(i)])

        rule = model['rule'] = And(*subrules, world=World.AXIOM)

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

        rule.print(params=True)
        num_chosen = len(chosen_idx)
        assert num_chosen == 1, (
                f'expected 1 got {num_chosen} in {chosen_idx}')
        assert chosen_idx[0] == target[1], (
                'expected ' + str(target[1]) +
                ' got ' + str(chosen_idx[0]) +
                ' from ' + str(chosen_idx))


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    print('success')
