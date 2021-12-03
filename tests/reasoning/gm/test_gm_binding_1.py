##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Predicate, TRUE, FALSE


def test():
    model = Model()
    x, y, z = map(Variable, ('x', 'y', 'z'))

    model['p2'] = Predicate('p2', arity=2)
    p2_facts = dict([
                     (('s1', 's7'), TRUE),
                     (('s1', 's6'), TRUE),
                     (('s2', 's6'), FALSE),
                     (('s3', 's7'), FALSE),
                     (('s4', 's6'), TRUE)])
    model.add_facts({'p2': p2_facts})

    model['p2a'] = Predicate('p2a', arity=2)
    p2a_facts = dict([
                      (('s1', 's7'), TRUE),
                      (('s1', 's6'), FALSE),
                      (('s2', 's5'), FALSE),
                      (('s4', 's7'), FALSE),
                      (('s3', 's7'), FALSE),
                      (('s7', 's6'), TRUE)])
    model.add_facts({'p2a': p2a_facts})

    # the unbounded case
    model['p2_and_p2a'] = And(model['p2'](x, y), model['p2a'](x, y))
    # GT = dict([
    #    (('s1', 's7'), TRUE),
    #    (('s1', 's6'), FALSE),
    #    (('s3', 's7'), FALSE),
    #    (('s2', 's6'), FALSE),
    #    (('s4', 's6'), UNKNOWN),
    #    (('s2', 's5'), FALSE),
    #    (('s4', 's7'), FALSE),
    #     (('s7', 's6'), UNKNOWN)])

    model['p2_and_p2a'].upward()
    # assert all([model['p2_and_p2a'].state(groundings=g) is GT[g]
    #            for g in GT]), "FAILED 😔"
    # assert len(model['p2_and_p2a'].state()) == len(GT), "FAILED 😔"

    # One variable bound
    model['p2_and_p2b'] = And(model['p2'](x, y), model['p2a'](x, (y, 's7')))
    # GT = dict([
    #    (('s1', 's7'), TRUE),
    #    (('s3', 's7'), FALSE),
    #    (('s4', 's7'), FALSE)])

    model['p2_and_p2b'].upward()
    # assert all([model['p2_and_p2b'].state(groundings=g) is GT[g]
    #            for g in GT]), "FAILED 😔"
    # assert len(model['p2_and_p2b'].state()) == len(GT), "FAILED 😔"

    # One variable bound to a list
    model['p2_and_p2c'] = \
        And(model['p2'](x, y), model['p2a']((x, ['s1', 's2']), (y, 's6')))

    # GT = dict([
    #    (('s1', 's6'), FALSE),
    #    (('s2', 's6'), FALSE)])
    model['p2_and_p2c'].upward()
    # assert all([model['p2_and_p2c'].state(groundings=g) is GT[g]
    #            for g in GT]), "FAILED 😔"
    # assert len(model['p2_and_p2c'].state()) == len(GT), "FAILED 😔"

    # 1 variable vs 2 variables bound
    model['p1'] = Predicate('p1')
    model.add_facts({'p1': {
        's1': TRUE,
        's2': TRUE,
        's3': TRUE,
        's4': FALSE,
        's10': FALSE}})
    model['p1_and_p2'] = And(model['p1']((x, ['s1', 's2'])), model['p2'](x, y))
    model['p1_and_p2'].upward()
    # GT = dict([
    #    (('s1', 's6'), TRUE),
    #    (('s2', 's6'), FALSE),
    #    (('s1', 's7'), TRUE),
    #    (('s2', 's5'), UNKNOWN)])
    # assert all([model['p1_and_p2'].state(groundings=g) is GT[g]
    #            for g in GT]), "FAILED 😔"
    # assert len(model['p1_and_p2'].state()) == len(GT), "FAILED 😔"

    # 2 variable vs 3 variables bound
    model['p3'] = Predicate('p3', arity=3)
    model.add_facts({'p3': {
        ('s1', 's5', 's3'): TRUE,
        ('s1', 's4', 's7'): TRUE,
        ('s1', 's8', 's3'): FALSE,
        ('s2', 's8', 's6'): TRUE,
        ('s4', 's6', 's8'): FALSE}})

    model['p2_and_p3'] = And(model['p2'](x, y), model['p3'](x, (z, 's4'), y))
    model['p2_and_p3'].upward()
    # GT = dict([
    #    (('s1', 's7', 's4'), TRUE)])
    # assert all([model['p2_and_p3'].state(groundings=g) is GT[g]
    #            for g in GT]), "FAILED 😔"
    # assert len(model['p2_and_p3'].state()) == len(GT), "FAILED 😔"

    # 2 variable vs 2 variable reversed, bound
    model['p2r'] = Predicate('p2r', arity=2)
    model.add_facts({'p2r': {
        ('s6', 's2'): TRUE,
        ('s7', 's1'): FALSE}})
    model['p2_and_p2r'] = \
        And(model['p2'](x, y), model['p2r']((y, ['s7', 's3']), x))
    model['p2_and_p2r'].upward()
    # GT = dict([
    #    (('s1', 's7'), FALSE)])
    # assert all([model['p2_and_p2r'].state(groundings=g) is GT[g]
    #            for g in GT]), "FAILED 😔"
    # assert len(model['p2_and_p2r'].state()) == len(GT), "FAILED 😔"


if __name__ == "__main__":
    test()
    print('success')
