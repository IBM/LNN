##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (Model, And, Variable, Predicate, TRUE, FALSE,
                 UNKNOWN)


def test():
    model = Model()
    x, y, z = map(Variable, ('x', 'y', 'z'))

    # This is the normal 2 var vs 2 var ; should go thru the memory join
    model['p2'] = Predicate('p2', arity=2)
    model.add_facts({'p2': {
        ('s1', 's7'): TRUE,
        ('s1', 's6'): TRUE,
        ('s2', 's6'): FALSE,
        ('s3', 's7'): FALSE,
        ('s4', 's6'): TRUE}})

    model['p2a'] = Predicate('p2a', arity=2)
    model.add_facts({'p2a': {
        ('s1', 's7'): TRUE,
        ('s1', 's6'): FALSE,
        ('s2', 's5'): FALSE,
        ('s4', 's7'): FALSE,
        ('s7', 's6'): TRUE}})

    GT = dict([
       (('s1', 's7'), TRUE),
       (('s1', 's6'), FALSE),
       (('s2', 's6'), FALSE),
       (('s3', 's7'), FALSE),
       (('s4', 's6'), UNKNOWN),
       (('s2', 's5'), FALSE),
       (('s4', 's7'), FALSE),
       (('s7', 's6'), UNKNOWN)])

    model['p2_and_p2a'] = And(model['p2'](x, y), model['p2a'](x, y))
    model['p2_and_p2a'].upward()
    assert all([model['p2_and_p2a'].state(groundings=g) is GT[g]
                for g in GT]), "FAILED 😔"
    assert len(model['p2_and_p2a'].state()) == len(GT), "FAILED 😔"

    # 1 variable vs 2 variables

    model = Model()  # Reset the model for each new test.

    model['p2'] = Predicate('p2', arity=2)
    model.add_facts({'p2': {
        ('s1', 's7'): TRUE,
        ('s1', 's6'): TRUE,
        ('s2', 's6'): FALSE,
        ('s3', 's7'): FALSE,
        ('s4', 's6'): TRUE}})

    model['p1'] = Predicate('p1')
    model.add_facts({'p1': {
        's1': TRUE,
        's2': TRUE,
        's3': TRUE,
        's4': FALSE,
        's10': FALSE}})

    model['p1_and_p2'] = And(model['p1'](x), model['p2'](x, y))
    model['p1_and_p2'].upward()

    GT = dict([
        (('s1', 's6'), TRUE),
        (('s3', 's7'), FALSE),
        (('s2', 's6'), FALSE),
        (('s1', 's7'), TRUE),
        (('s4', 's6'), FALSE)])

    assert all([model['p1_and_p2'].state(groundings=g) is GT[g]
                for g in GT]), "FAILED 😔"
    assert len(model['p1_and_p2'].state()) == len(GT), "FAILED 😔"

    # 2 variable vs 3 variables
    model['p3'] = Predicate('p3', arity=3)
    model.add_facts({'p3': {
        ('s1', 's5', 's3'): TRUE,
        ('s1', 's4', 's7'): TRUE,
        ('s1', 's8', 's3'): FALSE,
        ('s2', 's8', 's6'): TRUE,
        ('s4', 's6', 's8'): FALSE}})

    model['p2_and_p3'] = And(model['p2'](x, y), model['p3'](x, z, y))
    model['p2_and_p3'].upward()

    GT = dict([
        (('s2', 's6', 's8'), FALSE),
        (('s1', 's7', 's4'), TRUE)])
    assert all(model['p2_and_p3'].state(groundings=g) is GT[g]
               for g in GT), "FAILED 😔"
    assert len(model['p2_and_p3'].state()) == len(GT), "FAILED 😔"

    # 1 vs 2 vs 3
    model['p1_and_p2_and_p3'] = And(model['p1'](x),
                                    model['p2'](x, y),
                                    model['p3'](x, z, y))
    model['p1_and_p2_and_p3'].upward()

    GT = dict([
        (('s2', 's6', 's8'), FALSE),
        (('s1', 's7', 's4'), TRUE)])
    assert all(model['p1_and_p2_and_p3'].state(groundings=g) is GT[g]
               for g in GT), "FAILED 😔"
    assert len(model['p1_and_p2_and_p3'].state()) == len(GT), "FAILED 😔"

    # 2 variable vs 2 variable reversed
    model['p2r'] = Predicate('p2r', arity=2)
    model.add_facts({'p2r': {
        ('s6', 's2'): TRUE,
        ('s7', 's1'): FALSE}})
    model['p2_and_p2r'] = And(model['p2'](x, y), model['p2r'](y, x))
    model['p2_and_p2r'].upward()
    GT = dict([
        (('s2', 's6'), FALSE),
        (('s1', 's7'), FALSE)])
    assert all([model['p2_and_p2r'].state(groundings=g) is GT[g]
               for g in GT]), "FAILED 😔"
    assert len(model['p2_and_p2r'].state()) == len(GT), "FAILED 😔"


if __name__ == "__main__":
    test()
    print('success')
