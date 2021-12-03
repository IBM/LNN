##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Predicate, Fact, Join


def test():
    join = Join.OUTER_PRUNED
    model = Model()
    x, y, z, a, b = map(Variable, ('x', 'y', 'z', 'a', 'b'))

    # TEST 1

    # This is the normal 2 var vs 2 var ; should go thru the memory join
    model['p2'] = Predicate('p2', arity=2)
    # model.add_facts({'p2': {
    #    ('x1', 'y1'): Fact.TRUE,
    #    ('x2', 'y2'): Fact.TRUE}})

    model['p2a'] = Predicate('p2a', arity=2)
    model.add_facts({'p2a': {
        ('y1', 'z1'): Fact.TRUE,
        ('y3', 'z2'): Fact.TRUE}})

    v1_before = model['p2'].state()
    v2_before = model['p2a'].state()

    # GT_i = dict([
    #    (('x1', 'y1', 'z1'), Fact.TRUE)])

    GT_o = dict([
        (('x1', 'y1', 'z1'), Fact.TRUE),
        (('x1', 'y1', 'z2'), Fact.TRUE),
        (('x1', 'y3', 'z2'), Fact.TRUE),
        (('x2', 'y2', 'z1'), Fact.TRUE),
        (('x2', 'y1', 'z1'), Fact.TRUE),
        (('x2', 'y2', 'z2'), Fact.TRUE),
        (('x2', 'y3', 'z2'), Fact.TRUE)])

    model['p2_and_p2a'] = And(model['p2'](x, y), model['p2a'](y, z),
                              join=join)
    # model['p2_and_p2a'].upward()
    # model['p2_and_p2a'].print()

    v1_after = model['p2'].state()
    v2_after = model['p2a'].state()

    # assert (v1_after == v1_before),  "FAILED 😔"
    # assert (v2_after == v2_before),  "FAILED 😔"
    # assert all([model['p2_and_p2a'].state(groundings=g) is GT_o[g]
    #            for g in GT_o]), "FAILED 😔"
    # assert len(model['p2_and_p2a'].state()) == len(GT_o), "FAILED 😔"

    # TEST 2
    model = Model()

    model['t2_p3'] = Predicate('t2_p3', arity=3)
    model.add_facts({'t2_p3': {
        ('x1', 'y1', 'z1'): Fact.TRUE,
        ('x3', 'y3', 'z3'): Fact.TRUE}})

    model['t2_p2'] = Predicate('t2_p2', arity=2)
    # model.add_facts({'t2_p2': {
    #    ('y1', 'z1'): Fact.TRUE,
    #    ('y2', 'z2'): Fact.TRUE}})

    GT_o = dict([
        (('x1', 'y1', 'z1'), Fact.TRUE),
        (('x1', 'y2', 'z2'), Fact.TRUE),
        (('x3', 'y1', 'z1'), Fact.TRUE),
        (('x3', 'y3', 'z3'), Fact.TRUE),
        (('x3', 'y2', 'z2'), Fact.TRUE)])

    model['t2_p3_and_t2_p2'] = And(model['t2_p3'](x, y, z),
                                   model['t2_p2'](y, z),
                                   join=join)

    v1_before = model['t2_p3'].state()
    v2_before = model['t2_p2'].state()

    model['t2_p3_and_t2_p2'].upward()
    model['t2_p3_and_t2_p2'].print()

    v1_after = model['t2_p3'].state()
    v2_after = model['t2_p2'].state()

    # assert (v1_after == v1_before),  "FAILED 😔"
    # assert (v2_after == v2_before),  "FAILED 😔"

    # assert all([model['t2_p3_and_t2_p2'].state(groundings=g) is GT_o[g]
    #            for g in GT_o]), "FAILED 😔"
    # assert len(model['t2_p3_and_t2_p2'].state()) == len(GT_o), "FAILED 😔"

    # TEST 3
    model = Model()
    model['t2_p3'] = Predicate('t2_p3', arity=3)
    model.add_facts({'t2_p3': {
        ('x1', 'y1', 'z1'): Fact.TRUE,
        ('x3', 'y3', 'z3'): Fact.TRUE}})

    model['t2_p2'] = Predicate('t2_p2', arity=2)
    model.add_facts({'t2_p2': {
        ('y1', 'z1'): Fact.TRUE,
        ('y2', 'z2'): Fact.TRUE}})
    model['t3_p1'] = Predicate('t3_p1')
    model.add_facts({'t3_p1': {
        ('z1'): Fact.TRUE,
        ('z4'): Fact.TRUE}})
    model['t2_p3_and_t2_p2_t3_p1'] = And(model['t2_p3'](x, y, z),
                                         model['t2_p2'](y, z),
                                         model['t3_p1'](z),
                                         join=join)

    v1_before = model['t2_p3'].state()
    v2_before = model['t2_p2'].state()
    v3_before = model['t3_p1'].state()

    model['t2_p3_and_t2_p2_t3_p1'].upward()

    v1_after = model['t2_p3'].state()
    v2_after = model['t2_p2'].state()
    v3_after = model['t3_p1'].state()

    GT_o = dict([
        (('x1', 'y1', 'z1'), Fact.TRUE),
        (('x1', 'y2', 'z2'), Fact.TRUE),
        (('x3', 'y1', 'z1'), Fact.TRUE),
        (('x3', 'y3', 'z3'), Fact.TRUE),
        (('x3', 'y2', 'z2'), Fact.TRUE),
        (('x1', 'y2', 'z1'), Fact.TRUE),
        (('x3', 'y3', 'z1'), Fact.TRUE),
        (('x3', 'y2', 'z1'), Fact.TRUE),
        (('x1', 'y1', 'z4'), Fact.TRUE),
        (('x1', 'y2', 'z4'), Fact.TRUE),
        (('x3', 'y1', 'z4'), Fact.TRUE),
        (('x3', 'y3', 'z4'), Fact.TRUE),
        (('x3', 'y2', 'z4'), Fact.TRUE)])

    assert (v1_after == v1_before),  "FAILED 😔"
    assert (v2_after == v2_before),  "FAILED 😔"
    assert (v3_after == v3_before),  "FAILED 😔"

    assert all([model['t2_p3_and_t2_p2_t3_p1'].state(groundings=g) is GT_o[g]
                for g in GT_o]), "FAILED 😔"
    assert len(model['t2_p3_and_t2_p2_t3_p1'].state()) == len(GT_o), "FAILED 😔"

    # TEST 4
    model = Model()
    model['t2_p3'] = Predicate('t2_p3', arity=3)
    model.add_facts({'t2_p3': {
        ('x1', 'y1', 'z1'): Fact.TRUE,
        ('x3', 'y3', 'z3'): Fact.TRUE}})

    model['t2_p2'] = Predicate('t2_p2', arity=2)
    model.add_facts({'t2_p2': {
        ('y1', 'z1'): Fact.TRUE,
        ('y2', 'z2'): Fact.TRUE}})

    model['t4_p1'] = Predicate('t4_p1')
    model.add_facts({'t4_p1': {
        ('x1'): Fact.TRUE,
        ('x4'): Fact.TRUE}})

    model['t2_p3_and_t2_p2_t4_p1'] = And(model['t2_p3'](x, y, z),
                                         model['t2_p2'](y, z),
                                         model['t4_p1'](x),
                                         join=join)

    v1_before = model['t2_p3'].state()
    v2_before = model['t2_p2'].state()
    v3_before = model['t4_p1'].state()

    model['t2_p3_and_t2_p2_t4_p1'].upward()

    v1_after = model['t2_p3'].state()
    v2_after = model['t2_p2'].state()
    v3_after = model['t4_p1'].state()

    GT_o = dict([
        (('x1', 'y1', 'z1'), Fact.TRUE),
        (('x1', 'y2', 'z2'), Fact.TRUE),
        (('x3', 'y1', 'z1'), Fact.TRUE),
        (('x3', 'y3', 'z3'), Fact.TRUE),
        (('x3', 'y2', 'z2'), Fact.TRUE),
        (('x1', 'y3', 'z3'), Fact.TRUE),
        (('x4', 'y1', 'z1'), Fact.TRUE),
        (('x4', 'y3', 'z3'), Fact.TRUE),
        (('x4', 'y2', 'z2'), Fact.TRUE)])

    assert (v1_after == v1_before),  "FAILED 😔"
    assert (v2_after == v2_before),  "FAILED 😔"
    assert (v3_after == v3_before),  "FAILED 😔"

    assert all([model['t2_p3_and_t2_p2_t4_p1'].state(groundings=g)
                is GT_o[g] for g in GT_o]), "FAILED 😔"
    assert len(model['t2_p3_and_t2_p2_t4_p1'].state()) == len(GT_o), "FAILED 😔"

    # TEST 5
    model = Model()
    model['t2_p3'] = Predicate('t2_p3', arity=3)
    model.add_facts({'t2_p3': {
        ('x1', 'y1', 'z1'): Fact.TRUE,
        ('x3', 'y3', 'z3'): Fact.TRUE}})

    model['t2_p2'] = Predicate('t2_p2', arity=2)
    model.add_facts({'t2_p2': {
        ('y1', 'z1'): Fact.TRUE,
        ('y2', 'z2'): Fact.TRUE}})

    model['t5_p2'] = Predicate('t5_p2', arity=2)
    model.add_facts({'t5_p2': {
        ('a1', 'b1'): Fact.TRUE,
        ('a2', 'b2'): Fact.TRUE}})

    model['t2_p3_and_t2_p2_t5_p2'] = And(model['t2_p3'](x, y, z),
                                         model['t2_p2'](y, z),
                                         model['t5_p2'](a, b),
                                         join=join)

    v1_before = model['t2_p3'].state()
    v2_before = model['t2_p2'].state()
    v3_before = model['t5_p2'].state()

    model['t2_p3_and_t2_p2_t5_p2'].upward()

    v1_after = model['t2_p3'].state()
    v2_after = model['t2_p2'].state()
    v3_after = model['t5_p2'].state()

    GT_o = dict([
        (('x1', 'y1', 'z1', 'a1', 'b1'), Fact.TRUE),
        (('x1', 'y2', 'z2', 'a1', 'b1'), Fact.TRUE),
        (('x3', 'y1', 'z1', 'a1', 'b1'), Fact.TRUE),
        (('x3', 'y3', 'z3', 'a1', 'b1'), Fact.TRUE),
        (('x3', 'y2', 'z2', 'a1', 'b1'), Fact.TRUE),
        (('x1', 'y1', 'z1', 'a2', 'b2'), Fact.TRUE),
        (('x1', 'y2', 'z2', 'a2', 'b2'), Fact.TRUE),
        (('x3', 'y1', 'z1', 'a2', 'b2'), Fact.TRUE),
        (('x3', 'y3', 'z3', 'a2', 'b2'), Fact.TRUE),
        (('x3', 'y2', 'z2', 'a2', 'b2'), Fact.TRUE)])

    assert (v1_after == v1_before),  "FAILED 😔"
    assert (v2_after == v2_before),  "FAILED 😔"
    assert (v3_after == v3_before),  "FAILED 😔"

    assert all([model['t2_p3_and_t2_p2_t5_p2'].state(groundings=g)
                is GT_o[g] for g in GT_o]), "FAILED 😔"
    assert len(model['t2_p3_and_t2_p2_t5_p2'].state()) == len(GT_o), "FAILED 😔"


if __name__ == "__main__":
    test()
    print('success')
