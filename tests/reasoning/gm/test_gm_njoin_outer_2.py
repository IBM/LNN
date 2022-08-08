##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Predicate, Fact, Join


def test():
    join = Join.OUTER
    model = Model()
    x, y, z, a, b = map(Variable, ("x", "y", "z", "a", "b"))

    # TEST 1

    # This is the normal 2 var vs 2 var ; should go thru the memory join
    p2, p2a = model.add_predicates(2, "p2", "p2a")
    model.add_data(
        {
            p2: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE},
            p2a: {("y1", "z1"): Fact.TRUE, ("y3", "z2"): Fact.TRUE},
        }
    )

    # print("Predicates before outer Join")
    # 'p2'.print()
    # 'p2a'.print()

    # GT_i = dict([
    #    (('x1', 'y1', 'z1'), Fact.TRUE)])

    GT_o = dict(
        [
            (("x1", "y1", "z1"), Fact.TRUE),
            (("x1", "y1", "z2"), Fact.UNKNOWN),
            (("x1", "y3", "z2"), Fact.UNKNOWN),
            (("x2", "y2", "z1"), Fact.UNKNOWN),
            (("x2", "y1", "z1"), Fact.UNKNOWN),
            (("x2", "y2", "z2"), Fact.UNKNOWN),
            (("x2", "y3", "z2"), Fact.UNKNOWN),
        ]
    )
    p2_and_p2a = And(p2(x, y), p2a(y, z), join=join)
    model.add_knowledge(p2_and_p2a)
    p2_and_p2a.upward()

    # print("Predicates After Join")
    # 'p2'.print()
    # 'p2a'.print()
    # 'p2_and_p2a'.print()

    # for g in GT_o :
    #    print(g, 'p2_and_p2a'.state(groundings=g), GT_o[g])

    assert all([p2_and_p2a.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"
    assert len(p2_and_p2a.state()) == len(GT_o), "FAILED 😔"

    # TEST 2
    model = Model()

    t2_p3 = model.add_predicates(3, "t2_p3")
    t2_p2 = model.add_predicates(2, "t2_p2")
    model.add_data(
        {t2_p3: {("x1", "y1", "z1"): Fact.TRUE, ("x3", "y3", "z3"): Fact.TRUE}}
    )

    # model.add_data({'t2_p2': {
    #    ('y1', 'z1'): Fact.TRUE,
    #    ('y2', 'z2'): Fact.TRUE}})

    GT_o = dict(
        [
            (("x1", "y1", "z1"), Fact.TRUE),
            (("x1", "y2", "z2"), Fact.UNKNOWN),
            (("x3", "y1", "z1"), Fact.UNKNOWN),
            (("x3", "y3", "z3"), Fact.UNKNOWN),
            (("x3", "y2", "z2"), Fact.UNKNOWN),
        ]
    )
    t2_p3_and_t2_p2 = And(t2_p3(x, y, z), t2_p2(y, z), join=join)
    model.add_knowledge(t2_p3_and_t2_p2)
    t2_p3_and_t2_p2.upward()
    t2_p3_and_t2_p2.print()

    # assert all([t2_p3_and_t2_p2'.state(groundings=g) is GT_o[g]
    #            for g in GT_o]), "FAILED 😔"
    # assert len(t2_p3_and_t2_p2'.state()) == len(GT_o), "FAILED 😔"

    # TEST 3
    model = Model()
    t2_p3 = model.add_predicates(3, "t2_p3")
    model.add_data(
        {t2_p3: {("x1", "y1", "z1"): Fact.TRUE, ("x3", "y3", "z3"): Fact.TRUE}}
    )

    t2_p2 = model.add_predicates(2, "t2_p2")
    # model.add_data({'t2_p2': {
    #    ('y1', 'z1'): Fact.TRUE,
    #    ('y2', 'z2'): Fact.TRUE}})
    t3_p1 = model.add_predicates(1, "t3_p1")
    # model.add_data({'t3_p1': {
    #    ('z1'): Fact.TRUE,
    #    ('z4'): Fact.TRUE}})
    t2_p3_and_t2_p2_t3_p1 = And(t2_p3(x, y, z), t2_p2(y, z), t3_p1(z), join=join)
    model.add_knowledge(t2_p3_and_t2_p2_t3_p1)
    t2_p3_and_t2_p2_t3_p1.upward()
    t2_p3_and_t2_p2_t3_p1.print()

    # GT_inner = dict([
    #    (('x1', 'y1', 'z1'), Fact.TRUE)])

    GT_o = dict(
        [
            (("x1", "y1", "z1"), Fact.TRUE),
            (("x1", "y2", "z2"), Fact.UNKNOWN),
            (("x3", "y1", "z1"), Fact.UNKNOWN),
            (("x3", "y3", "z3"), Fact.UNKNOWN),
            (("x3", "y2", "z2"), Fact.UNKNOWN),
            (("x1", "y2", "z1"), Fact.UNKNOWN),
            (("x3", "y3", "z1"), Fact.UNKNOWN),
            (("x3", "y2", "z1"), Fact.UNKNOWN),
            (("x1", "y1", "z4"), Fact.UNKNOWN),
            (("x1", "y2", "z4"), Fact.UNKNOWN),
            (("x3", "y1", "z4"), Fact.UNKNOWN),
            (("x3", "y3", "z4"), Fact.UNKNOWN),
            (("x3", "y2", "z4"), Fact.UNKNOWN),
        ]
    )

    # for g in GT_o:
    #    print(g, t2_p3_and_t2_p2_t3_p1'.state(groundings=g), GT_o[g])

    # assert all([t2_p3_and_t2_p2_t3_p1'.state(groundings=g) is GT_o[g]
    #            for g in GT_o]), "FAILED 😔"
    # assert len(t2_p3_and_t2_p2_t3_p1'.state())
    #  == len(GT_o), "FAILED 😔"

    # TEST 4
    model = Model()
    model.add_knowledge(Predicate("t2_p3", arity=3))
    # model.add_data({'t2_p3': {
    #    ('x1', 'y1', 'z1'): Fact.TRUE,
    #    ('x3', 'y3', 'z3'): Fact.TRUE}})

    t2_p2 = model.add_predicates(2, "t2_p2")
    t4_p1 = model.add_predicates(1, "t4_p1")
    model.add_data(
        {
            t2_p2: {("y1", "z1"): Fact.TRUE, ("y2", "z2"): Fact.TRUE},
            t4_p1: {"x1": Fact.TRUE, "x4": Fact.TRUE},
        }
    )
    t2_p3_and_t2_p2_t4_p1 = And(t2_p3(x, y, z), t2_p2(y, z), t4_p1(x), join=join)
    model.add_knowledge(t2_p3_and_t2_p2_t4_p1)
    t2_p3_and_t2_p2_t4_p1.upward()
    t2_p3_and_t2_p2_t4_p1.print()

    # for g in GT_o:
    #    print(g, t2_p3_and_t2_p2_t4_p1'.state(groundings=g), GT_o[g])

    # GT_i = dict([
    #    (('x1', 'y1', 'z1'), Fact.TRUE)])

    GT_o = dict(
        [
            (("x1", "y1", "z1"), Fact.TRUE),
            (("x1", "y2", "z2"), Fact.UNKNOWN),
            (("x3", "y1", "z1"), Fact.UNKNOWN),
            (("x3", "y3", "z3"), Fact.UNKNOWN),
            (("x3", "y2", "z2"), Fact.UNKNOWN),
            (("x1", "y3", "z3"), Fact.UNKNOWN),
            (("x4", "y1", "z1"), Fact.UNKNOWN),
            (("x4", "y3", "z3"), Fact.UNKNOWN),
            (("x4", "y2", "z2"), Fact.UNKNOWN),
        ]
    )

    # assert all([t2_p3_and_t2_p2_t4_p1'.state(groundings=g)
    #            is GT_o[g] for g in GT_o]), "FAILED 😔"
    # assert len(t2_p3_and_t2_p2_t4_p1'.state()) ==
    # len(GT_o), "FAILED 😔"

    # TEST 5
    model = Model()
    t2_p3 = model.add_predicates(3, "t2_p3")
    model.add_data(
        {t2_p3: {("x1", "y1", "z1"): Fact.TRUE, ("x3", "y3", "z3"): Fact.TRUE}}
    )

    t2_p2, t5_p2 = model.add_predicates(2, "t2_p2", "t5_p2")
    model.add_data(
        {
            t2_p2: {("y1", "z1"): Fact.TRUE, ("y2", "z2"): Fact.TRUE},
            t5_p2: {("a1", "b1"): Fact.TRUE, ("a2", "b2"): Fact.TRUE},
        }
    )
    t2_p3_and_t2_p2_t5_p2 = And(t2_p3(x, y, z), t2_p2(y, z), t5_p2(a, b), join=join)
    model.add_knowledge(t2_p3_and_t2_p2_t5_p2)

    t2_p3_and_t2_p2_t5_p2.upward()
    # t2_p3_and_t2_p2_t5_p2'.print()

    # GT_inner = dict([
    #    (('x1', 'y1', 'z1','a1','b1'), Fact.TRUE),
    #   (('x1', 'y1', 'z1','a2','b2'), Fact.TRUE)])

    GT_o = dict(
        [
            (("x1", "y1", "z1", "a1", "b1"), Fact.TRUE),
            (("x1", "y2", "z2", "a1", "b1"), Fact.UNKNOWN),
            (("x3", "y1", "z1", "a1", "b1"), Fact.UNKNOWN),
            (("x3", "y3", "z3", "a1", "b1"), Fact.UNKNOWN),
            (("x3", "y2", "z2", "a1", "b1"), Fact.UNKNOWN),
            (("x1", "y1", "z1", "a2", "b2"), Fact.TRUE),
            (("x1", "y2", "z2", "a2", "b2"), Fact.UNKNOWN),
            (("x3", "y1", "z1", "a2", "b2"), Fact.UNKNOWN),
            (("x3", "y3", "z3", "a2", "b2"), Fact.UNKNOWN),
            (("x3", "y2", "z2", "a2", "b2"), Fact.UNKNOWN),
        ]
    )

    assert all(
        [t2_p3_and_t2_p2_t5_p2.state(groundings=g) is GT_o[g] for g in GT_o]
    ), "FAILED 😔"
    assert len(t2_p3_and_t2_p2_t5_p2.state()) == len(GT_o), "FAILED 😔"


if __name__ == "__main__":
    test()
    print("success")
