##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Predicate, Predicates, Fact


def test():
    model = Model()
    x, y, z, a, b = map(Variable, ("x", "y", "z", "a", "b"))

    # TEST 1

    # This is the normal 2 var vs 2 var ; should go thru the memory join
    p2 = Predicate("p2", arity=2, model=model)
    p2.add_data({("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE})

    p2a = Predicate("p2a", arity=2, model=model)
    p2a.add_data({("y1", "z1"): Fact.TRUE, ("y3", "z2"): Fact.TRUE})

    # print("Predicates before outer Join")

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
    p2_and_p2a = And(p2(x, y), p2a(y, z))
    p2_and_p2a.upward()

    assert all([p2_and_p2a.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"
    assert len(p2_and_p2a.state()) == len(GT_o), "FAILED 😔"
    model.flush()

    # TEST 2
    model = Model()

    t2_p3 = Predicate("t2_p3", arity=3, model=model)
    model.add_data(
        {t2_p3: {("x1", "y1", "z1"): Fact.TRUE, ("x3", "y3", "z3"): Fact.TRUE}}
    )

    t2_p2 = Predicate("t2_p2", arity=2, model=model)
    model.add_data({t2_p2: {("y1", "z1"): Fact.TRUE, ("y2", "z2"): Fact.TRUE}})

    GT_o = dict(
        [
            (("x1", "y1", "z1"), Fact.TRUE),
            (("x1", "y2", "z2"), Fact.UNKNOWN),
            (("x3", "y1", "z1"), Fact.UNKNOWN),
            (("x3", "y3", "z3"), Fact.UNKNOWN),
            (("x3", "y2", "z2"), Fact.UNKNOWN),
        ]
    )
    t2_p3_and_t2_p2 = And(t2_p3(x, y, z), t2_p2(y, z))
    t2_p3_and_t2_p2.upward()

    assert all(
        [t2_p3_and_t2_p2.state(groundings=g) is GT_o[g] for g in GT_o]
    ), "FAILED 😔"
    assert len(t2_p3_and_t2_p2.state()) == len(GT_o), "FAILED 😔"
    model.flush()

    # TEST 3
    model = Model()
    t2_p3 = Predicate("t2_p3", arity=3, model=model)
    model.add_data(
        {t2_p3: {("x1", "y1", "z1"): Fact.TRUE, ("x3", "y3", "z3"): Fact.TRUE}}
    )

    t2_p2 = Predicate("t2_p2", arity=2, model=model)
    model.add_data({t2_p2: {("y1", "z1"): Fact.TRUE, ("y2", "z2"): Fact.TRUE}})

    t3_p1 = Predicate("t3_p1", model=model)
    model.add_data({t3_p1: {"z1": Fact.TRUE, "z4": Fact.TRUE}})
    t2_p3_and_t2_p2_t3_p1 = And(t2_p3(x, y, z), t2_p2(y, z), t3_p1(z))
    t2_p3_and_t2_p2_t3_p1.upward()

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

    assert all(
        [t2_p3_and_t2_p2_t3_p1.state(groundings=g) is GT_o[g] for g in GT_o]
    ), "FAILED 😔"
    assert len(t2_p3_and_t2_p2_t3_p1.state()) == len(GT_o), "FAILED 😔"
    model.flush()

    # TEST 4
    model = Model()
    t2_p3 = Predicate("t2_p3", arity=3, model=model)
    model.add_data(
        {t2_p3: {("x1", "y1", "z1"): Fact.TRUE, ("x3", "y3", "z3"): Fact.TRUE}}
    )

    t2_p2 = Predicate("t2_p2", arity=2, model=model)
    model.add_data({t2_p2: {("y1", "z1"): Fact.TRUE, ("y2", "z2"): Fact.TRUE}})

    t4_p1 = Predicate("t4_p1", model=model)
    model.add_data({t4_p1: {"x1": Fact.TRUE, "x4": Fact.TRUE}})
    t2_p3_and_t2_p2_t4_p1 = And(t2_p3(x, y, z), t2_p2(y, z), t4_p1(x))
    t2_p3_and_t2_p2_t4_p1.upward()

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

    assert all(
        [t2_p3_and_t2_p2_t4_p1.state(groundings=g) is GT_o[g] for g in GT_o]
    ), "FAILED 😔"
    assert len(t2_p3_and_t2_p2_t4_p1.state()) == len(GT_o), "FAILED 😔"
    model.flush()

    # TEST 5
    model = Model()
    t2_p3 = Predicate("t2_p3", arity=3, model=model)
    model.add_data(
        {t2_p3: {("x1", "y1", "z1"): Fact.TRUE, ("x3", "y3", "z3"): Fact.TRUE}}
    )

    t2_p2, t5_p2 = Predicates("t2_p2", "t5_p2", arity=2, model=model)

    model.add_data(
        {
            t2_p2: {("y1", "z1"): Fact.TRUE, ("y2", "z2"): Fact.TRUE},
            t5_p2: {("a1", "b1"): Fact.TRUE, ("a2", "b2"): Fact.TRUE},
        }
    )
    t2_p3_and_t2_p2_t5_p2 = And(t2_p3(x, y, z), t2_p2(y, z), t5_p2(a, b))

    t2_p3_and_t2_p2_t5_p2.upward()

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
