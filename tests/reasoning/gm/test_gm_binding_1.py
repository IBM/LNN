##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Fact


TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test():
    model = Model()
    x, y, z = map(Variable, ("x", "y", "z"))

    p2 = model.add_predicates(2, "p2")
    p2_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), TRUE),
            (("s2", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s4", "s7"), TRUE),
        ]
    )
    model.add_data({p2: p2_facts})

    p2a = model.add_predicates(2, "p2a")
    p2a_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), FALSE),
            (("s2", "s5"), FALSE),
            (("s4", "s7"), FALSE),
            (("s3", "s7"), FALSE),
            (("s7", "s6"), TRUE),
        ]
    )
    model.add_data({p2a: p2a_facts})

    # the unbounded case
    p2_and_p2a = And(p2(x, y), p2a(x, y))
    model.add_knowledge(p2_and_p2a)
    GT = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s2", "s6"), FALSE),
            (("s2", "s5"), FALSE),
            (("s4", "s7"), FALSE),
            (("s7", "s6"), UNKNOWN),
        ]
    )

    p2_and_p2a.upward()
    assert all([p2_and_p2a.state(groundings=g)
               is GT[g] for g in GT]), "FAILED 😔"
    assert len(p2_and_p2a.state()) == len(GT), "FAILED 😔"

    model = Model()
    p2 = model.add_predicates(2, "p2")
    p2_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), TRUE),
            (("s2", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s4", "s7"), TRUE),
        ]
    )
    model.add_data({p2: p2_facts})

    p2a = model.add_predicates(2, "p2a")
    p2a_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), FALSE),
            (("s2", "s5"), FALSE),
            (("s4", "s7"), FALSE),
            (("s3", "s7"), FALSE),
            (("s7", "s6"), TRUE),
        ]
    )
    model.add_data({p2a: p2a_facts})

    # One variable bound
    p2_and_p2b = And(p2(x, y), p2a(x, y, bind={y: "s7"}))

    model.add_knowledge(p2_and_p2b)

    GT = dict([
        (('s1', 's7'), TRUE),
        (('s3', 's7'), FALSE),
        (('s4', 's7'), FALSE)])

    model[p2_and_p2b].upward()
    assert all([model[p2_and_p2b].state(groundings=g) is GT[g]
                for g in GT]), "FAILED 😔"
    assert len(model[p2_and_p2b].state()) == len(GT), "FAILED 😔"
    # New test case
    model = Model()
    p2 = model.add_predicates(2, "p2")
    p2_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), TRUE),
            (("s2", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s4", "s7"), TRUE),
        ]
    )
    model.add_data({p2: p2_facts})

    p2a = model.add_predicates(2, "p2a")
    p2a_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), FALSE),
            (("s2", "s6"), FALSE),
            (("s4", "s7"), FALSE),
            (("s3", "s7"), FALSE),
            (("s7", "s6"), TRUE),
        ]
    )
    model.add_data({p2a: p2a_facts})
    p2_and_p2c = And(p2(x, y), p2a(x, y, bind={x: ["s1", "s2"], y: "s6"}))
    model.add_knowledge(p2_and_p2c)

    GT = dict([
        (('s1', 's6'), FALSE),
        (('s2', 's6'), FALSE)])
    model[p2_and_p2c].upward()
    assert all([model[p2_and_p2c].state(groundings=g) is GT[g]
                for g in GT]), "FAILED 😔"
    assert len(model[p2_and_p2c].state()) == len(GT), "FAILED 😔"
    # 1 variable vs 2 variables bound
    model = Model()
    p1 = model.add_predicates(1, "p1")
    model.add_data(
        {p1: {"s1": TRUE, "s2": TRUE, "s3": TRUE, "s4": FALSE, "s10": FALSE}}
    )
    p2 = model.add_predicates(2, "p2")
    p2_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), TRUE),
            (("s2", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s4", "s7"), TRUE),
        ]
    )
    model.add_data({p2: p2_facts})
    p1_and_p2 = And(p1(x, bind={x: ["s1", "s2"]}), p2(x, y))

    model.add_knowledge(p1_and_p2)
    GT = dict([
        (('s1', 's6'), TRUE),
        (('s2', 's6'), FALSE),
        (('s1', 's7'), TRUE)])
    model[p1_and_p2].upward()
    assert all([model[p1_and_p2].state(groundings=g) is GT[g]
                for g in GT]), "FAILED 😔"
    assert len(model[p1_and_p2].state()) == len(GT), "FAILED 😔"
    # New test case
    model = Model()
    p2 = model.add_predicates(2, "p2")
    p2_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), TRUE),
            (("s2", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s4", "s7"), TRUE),
        ]
    )
    model.add_data({p2: p2_facts})
    # 2 variable vs 3 variables bound
    p3 = model.add_predicates(3, "p3")
    model.add_data(
        {
            p3: {
                ("s1", "s5", "s3"): TRUE,
                ("s1", "s4", "s7"): TRUE,
                ("s1", "s8", "s3"): FALSE,
                ("s2", "s8", "s6"): TRUE,
                ("s4", "s6", "s8"): FALSE,
            }
        }
    )
    p2_and_p3 = And(p2(x, y), p3(x, z, y, bind={z: "s4"}))
    model.add_knowledge(p2_and_p3)
    model[p2_and_p3].upward()
    GT = dict([
        (('s1', 's7', 's4'), TRUE)])
    assert all([model[p2_and_p3].state(groundings=g) is GT[g]
                for g in GT]), "FAILED 😔"
    assert len(model[p2_and_p3].state()) == len(GT), "FAILED 😔"
    # New test case
    model = Model()

    p2 = model.add_predicates(2, "p2")
    p2_facts = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), TRUE),
            (("s2", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s4", "s7"), TRUE),
        ]
    )
    model.add_data({p2: p2_facts})

    p2r = model.add_predicates(2, "p2r")
    model.add_data({p2r: {("s6", "s2"): TRUE, ("s7", "s1"): FALSE}})
    p2_and_p2r = And(p2(x, y), p2r(y, x, bind={y: ["s7", "s3"]}))

    model.add_knowledge(p2_and_p2r)

    p2_and_p2r.upward()
    GT = dict([
        (('s1', 's7'), FALSE)])
    assert all([model[p2_and_p2r].state(groundings=g) is GT[g]
                for g in GT]), "FAILED 😔"
    assert len(model[p2_and_p2r].state()) == len(GT), "FAILED 😔"
    return


if __name__ == "__main__":
    test()
    print("success")
