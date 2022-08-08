##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Fact, Variables

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test():
    model = Model()
    x, y, z = map(Variable, ("x", "y", "z"))

    # This is the normal 2 var vs 2 var ; should go thru the memory join
    p2 = model.add_predicates(2, "p2")
    model.add_data(
        {
            p2: {
                ("s1", "s7"): TRUE,
                ("s1", "s6"): TRUE,
                ("s2", "s6"): FALSE,
                ("s3", "s7"): FALSE,
                ("s4", "s6"): TRUE,
            }
        }
    )

    p2a = model.add_predicates(2, "p2a")
    model.add_data(
        {
            p2a: {
                ("s1", "s7"): TRUE,
                ("s1", "s6"): FALSE,
                ("s2", "s5"): FALSE,
                ("s4", "s7"): FALSE,
                ("s7", "s6"): TRUE,
            }
        }
    )

    GT = dict(
        [
            (("s1", "s7"), TRUE),
            (("s1", "s6"), FALSE),
            (("s2", "s6"), FALSE),
            (("s3", "s7"), FALSE),
            (("s4", "s6"), UNKNOWN),
            (("s2", "s5"), FALSE),
            (("s4", "s7"), FALSE),
            (("s7", "s6"), UNKNOWN),
        ]
    )
    p2_and_p2a = And(p2(x, y), p2a(x, y))
    model.add_knowledge(p2_and_p2a)
    p2_and_p2a.upward()
    assert all([p2_and_p2a.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
    assert len(p2_and_p2a.state()) == len(GT), "FAILED 😔"

    # 1 variable vs 2 variables

    model = Model()  # Reset the model for each new test.

    p2 = model.add_predicates(2, "p2")
    model.add_data(
        {
            p2: {
                ("s1", "s7"): TRUE,
                ("s1", "s6"): TRUE,
                ("s2", "s6"): FALSE,
                ("s3", "s7"): FALSE,
                ("s4", "s6"): TRUE,
            }
        }
    )

    p1 = model.add_predicates(1, "p1")
    model.add_data(
        {p1: {"s1": TRUE, "s2": TRUE, "s3": TRUE, "s4": FALSE, "s10": FALSE}}
    )
    p1_and_p2 = And(p1(x), p2(x, y))
    model.add_knowledge(p1_and_p2)
    p1_and_p2.upward()

    GT = dict(
        [
            (("s1", "s6"), TRUE),
            (("s3", "s7"), FALSE),
            (("s2", "s6"), FALSE),
            (("s1", "s7"), TRUE),
            (("s4", "s6"), FALSE),
        ]
    )

    assert all([p1_and_p2.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
    assert len(p1_and_p2.state()) == len(GT), "FAILED 😔"

    # 2 variable vs 3 variables
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
    p2_and_p3 = And(p2(x, y), p3(x, z, y))
    model.add_knowledge(p2_and_p3)
    p2_and_p3.upward()

    GT = dict([(("s2", "s6", "s8"), FALSE), (("s1", "s7", "s4"), TRUE)])
    assert all(p2_and_p3.state(groundings=g) is GT[g] for g in GT), "FAILED 😔"
    assert len(p2_and_p3.state()) == len(GT), "FAILED 😔"

    # 1 vs 2 vs 3
    model = Model()  # Reset the model for each new test.

    p1 = model.add_predicates(1, "p1")
    p2 = model.add_predicates(2, "p2")
    p3 = model.add_predicates(3, "p3")
    model.add_data(
        {
            p1: {"s1": TRUE, "s2": TRUE, "s3": TRUE, "s4": FALSE, "s10": FALSE},
            p2: {
                ("s1", "s7"): TRUE,
                ("s1", "s6"): TRUE,
                ("s1", "s3"): TRUE,
                ("s2", "s6"): FALSE,
                ("s3", "s7"): FALSE,
                ("s4", "s8"): TRUE,
            },
            p3: {
                ("s1", "s5", "s3"): TRUE,
                ("s1", "s8", "s3"): FALSE,
                ("s1", "s4", "s7"): TRUE,
                ("s2", "s8", "s6"): TRUE,
                ("s4", "s6", "s8"): FALSE,
            },
        }
    )

    x, y, z = Variables("x", "y", "z")
    p1_and_p2_and_p3 = And(p1(x), p2(x, y), p3(x, z, y))

    model.add_knowledge(p1_and_p2_and_p3)
    p1_and_p2_and_p3.upward()

    GT = dict(
        [
            (("s1", "s3", "s5"), TRUE),
            (("s1", "s3", "s8"), FALSE),
            (("s1", "s7", "s4"), TRUE),
            (("s2", "s6", "s8"), FALSE),
            (("s4", "s8", "s6"), FALSE),
        ]
    )

    p1_and_p2_and_p3.print()
    assert all([p1_and_p2_and_p3.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
    assert len(p1_and_p2_and_p3.state()) == len(GT), "FAILED 😔"

    # 2 variable vs 2 variable reversed
    p2r = model.add_predicates(2, "p2r")
    model.add_data({p2r: {("s6", "s2"): TRUE, ("s7", "s1"): FALSE}})
    p2_and_p2r = And(p2(x, y), p2r(y, x))
    model.add_knowledge(p2_and_p2r)
    p2_and_p2r.upward()
    GT = dict([(("s2", "s6"), FALSE), (("s1", "s7"), FALSE)])
    assert all([p2_and_p2r.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
    assert len(p2_and_p2r.state()) == len(GT), "FAILED 😔"


if __name__ == "__main__":
    test()
    print("success")
