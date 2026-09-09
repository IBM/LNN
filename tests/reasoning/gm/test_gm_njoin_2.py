##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Predicate, Fact

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test():
    model = Model()
    x, y, z = map(Variable, ("x", "y", "z"))

    p1_null = Predicate("p1_null")
    p2 = Predicate("p2", arity=2)
    p1_and_p2 = And(p1_null(x), p2(x, y))
    model.add_knowledge(p1_and_p2)
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

    p1_and_p2.upward()

    GT = dict(
        [
            (("s1", "s6"), UNKNOWN),
            (("s3", "s7"), FALSE),
            (("s2", "s6"), FALSE),
            (("s1", "s7"), UNKNOWN),
            (("s4", "s6"), UNKNOWN),
        ]
    )

    assert all([p1_and_p2.state(groundings=g) is GT[g] for g in GT]), "FAILED 😔"
    assert len(p1_and_p2.state()) == len(GT), "FAILED 😔"

    # 2 variable vs 3 variables
    p2_null = Predicate("p2_null", arity=2)
    p3 = Predicate("p3", arity=3)
    p2_and_p3 = And(p2_null(x, y), p3(x, z, y))
    model.add_knowledge(p2_and_p3)
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

    p2_and_p3.upward()
    GT = dict(
        [
            (("s4", "s8", "s6"), FALSE),
            (("s1", "s3", "s8"), FALSE),
            (("s1", "s3", "s5"), UNKNOWN),
            (("s2", "s6", "s8"), UNKNOWN),
            (("s1", "s7", "s4"), UNKNOWN),
        ]
    )

    assert all(p2_and_p3.state(groundings=g) is GT[g] for g in GT), "FAILED 😔"
    assert len(p2_and_p3.state()) == len(GT), "FAILED 😔"

    # 1 vs 2 vs 3
    model = Model()  # Reset the model for each new test.
    p1_null = model.add_predicates(1, "p1_null")
    p2_null = model.add_predicates(2, "p2_null")
    p3 = model.add_predicates(3, "p3")
    p1_and_p2_and_p3 = And(p1_null(x), p2_null(x, y), p3(x, z, y))
    model.add_knowledge(p1_and_p2_and_p3)
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

    p1_and_p2_and_p3.upward()

    assert all(p1_and_p2_and_p3.state(groundings=g) is GT[g] for g in GT), "FAILED 😔"
    assert len(p1_and_p2_and_p3.state()) == len(GT), "FAILED 😔"


if __name__ == "__main__":
    test()
