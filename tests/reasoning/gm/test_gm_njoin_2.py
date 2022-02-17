##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Predicate, TRUE, FALSE, UNKNOWN


def test():
    model = Model()
    x, y, z = map(Variable, ("x", "y", "z"))

    model["p2"] = Predicate("p2", arity=2)
    model.add_facts(
        {
            "p2": {
                ("s1", "s7"): TRUE,
                ("s1", "s6"): TRUE,
                ("s2", "s6"): FALSE,
                ("s3", "s7"): FALSE,
                ("s4", "s6"): TRUE,
            }
        }
    )

    model["p1_null"] = Predicate("p1_null")
    model["p1_and_p2"] = And(model["p1_null"](x), model["p2"](x, y))
    model["p1_and_p2"].upward()

    GT = dict(
        [
            (("s1", "s6"), UNKNOWN),
            (("s3", "s7"), FALSE),
            (("s2", "s6"), FALSE),
            (("s1", "s7"), UNKNOWN),
            (("s4", "s6"), UNKNOWN),
        ]
    )

    assert all(
        [model["p1_and_p2"].state(groundings=g) is GT[g] for g in GT]
    ), "FAILED 😔"
    assert len(model["p1_and_p2"].state()) == len(GT), "FAILED 😔"

    # 2 variable vs 3 variables
    model["p2_null"] = Predicate("p2_null", arity=2)
    model["p3"] = Predicate("p3", arity=3)
    model.add_facts(
        {
            "p3": {
                ("s1", "s5", "s3"): TRUE,
                ("s1", "s4", "s7"): TRUE,
                ("s1", "s8", "s3"): FALSE,
                ("s2", "s8", "s6"): TRUE,
                ("s4", "s6", "s8"): FALSE,
            }
        }
    )

    model["p2_and_p3"] = And(model["p2_null"](x, y), model["p3"](x, z, y))
    model["p2_and_p3"].upward()
    GT = dict(
        [
            (("s4", "s8", "s6"), FALSE),
            (("s1", "s3", "s8"), FALSE),
            (("s1", "s3", "s5"), UNKNOWN),
            (("s2", "s6", "s8"), UNKNOWN),
            (("s1", "s7", "s4"), UNKNOWN),
        ]
    )

    assert all(model["p2_and_p3"].state(groundings=g) is GT[g] for g in GT), "FAILED 😔"
    assert len(model["p2_and_p3"].state()) == len(GT), "FAILED 😔"

    # 1 vs 2 vs 3
    model = Model()  # Reset the model for each new test.
    model["p1_null"] = Predicate("p1_null")

    model["p2_null"] = Predicate("p2_null", arity=2)
    model["p3"] = Predicate("p3", arity=3)
    model.add_facts(
        {
            "p3": {
                ("s1", "s5", "s3"): TRUE,
                ("s1", "s4", "s7"): TRUE,
                ("s1", "s8", "s3"): FALSE,
                ("s2", "s8", "s6"): TRUE,
                ("s4", "s6", "s8"): FALSE,
            }
        }
    )

    model["p1_and_p2_and_p3"] = And(
        model["p1_null"](x), model["p2_null"](x, y), model["p3"](x, z, y)
    )
    model["p1_and_p2_and_p3"].upward()

    assert all(
        model["p1_and_p2_and_p3"].state(groundings=g) is GT[g] for g in GT
    ), "FAILED 😔"
    assert len(model["p1_and_p2_and_p3"].state()) == len(GT), "FAILED 😔"


if __name__ == "__main__":
    test()
    print("success")
