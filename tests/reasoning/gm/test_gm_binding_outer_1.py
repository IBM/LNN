##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Variable, Predicate, Fact


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
            ("y1", Fact.UNKNOWN),
            ("y3", Fact.UNKNOWN),
        ]
    )
    p2_and_p2a = And(p2("x1", y), p2a(y, "z2"))
    p2_and_p2a.upward()
    model.print()

    assert all([p2_and_p2a.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"
    assert len(p2_and_p2a.state()) == len(GT_o), "FAILED 😔"
    model.flush()


if __name__ == "__main__":
    test()
