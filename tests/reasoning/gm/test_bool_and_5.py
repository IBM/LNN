##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, And, Model, Variable, Fact, World
import random

TRUE = Fact.TRUE


def test_1():
    """
    And(P1(x0), P2(x0, x1, ..., xn))
    """
    n_vars = 1000
    gt = 3

    model = Model()
    var_labels = tuple(f"x{i}" for i in range(0, n_vars))
    variables = list(map(Variable, var_labels))

    p0 = Predicate("p0")
    p1 = Predicate("p1", arity=n_vars)
    _and = And(p0(variables[0]), p1(*variables), world=World.AXIOM)
    model.add_knowledge(_and)
    model.add_data({p0: {"0": TRUE, "1": TRUE, "2": TRUE}, p1: {("0",) * n_vars: TRUE}})

    model.infer()
    assert len(_and.groundings) == gt, (
        f"Expected {gt} grounding, " f"received {len(_and.groundings)}"
    )


def test_2():
    """
    And(P1(xr), P2(x0, x1, ..., xn))
    """
    n_vars = 1000
    gt = 3

    model = Model()
    var_labels = tuple(f"x{i}" for i in range(0, n_vars))
    variables = list(map(Variable, var_labels))

    p0 = Predicate("p0")
    p1 = Predicate("p1", arity=n_vars)

    r = random.randrange(0, n_vars)
    _and = And(p0(variables[r]), p1(*variables), world=World.AXIOM)
    model.add_knowledge(_and)
    model.add_data({p0: {"0": TRUE, "1": TRUE, "2": TRUE}, p1: {("0",) * n_vars: TRUE}})

    model.infer()
    assert len(_and.groundings) == gt, (
        f"Expected {gt} grounding, " f"received {len(_and.groundings)}"
    )


def test_3():
    """
    And(P1(x0, x1, x2), P2(x0, x1, ..., xn))
    """
    n_vars = 5
    gt = 3

    model = Model()
    var_labels = tuple(f"x{i}" for i in range(0, n_vars))
    variables = list(map(Variable, var_labels))

    p0 = Predicate("p0", arity=3)
    p1 = Predicate("p1", arity=n_vars)
    _and = And(
        p0(variables[0], variables[1], variables[2]),
        p1(*variables),
        world=World.AXIOM,
    )

    model.add_knowledge(_and)

    model.add_data(
        {
            p0: {
                ("0", "0", "0"): TRUE,
                ("0", "0", "1"): TRUE,
                ("0", "0", "2"): TRUE,
            }
        }
    )

    model.add_data(
        {
            p1: {
                ("0", "0", "0", "0", "0"): TRUE,
                ("0", "0", "1", "0", "0"): TRUE,
                ("0", "0", "2", "0", "0"): TRUE,
            }
        }
    )

    model.infer()
    assert len(_and.groundings) == gt, (
        f"Expected {gt} groundings, " f"received {len(_and.groundings)}"
    )


def test_4():
    """
    And(P1(nc3), P2(x0, x1, ..., xn))
    """
    n_vars = 1000
    gt = 3

    model = Model()
    var_labels = tuple(f"x{i}" for i in range(0, n_vars))
    variables = list(map(Variable, var_labels))

    p0 = Predicate("p0", arity=n_vars)
    p1 = Predicate("p1", arity=n_vars)
    _and = And(p0(*variables), p1(*variables), world=World.AXIOM)
    model.add_knowledge(_and)

    key_arr = ["0"] * n_vars
    random_indices = random.sample(range(1, n_vars), 3)

    for i in random_indices:
        key_arr[i] = "1"

    key = tuple(key_arr)

    model.add_data({p0: {key: TRUE, ("0",) * n_vars: TRUE}})
    model.add_data({p1: {key: TRUE, ("1",) * n_vars: TRUE}})

    model.infer()
    assert len(_and.groundings) == gt, (
        f"Expected {gt} groundings, " f"received {len(_and.groundings)}"
    )


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
