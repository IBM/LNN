##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Predicate, And, Model, Variable, World, TRUE, Join
import random


def test_1():
    """
    And(P1(x0), P2(x0, x1, ..., xn))
    """
    for join, gt in zip([Join.INNER, Join.OUTER], [1, 3]):
        n_vars = 1000

        model = Model()
        var_labels = tuple(f'x{i}' for i in range(0, n_vars))
        variables = list(map(Variable, var_labels))

        p0 = model['p0'] = Predicate()
        p1 = model['p1'] = Predicate(arity=n_vars)

        model['And'] = And(p0(variables[0]), p1(*variables),
                           world=World.AXIOM, join=join)

        model.add_facts({'p0': {'0': TRUE, '1': TRUE, '2': TRUE}})
        model.add_facts({'p1': {('0',) * n_vars: TRUE}})

        model.infer()
        assert len(model['And'].groundings) == gt, (
            f"Expected {gt} grounding, "
            f"received {len(model['And'].groundings)}")


def test_2():
    """
    And(P1(xr), P2(x0, x1, ..., xn))
    """
    for join, gt in zip([Join.INNER, Join.OUTER], [1, 3]):
        n_vars = 1000

        model = Model()
        var_labels = tuple(f'x{i}' for i in range(0, n_vars))
        variables = list(map(Variable, var_labels))

        p0 = model['p0'] = Predicate()
        p1 = model['p1'] = Predicate(arity=n_vars)

        r = random.randrange(0, n_vars)
        model['And'] = And(p0(variables[r]), p1(*variables),
                           world=World.AXIOM, join=join)

        model.add_facts({'p0': {'0': TRUE, '1': TRUE, '2': TRUE}})
        model.add_facts({'p1': {('0',) * n_vars: TRUE}})

        model.infer()
        assert len(model['And'].groundings) == gt, (
            f"Expected {gt} grounding, "
            f"received {len(model['And'].groundings)}")


def test_3():
    """
    And(P1(x0, x1, x2), P2(x0, x1, ..., xn))
    """
    n_vars = 5

    for join, gt in zip([Join.INNER, Join.OUTER], [3, 3]):
        model = Model()
        var_labels = tuple(f'x{i}' for i in range(0, n_vars))
        variables = list(map(Variable, var_labels))

        p0 = model['p0'] = Predicate(arity=3)
        p1 = model['p1'] = Predicate(arity=n_vars)

        model['And'] = And(p0(variables[0], variables[1], variables[2]),
                           p1(*variables), world=World.AXIOM)

        model.add_facts({'p0': {('0', '0', '0'): TRUE,
                                ('0', '0', '1'): TRUE,
                                ('0', '0', '2'): TRUE, }})

        model.add_facts({'p1': {('0', '0', '0', '0', '0'): TRUE,
                                ('0', '0', '1', '0', '0'): TRUE,
                                ('0', '0', '2', '0', '0'): TRUE, }})

        model.infer()
        assert len(model['And'].groundings) == gt, (
            f"Expected {gt} groundings, "
            f"received {len(model['And'].groundings)}")


def test_4():
    """
    And(P1(nc3), P2(x0, x1, ..., xn))
    """
    n_vars = 1000

    for join, gt in zip([Join.INNER, Join.OUTER], [3, 3]):

        model = Model()
        var_labels = tuple(f'x{i}' for i in range(0, n_vars))
        variables = list(map(Variable, var_labels))

        p0 = model['p0'] = Predicate(arity=n_vars)
        p1 = model['p1'] = Predicate(arity=n_vars)

        model['And'] = And(p0(*variables), p1(*variables),
                           world=World.AXIOM, join=join)

        key_arr = ['0'] * n_vars
        random_indices = random.sample(range(1, n_vars), 3)

        for i in random_indices:
            key_arr[i] = '1'

        key = tuple(key_arr)

        model.add_facts({'p0': {key: TRUE, ('0',) * n_vars: TRUE}})
        model.add_facts({'p1': {key: TRUE, ('1',) * n_vars: TRUE}})

        model.infer()
        assert len(model['And'].groundings) == gt, (
            f"Expected {gt} groundings, "
            f"received {len(model['And'].groundings)}")


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
