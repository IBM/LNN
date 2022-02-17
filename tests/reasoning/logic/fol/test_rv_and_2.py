##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import numpy as np
from lnn import Predicate, And, Or, Implies, Model, Variable


def test_and():
    """FOL upward 2-input conjunction real value truth table"""

    samples = 101
    steps = np.linspace(0, 1, samples)
    x_grid, y_grid = np.meshgrid(steps, steps)

    x = Variable('x')
    model = Model()
    model['A'] = Predicate()
    model['B'] = Predicate()
    model['AB'] = And(model['A'](x), model['B'](x))

    for row in range(samples):
        for col in range(samples):

            # inputs
            a, b = x_grid[row][col], y_grid[row][col]

            # facts per model
            model.add_facts({
                'A': {
                    f'({row}, {col})': (a, a),
                },
                'B': {
                    f'({row}, {col})': (b, b),
                }
            })

            # ground truth
            GT = float(max(0, a + b - 1))
            model.add_labels({
                'AB': {f'({row}, {col})': (GT, GT)}})

    # evaluate the conjunction
    model['AB'].upward()

    # test the prediction
    for g in model['AB'].groundings:
        prediction = model['AB'].get_facts(g)[0]
        label = model['AB'].get_labels(g)[0]
        lower_bound = prediction[0].item()
        upper_bound = prediction[1].item()
        assert lower_bound == upper_bound, (
            f'Expected upper and lower bound to be the same but \
            got {lower_bound}, {upper_bound}'
        )
        assert round(lower_bound, 4) == round(label.item(), 4), (
            f'And({a}, {b}) expected {label}, but got {lower_bound}')


def test_or():
    """FOL upward 2-input disjunction real value truth table"""

    samples = 101
    steps = np.linspace(0, 1, samples)
    x_grid, y_grid = np.meshgrid(steps, steps)

    x = Variable('x')
    model = Model()
    model['A'] = Predicate()
    model['B'] = Predicate()
    model['AB'] = Or(model['A'](x), model['B'](x))

    for row in range(samples):
        for col in range(samples):
            # inputs
            a, b = x_grid[row][col], y_grid[row][col]

            # facts per model
            model.add_facts({
                'A': {
                    f'({row}, {col})': (a, a),
                },
                'B': {
                    f'({row}, {col})': (b, b),
                }
            })

            # ground truth
            GT = float(min(1, a + b))
            model.add_labels({
                'AB': {f'({row}, {col})': (GT, GT)}})

    # evaluate the conjunction
    model['AB'].upward()

    # test the prediction
    for g in model['AB'].groundings:
        prediction = model['AB'].get_facts(g)[0]
        label = model['AB'].get_labels(g)[0]
        lower_bound = prediction[0].item()
        upper_bound = prediction[1].item()
        assert lower_bound == upper_bound, (
            f'Expected upper and lower bound to be the same but \
                got {lower_bound}, {upper_bound}'
        )
        assert round(lower_bound, 4) == round(label.item(), 4), (
            f'And({a}, {b}) expected {label}, but got {lower_bound}')


def test_implies():
    """FOL upward 2-input implies real value truth table"""

    samples = 101
    steps = np.linspace(0, 1, samples)
    x_grid, y_grid = np.meshgrid(steps, steps)

    x = Variable('x')
    model = Model()
    model['A'] = Predicate()
    model['B'] = Predicate()
    model['AB'] = Implies(model['A'](x), model['B'](x))

    for row in range(samples):
        for col in range(samples):
            # inputs
            a, b = x_grid[row][col], y_grid[row][col]

            # facts per model
            model.add_facts({
                'A': {
                    f'({row}, {col})': (a, a),
                },
                'B': {
                    f'({row}, {col})': (b, b),
                }
            })

            # ground truth
            GT = float(min(1, 1 - a + b))
            model.add_labels({
                'AB': {f'({row}, {col})': (GT, GT)}})

    # evaluate the conjunction
    model['AB'].upward()

    # test the prediction
    for g in model['AB'].groundings:
        prediction = model['AB'].get_facts(g)[0]
        label = model['AB'].get_labels(g)[0]
        lower_bound = prediction[0].item()
        upper_bound = prediction[1].item()
        assert lower_bound == upper_bound, (
            f'Expected upper and lower bound to be the same but \
                got {lower_bound}, {upper_bound}'
        )
        assert round(lower_bound, 4) == round(label.item(), 4), (
            f'And({a}, {b}) expected {label}, but got {lower_bound}')


if __name__ == "__main__":
    test_and()
    test_or()
    test_implies()
