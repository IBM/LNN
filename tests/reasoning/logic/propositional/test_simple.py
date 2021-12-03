##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (Model, And, Or, Proposition,
                 AXIOM, UPWARD, FALSE, TRUE,
                 plot_params, plot_loss)


def test_1():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    given Or(A, B) - do not do anything
    training in the upward direction
    """
    # model
    model = Model()

    # rules
    model['A'] = Proposition('A')
    model['B'] = Proposition('B')
    model['AB'] = And(model['A'], model['B'], world=AXIOM)
    model['A|B'] = Or(model['A'], model['B'])

    # facts
    model.add_facts({
        'A': TRUE,
        'B': FALSE
    })

    # train/inference
    losses = {
        'contradiction': 1,
        'logical': 1e-1}
    total_loss, _ = model.train(
        pbar=False,
        direction=UPWARD,
        losses=losses,
        parameter_history={'weights': True, 'bias': True}
    )
    model.print(params=True)
    plot_params(model)
    plot_loss(total_loss, losses)

    A_and_B_weights = model['AB'].params('weights')
    assert A_and_B_weights[1] <= .5, (
        'expected input B to be downweighted <= .5, '
        f'received {A_and_B_weights[1]}')
    A_or_B_weights = model['A|B'].params('weights')
    assert all(A_or_B_weights > .99), (
        'expected weights at A or B to remain high, '
        f'received {A_or_B_weights}')
