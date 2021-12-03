##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Or, Model, TRUE, FALSE, plot_params, plot_loss


def test_upward():
    """standard upward, 2-input conjunction boolean truth table"""

    # instantiate a context
    model = Model()

    # define the rules
    A, B = model.add_propositions('A', 'B')
    AB = model['AB'] = Or(A, B)

    # set the facts
    model.add_facts({
        'A': TRUE,
        'B': FALSE,
    })

    # set labels
    model.add_labels({
        'AB': FALSE
    })

    # learning/reasoning
    losses = {'supervised': None, 'logical': 1e-1, 'contradiction': None}
    parameter_history = {'weights': True, 'bias': True}
    total_loss, _ = model.train(
        losses=losses,
        parameter_history=parameter_history)

    # evaluation
    prediction = model['AB'].state()
    model.print(params=True)
    plot_params(model)
    plot_loss(total_loss, losses)
    print(total_loss[0][-1])


if __name__ == "__main__":
    test_upward()
    print('success')
