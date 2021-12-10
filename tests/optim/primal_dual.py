##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, TRUE, FALSE, And, Or, plot_loss, plot_params
from lnn.optim import PrimalDual


def test_primal_dual_1():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in both directions
    """
    model = Model()
    A, B = model.add_propositions('A', 'B')
    model['AB'] = And(A, B)
    model['A|B'] = Or(A, B)

    model.add_facts({
        'A': TRUE,
        'B': FALSE
    })
    model.add_labels({
        'AB': TRUE,
        'A|B': FALSE
    })

    parameter_history = {'bias': True, 'weights': True}
    losses = {'supervised': 1}
    total_loss, _ = model.train(
        optimizer=PrimalDual(model.parameters_grouped_by_neuron(), lr=.1),
        losses=losses,
        parameter_history=parameter_history
    )

    model.print(params=True, roundoff=10)
    plot_loss(total_loss, losses)
    plot_params(model)


if __name__ == "__main__":
    test_primal_dual_1()
    print('success')
