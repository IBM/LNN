##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, TRUE, FALSE, Predicate, And, \
    Variable, plot_loss, plot_params


def test_1():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in both directions
    """
    model = Model()
    A = model['A'] = Predicate()
    B = model['B'] = Predicate()
    x = Variable('x')
    model['AB'] = And(A(x), B(x),
                      neuron={
                          'bias_learning': False,
                          'alpha': .95,
                          # 'alpha_sigma': 2,
                          # 'alpha_min': True,
                          # 'w_max': 1,
                          })

    model.add_facts({
        'A': {
            '0': TRUE,
            '1': TRUE,
            '2': FALSE,
            '3': FALSE,
        },
        'B': {
            '0': TRUE,
            '1': FALSE,
            '2': TRUE,
            '3': FALSE,
        },
    })
    model.add_labels({
        'AB': {
            '0': TRUE,
            '1': TRUE,
            '2': TRUE,
            '3': FALSE,
        },
    })
    parameter_history = {'bias': True, 'weights': True}
    losses = {'supervised': .1,
              'logical': {
                  'coeff': 0,
                  'slacks': 0
              },
              'contradiction': 1
              }
    total_loss, _ = model.train(
        epochs=500,
        learning_rate=.01,
        losses=losses,
        parameter_history=parameter_history,
        # stop_at_convergence=False,
    )

    model.print(params=True, roundoff=10)
    plot_loss(total_loss, losses)
    plot_params(model)
    model["AB"].print(params=True)
    print('infeasibility', model['AB'].neuron.feasibility)
    print('slacks', model['AB'].neuron.slacks if (
        hasattr(model['AB'].neuron, 'slacks')) else (
        losses['logical']['slacks']))
    print('s*w',
          model['AB'].neuron.slacks[1]*model['AB'].neuron.weights if (
              hasattr(model['AB'].neuron, 'slacks')) else (
              losses['logical']['slacks']))

    # scenario 1, beta-learnable:
    # loss:             0.
    # weights:          [1.50092375 0.49848363]
    # infeasibility:    (tensor(0.), tensor([0., 0.]))
    # slacks:           (tensor(0.), tensor([0.0250, 0.9763]))
    # s*w:              tensor([0.0375, 0.4867])

    # scenario 2, beta=1:
    # loss:             0.
    # weights:          [1. 0.]
    # infeasibility:    (tensor(0.), tensor([0., 0.]))
    # slacks:           (tensor(0.), tensor([0.0000, 0.9500]))
    # s * w:            tensor([0., 0.])


if __name__ == "__main__":
    test_1()
    print('success')
