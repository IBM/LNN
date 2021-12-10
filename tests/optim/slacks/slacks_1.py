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
    C = model['C'] = Predicate()
    x = Variable('x')
    model['AB'] = And(A(x), B(x), C(x),
                      neuron={
                          # 'bias_learning': False,
                          'alpha': .95,
                          # 'alpha_sigma': 2,
                          # 'alpha_min': True,
                          # 'w_max': 1,
                          })

    model.add_facts({
        'A': {
            '0': TRUE,
            '1': TRUE,
            '2': TRUE,
            '3': TRUE,
            '4': FALSE,
            '5': FALSE,
            '6': FALSE,
            '7': FALSE,
        },
        'B': {
            '0': TRUE,
            '1': TRUE,
            '2': FALSE,
            '3': FALSE,
            '4': TRUE,
            '5': TRUE,
            '6': FALSE,
            '7': FALSE,
        },
        'C': {
            '0': TRUE,
            '1': FALSE,
            '2': TRUE,
            '3': FALSE,
            '4': TRUE,
            '5': FALSE,
            '6': TRUE,
            '7': FALSE,
        },
    })
    model.add_labels({
        'AB': {
            '0': TRUE,
            '1': FALSE,
            '2': FALSE,
            '3': FALSE,
            '4': TRUE,
            '5': FALSE,
            '6': TRUE,
            '7': FALSE,
        },
    })
    parameter_history = {'bias': True, 'weights': True}
    losses = {'supervised': 1,
              'logical': {
                  'coeff': .1,
                  'slacks': False
              },
              'contradiction': 1
              }
    total_loss, _ = model.train(
        epochs=300,
        learning_rate=.05,
        losses=losses,
        parameter_history=parameter_history,
        # stop_at_convergence=False,
    )

    model.print(params=True, roundoff=10)
    plot_loss(total_loss, losses)
    plot_params(model)
    print('infeasibility', model['AB'].neuron.feasibility)
    print('slacks', model['AB'].neuron.slacks if (
        hasattr(model['AB'].neuron, 'slacks')) else (
        losses['logical']['slacks']))


if __name__ == "__main__":
    test_1()
    print('success')

    # accuracy
    # 0, None = loss tensor(0.0625
    # β: 1.5551519394,  w: [1.05515206 0.         0.53930491]
    # infeasibility (tensor(0.), tensor([0., 0., 0.]))
    # slacks (tensor(0.), tensor([0.5025, 1.5097, 0.9949]))

    # accuracy
    # 1, TRUE = loss tensor(0.0625
    # β: 1.5551520586,  w: [1.05515218 0.         0.53930503]
    # infeasibility (tensor(0.), tensor([0., 0., 0.]))
    # slacks (tensor(0.), tensor([0.5025, 1.5097, 0.9949]))

    # ~accuracy/~interpretability
    # 1, 1 =  loss tensor(0.2234
    # β: 1.3927744627,  w: [0.49731901 0.2989068  0.38981315]
    # infeasibility (tensor(0.), tensor([-0.1328,  0.0654, -0.0300]))
    # slacks 1

    # interpretability
    # 1, 0 =  loss tensor(0.6082
    # β: 0.9413008094,  w: [0.80416089 0.56286442 0.68351269]
    # infeasibility (tensor(0.1065), tensor([0.1282, 0.3585, 0.2434]))
    # slacks 0
