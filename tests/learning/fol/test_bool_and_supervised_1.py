##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import (Model, And, Variable, plot_loss, plot_params,
                 TRUE, FALSE, UNKNOWN, UPWARD)


def test():
    model = Model()
    p1, p2 = model.add_predicates(1, 'P1', 'P2')
    model.add_facts({
        p1.name: {
            '0': TRUE,
            '1': TRUE,
            '2': TRUE,
            '3': TRUE
        },
        p2.name: {
            '0': TRUE,
            '1': UNKNOWN,
            '2': FALSE,
            '3': FALSE,
        }
    })

    x = Variable('x')
    model['AB'] = And(p1(x), p2(x))
    model.add_labels({
        'AB': {
            '0': TRUE,
            '1': UNKNOWN,
            '2': TRUE,
            '3': FALSE,
        }
    })
    parameter_history = {'weights': True, 'bias': True}
    losses = ['logical', 'supervised']
    total_loss, _ = model.train(
        direction=UPWARD,
        losses=losses,
        parameter_history=parameter_history
    )
    model.print(params=True)
    predictions = model[p1.name].state().values()
    assert all([fact is TRUE for fact in predictions]), (
        'expected AB Facts to all be TRUE, received bounds '
        f'{[p.name for p in predictions]}')

    return model, total_loss, losses


if __name__ == "__main__":
    my_model, total_loss, losses = test()
    plot_loss(total_loss, losses)
    plot_params(my_model)
