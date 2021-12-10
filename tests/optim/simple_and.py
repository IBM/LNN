from lnn import Model, TRUE, FALSE, DDLukasiewicz, And, UPWARD, \
    plot_loss, plot_params


def test_constraints_in_loss():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in both directions
    """
    model = Model()
    neuron = {
        'type': DDLukasiewicz,
        'alpha': .55
    }
    A, B = model.add_propositions('A', 'B', neuron=neuron)
    AB = model['AB'] = And(A, B, neuron=neuron)
    model.add_facts({
        'A': TRUE,
        'B': FALSE
    })
    model.add_labels({
        'AB': TRUE
    })

    parameter_history = {'bias': True, 'weights': True}
    losses = {'supervised': 1, 'logical': 1e-2}
    total_loss, _ = model.train(
        losses=losses,
        direction=UPWARD,
        parameter_history=parameter_history
    )

    a, w, b = AB.params('alpha', 'weights', 'bias')
    model.print(params=True)
    print(a.tolist())
    # plot_loss(total_loss, losses)
    # plot_params(model)
    # bounds = model['B'].state()
    # assert weights[1] <= 1/2, (
    #     f'expected input B to be downweighted <= 0., received {weights[1]}')
    # assert bounds is FALSE, (
    #     f'expected bounds to remain False, received {bounds}')


if __name__ == "__main__":
    test_constraints_in_loss()
    print('success')
