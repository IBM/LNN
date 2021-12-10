##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Predicate, bool_to_fact, plot_loss, UPWARD
import torch
import itertools
import numpy as np
import pandas as pd


def run():

    # load data from file
    data = pd.read_csv('train.csv', header=None)
    y = torch.from_numpy(data[0].astype('float32').to_numpy().reshape((-1, 1)))
    X = torch.from_numpy(data.drop(0, axis='columns').astype(
        'float32').to_numpy())
    preds = open('train').readline().split('\n')[0].split(',')[1:]

    # generate model
    model = Model()
    alpha = .95
    preds = list(map(Predicate, preds))
    model.add_formulae(*preds)

    P_list = [0, 10, 15]  # [4] [4,6,14] #range(len(preds))# [0,1,2,3] [4,6,14]
    P_list_named = [preds[P] for P in P_list]
    choose_list = [1, 2]  # [1,2] #[2]

    P_combination_list = []
    for r in choose_list:
        for combination in list(itertools.combinations(P_list_named, r=r)):
            P_combination_list.append(combination)

    sub_rules = []
    for P_combined in P_combination_list:

        combined = []
        for P_name in P_combined:
            combined.append(
                P_name.And(P_name.Not(),
                           neuron={
                               'alpha': alpha,
                               # 'alpha_min': True,
                               'bias_learning': False,
                               'weight_learning': False}))

        if len(P_combined) > 1:
            sub_rules.append(combined[0].And(*combined[1:],
                                             neuron={
                                                 'alpha': alpha,
                                                 # 'alpha_min': True,
                                                 # 'bias_learning': False
                                             }))
        else:
            sub_rules.append(combined[0])

    model['rule'] = sub_rules[0].Or(*sub_rules[1:],
                                    neuron={
                                        'alpha': alpha,
                                        # 'alpha_min': True,
                                        # 'bias_learning': False
                                    })

    # training labels
    for i in range(X.shape[0]):
        for P in P_list:
            model.add_facts({preds[P].name: {f'{i}': bool_to_fact(X[i, P])}})
        model.add_labels({'rule': {f'{i}': bool_to_fact(y[i, 0])}})

    # train model
    losses = {'supervised': 1,
              'logical': {'coeff': 1e-3, 'slacks': True},
              'contradiction': 0}
    total_loss, _ = model.train(
        direction=UPWARD,
        learning_rate=.1,
        epochs=100,
        losses=losses,
        pbar=True
    )
    plot_loss(total_loss, losses)
    print('bias:', model['rule'].neuron.bias.detach(),
          'weights:', model['rule'].neuron.weights.detach())
    print('*'*10, 'results', '*'*10)
    print('final loss', total_loss[0][-1])

    # training results
    model.flush()
    for i in range(X.shape[0]):
        for P in P_list:
            model.add_facts({preds[P].name: {str(i): bool_to_fact(X[i, P])}})
    model.infer(direction=UPWARD)

    predictions_train = []
    for person in range(X.shape[0]):
        predictions_train.append(
            model['rule'].get_facts(str(person)).detach().mean().round())
    print('train:',
          sum(np.array(predictions_train) == np.array(y[:, 0])), '/', len(y))

    # test results
    file_path = 'test.csv'
    data = pd.read_csv(file_path, header=None)
    y_hat = torch.from_numpy(
        data[0].astype('float32').to_numpy().reshape((-1, 1)))
    X_hat = torch.from_numpy(
        data.drop(0, axis='columns').astype('float32').to_numpy())

    model.reset_bounds()
    for i in range(X_hat.shape[0]):
        for P in P_list:
            model.add_facts({
                preds[P].name: {str(i): bool_to_fact(X_hat[i, P])}})
    model.infer(direction=UPWARD)

    predictions = []
    for person in range(X_hat.shape[0]):
        predictions.append(
            model['rule'].get_facts(str(person)).detach().mean().round())

    # predictions
    print('test:',
          sum(np.array(predictions) == np.array(y_hat[:, 0])), '/', len(y_hat))
    print('*'*27)


if __name__ == "__main__":
    run()
