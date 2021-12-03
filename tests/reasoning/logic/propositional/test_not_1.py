##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Proposition, Not, TRUE, FALSE, UNKNOWN


def test_upward():
    GTs = [TRUE, FALSE, UNKNOWN]
    inputs = [FALSE, TRUE, UNKNOWN]
    for i in range(3):
        model = Model()
        model['not'] = Not(Proposition('A'))
        model.add_facts({'A': inputs[i]})
        model['not'].upward()
        prediction = model['not'].state()
        assert (prediction is GTs[i]), (
            f'expected Not({inputs[i]}) = {GTs[i]}, received {prediction}')


def test_downward():
    GTs = [TRUE, FALSE, UNKNOWN]
    inputs = [FALSE, TRUE, UNKNOWN]
    for i in range(3):
        model = Model()
        model['not'] = Not(Proposition('A'))
        model.add_facts({'not': inputs[i]})
        model['not'].downward()
        prediction = model['A'].state()
        assert (prediction is GTs[i]), (
            f'expected Not({inputs[i]}) = {GTs[i]}, received A={prediction}')


if __name__ == "__main__":
    test_upward()
    test_downward()
    print('success')
