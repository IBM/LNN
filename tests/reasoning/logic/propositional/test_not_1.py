##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Proposition, Not, Fact

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test_upward():
    GTs = [TRUE, FALSE, UNKNOWN]
    inputs = [FALSE, TRUE, UNKNOWN]
    for i in range(3):
        model = Model()
        A = Proposition("A")
        NotA = Not(A)
        model.add_knowledge(NotA)
        model.add_data({A: inputs[i]})
        NotA.upward()
        prediction = NotA.state()
        assert (
            prediction is GTs[i]
        ), f"expected Not({inputs[i]}) = {GTs[i]}, received {prediction}"


def test_downward():
    GTs = [TRUE, FALSE, UNKNOWN]
    inputs = [FALSE, TRUE, UNKNOWN]
    for i in range(3):
        model = Model()
        A = Proposition("A")
        NotA = Not(A)
        model.add_knowledge(NotA)
        model.add_data({NotA: inputs[i]})
        NotA.downward()
        prediction = A.state()
        assert (
            prediction is GTs[i]
        ), f"expected Not({inputs[i]}) = {GTs[i]}, received A={prediction}"


if __name__ == "__main__":
    test_upward()
    test_downward()
