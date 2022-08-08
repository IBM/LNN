##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Or, Proposition, Fact, Direction, World, Loss

TRUE = Fact.TRUE
FALSE = Fact.FALSE


def test_upward():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    """
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    AB = Or(A, B, world=World.FALSE)
    model.add_knowledge(AB)
    model.add_data({A: TRUE, B: FALSE})
    model.train(direction=Direction.UPWARD, losses={Loss.CONTRADICTION: 0.1})

    weights = AB.params("weights")
    bounds = A.state()
    assert (
        weights[0] <= 0.5
    ), f"expected input A to be downweighted <= 0., received {weights[0]}"
    assert bounds is TRUE, f"expected bounds to remain True, received {bounds}"


if __name__ == "__main__":
    test_upward()
    print("success")
