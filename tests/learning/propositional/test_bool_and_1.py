##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, And, Proposition, World, Fact, Direction, Loss

import random

TRUE = Fact.TRUE
FALSE = Fact.FALSE
UNKNOWN = Fact.UNKNOWN


def test_1():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in the upward direction
    """
    # model
    model = Model()

    # rules
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, world=World.AXIOM)
    model.add_knowledge(AB)

    # facts
    model.add_data({A: TRUE, B: FALSE})

    # train/inference
    model.train(direction=Direction.UPWARD, losses={Loss.CONTRADICTION: 1})
    model.print(params=True)

    weights = AB.params("weights")
    bounds = B.state()
    assert (
        weights[1] <= 1 / 2
    ), f"expected input B to be downweighted <= 0., received {weights[1]}"
    assert bounds is FALSE, f"expected bounds to remain False, received {bounds}"


def test_2():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in both directions
    """
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    AB = And(A, B, world=World.AXIOM)
    model.add_knowledge(AB)
    model.add_data({A: TRUE, B: FALSE})
    model.train(losses={Loss.CONTRADICTION: 1})

    weights = AB.params("weights")
    bounds = B.state()
    assert (
        weights[1] <= 1 / 2
    ), f"expected input B to be downweighted <= 0., received {weights[1]}"
    assert bounds is FALSE, f"expected bounds to remain False, received {bounds}"


def test_3():
    """decrease weights for contradictory facts

    given And(A, B, C) - reduce the weight on B
    training in both directions
    """
    model = Model()
    A = Proposition("A")
    B = Proposition("B")
    C = Proposition("C")
    ABC = And(A, B, C, world=World.AXIOM)
    model.add_knowledge(ABC)
    model.add_data({A: TRUE, B: FALSE, C: TRUE})
    model.train(losses={Loss.CONTRADICTION: 1})

    weights = ABC.params("weights")
    bounds = B.state()
    assert (
        weights[1] <= 1 / 2
    ), f"expected input B to be downweighted <= 0., received {weights[1]}"
    assert bounds is FALSE, f"expected bounds to remain False, received {bounds}"


def test_multiple():
    """decrease weights for contradictory facts

    given And(n inputs) - reduce the weight on r random
    training in both directions
    """
    inputs = (10, 100, 1000)
    for n in inputs:
        r = n - 1
        model = Model()
        activation = {"alpha": 1 - 1e-5}
        prop = [f"P{i}" for i in range(n)]
        truths = [TRUE] * n
        truths[:r] = [FALSE] * r
        truths = random.sample(truths, n)
        from tqdm import tqdm

        props = []

        for idx, p in tqdm(
            enumerate(prop), desc="populating graph", total=n, disable=True
        ):
            props.append(Proposition(f"{p}", activation=activation))
            props[-1].add_data(truths[idx])
        _and = And(*props, world=World.AXIOM, activation=activation)
        model.add_knowledge(_and)
        model.train(losses=[Loss.CONTRADICTION], learning_rate=1e-1)
        model.print(params=True)

        # test operator bounds
        prediction = _and.state()
        assert prediction is TRUE, (
            f"received {prediction} " f"{_and.get_data().tolist()}"
        )

        # test operator weights
        false_idxs = [idx for idx, truth in enumerate(truths) if not truth]
        weights = _and.params("weights", detach=True).numpy()
        for w in weights[false_idxs]:
            assert (
                w <= 0.5 + 1e-5
            ), f"expected False input to be downweighted <= 1., received {w}"


def test_bias():
    """decrease bias for contradictory facts

    given a False And, for all True inputs
    """
    model = Model()
    n = 500
    prop = [f"P{i}" for i in range(n)]
    props = []
    for p in prop:
        props.append(Proposition(f"{p}"))
        props[-1].add_data(TRUE)
    _and = And(*props, world=World.FALSE, activation={"bias_learning": True})
    model.add_knowledge(_and)
    model.train(losses=[Loss.CONTRADICTION])
    assert _and.neuron.bias <= 1e-5, f"expected bias <= 0, received {_and.neuron.bias}"


def test_all():
    """decrease weights for contradictory facts

    given a False And, for all True inputs - drop all weights
    """
    model = Model()
    n = 1000
    prop = [f"P{i}" for i in range(n)]
    props = []
    for p in prop:
        props.append(Proposition(f"{p}"))
        props[-1].add_data(FALSE)
    _and = And(*props, world=World.AXIOM)
    model.add_knowledge(_and)
    model.train(losses={Loss.CONTRADICTION: 1})
    bounds = _and.state()
    assert bounds is TRUE, f"expected bounds to remain True, received {bounds}"
    weights = _and.params("weights")
    assert all(
        [w <= 0.5 + 1e-5 for w in weights]
    ), f"expected all inputs to be downweighted (±0.0), received {weights}"


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_multiple()
    test_bias()
    test_all()
