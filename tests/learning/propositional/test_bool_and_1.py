##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import random

from lnn import Model, And, Proposition, World, UPWARD, FALSE, TRUE, CLOSED


def test_1():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in the upward direction
    """
    # model
    model = Model()

    # rules
    model["A"] = Proposition("A")
    model["B"] = Proposition("B")
    model["AB"] = And(model["A"], model["B"], world=World.AXIOM)

    # facts
    model.add_facts({"A": TRUE, "B": FALSE})

    # train/inference
    model.train(direction=UPWARD, losses={"contradiction": 1})

    weights = model["AB"].params("weights")
    bounds = model["B"].state()
    assert (
        weights[1] <= 1 / 2
    ), f"expected input B to be downweighted <= 0., received {weights[1]}"
    assert bounds is FALSE, f"expected bounds to remain False, received {bounds}"

    return model


def test_2():
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in both directions
    """
    model = Model()
    model["A"] = Proposition("A")
    model["B"] = Proposition("B")
    model["AB"] = And(
        model["A"],
        model["B"],
        world=World.AXIOM,
    )
    model.add_facts({"A": TRUE, "B": FALSE})
    model.train(losses={"contradiction": 1})

    weights = model["AB"].params("weights")
    bounds = model["B"].state()
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
    model["A"] = Proposition()
    model["B"] = Proposition()
    model["C"] = Proposition()
    model["and"] = And(
        model["A"],
        model["B"],
        model["C"],
        world=World.AXIOM,
    )
    model.add_facts({"A": TRUE, "B": FALSE, "C": TRUE})
    model.train(losses={"contradiction": 1})

    weights = model["and"].params("weights")
    bounds = model["B"].state()
    assert (
        weights[1] <= 1 / 2
    ), f"expected input B to be downweighted <= 0., received {weights[1]}"
    assert bounds is FALSE, f"expected bounds to remain False, received {bounds}"


def test_multiple(output=False):
    """decrease weights for contradictory facts

    given And(n inputs) - reduce the weight on r random
    training in both directions
    """
    inputs = (10, 100, 1000)
    for n in inputs:
        r = n - 1
        model = Model()
        neuron = {"alpha": 1 - 1e-5}
        prop = [f"P{i}" for i in range(n)]
        truths = [TRUE] * n
        truths[:r] = [FALSE] * r
        truths = random.sample(truths, n)
        from tqdm import tqdm

        for idx, p in tqdm(
            enumerate(prop), desc="populating graph", total=n, disable=True
        ):
            model[f"{p}"] = Proposition(neuron=neuron)
            model.add_facts({f"{p}": truths[idx]})
        model["and"] = And(
            *[model[f"{p}"] for p in prop], world=World.AXIOM, neuron=neuron
        )
        model.train(losses=["contradiction"], learning_rate=1e-1)

        if output:
            model.print(params=True)

        # test operator bounds
        prediction = model["and"].state()
        assert prediction is TRUE, (
            f"received {prediction} " f"{model['and'].get_facts().tolist()}"
        )

        # test operator weights
        false_idxs = [idx for idx, truth in enumerate(truths) if not truth]
        weights = model["and"].params("weights", detach=True).numpy()
        for w in weights[false_idxs]:
            assert (
                w <= 0.5 + 1e-5
            ), f"expected False input to be downweighted <= 1., received {w}"


def test_bias():
    """decrease bias for contradictory facts

    given a False And, for all True inputs
    """
    model = Model()
    n = 1000
    prop = [f"P{i}" for i in range(n)]
    for p in prop:
        model[f"{p}"] = Proposition()
        model.add_facts({f"{p}": TRUE})
    model["and"] = And(
        *[model[f"{p}"] for p in prop],
        world=CLOSED,
    )
    model.train(losses={"contradiction": 1})
    bias = model["and"].params("bias")
    assert bias <= 1e-5, f"expected bias <= 0, received {bias}"


def test_all():
    """decrease weights for contradictory facts

    given a False And, for all True inputs - drop all weights
    """
    model = Model()
    n = 1000
    prop = [f"P{i}" for i in range(n)]
    for p in prop:
        model[f"{p}"] = Proposition()
        model.add_facts({f"{p}": FALSE})
    model["and"] = And(
        *[model[f"{p}"] for p in prop],
        world=World.AXIOM,
    )
    model.train(losses={"contradiction": 1})
    bounds = model["and"].state()
    assert bounds is TRUE, f"expected bounds to remain True, received {bounds}"
    weights = model["and"].params("weights")
    assert all(
        [w <= 0.5 + 1e-5 for w in weights]
    ), f"expected all inputs to be downweighted (±0.0), received {weights}"


if __name__ == "__main__":
    model = test_1()
    model.print(params=True)
    test_2()
    test_3()
    test_multiple(output=True)
    test_bias()
    test_all()
    print("success")
