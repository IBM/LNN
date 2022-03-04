##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Predicate, AXIOM, plot_graph


def test():
    # create empty model
    model = Model()

    # define the rules/knowledge
    Grecian = model["G"] = Predicate()
    Human = model["H"] = Predicate()
    Mortal = model["M"] = Predicate()
    Lived = model["L"] = Predicate()
    Alive = model["A"] = Predicate()

    formulae = [
        Grecian.Implies(Human),
        Human.Implies(Mortal),
        Mortal.Implies(Lived.Or(Alive)),
    ]
    model.add_formulae(*formulae, world=AXIOM)

    # perform inference/learning on the model
    model.infer(lifted=True)
    model.print()
    plot_graph(model)

    GT = ["(G → M)", "(H → (L ∨ A))", "(G → (L ∨ A))"]
    for gt in GT:
        assert (
            gt in model.nodes.keys()
        ), f"lifted preprocessing could not find {gt} in the model"


if __name__ == "__main__":
    test()
    print("success")
