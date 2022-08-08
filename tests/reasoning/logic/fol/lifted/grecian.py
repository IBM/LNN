##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn._utils import add_checkpoint, unpack_checkpoints, reset_checkpoints
from lnn import Model, Predicate, World, Variable, Implies, ForAll


def test():
    print(f'\nchecking {"*" * 10} Lifted Transitivity {"*" * 10}')
    add_checkpoint("start")
    # create empty model
    model = Model()
    x = Variable("x")

    # define the rules/knowledge
    Grecian = Predicate("Grecian")
    Human = Predicate("Human")
    Mortal = Predicate("Mortal")
    model.add_knowledge(
        ForAll(Implies(Grecian(x), Human(x))),
        ForAll(Implies(Human(x), Mortal(x))),
    )
    model.set_query(ForAll(Implies(Grecian(x), Mortal(x))))
    add_checkpoint("build model")

    # perform inference/learning/lifting on the model
    model.lift()
    add_checkpoint("lifted reasoning")

    assert (
        model.query.world_state() is World.AXIOM
    ), f"expected {model.query.name} as AXIOM, received {model.query.world_state()}"
    print(unpack_checkpoints())
    reset_checkpoints()


if __name__ == "__main__":
    test()
    print("success")
