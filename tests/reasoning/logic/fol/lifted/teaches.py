##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, Predicate, Variable, Implies, And, Not, World
from lnn._utils import add_checkpoint, unpack_checkpoints


def test():
    add_checkpoint("start")
    model = Model()
    t = Variable("t", type="teacher")
    s = Variable("s", type="student")
    c = Variable("c", type="course")
    b = Variable("b", type="company")

    # define the rules (knowledge)
    Teaches = Predicate("Teaches", arity=2)
    Takes = Predicate("Takes", arity=2)
    JobOffers = Predicate("JobOffers", arity=2)
    formulae = [
        Teaches,
        Takes,
        Implies(And(Teaches(t, c), Takes(s, c)), JobOffers(s, b)),
        Not(JobOffers(s, b)),
    ]
    model.add_knowledge(*formulae, world=World.AXIOM)
    add_checkpoint("build model")

    # reasoning
    model.lift()
    add_checkpoint("lifted reasoning")
    model.print()

    prediction = JobOffers.world
    assert prediction is World.CONTRADICTION, (
        f"expected JobOffers to provide proof by contradiction, received "
        f"{prediction}"
    )
    print(unpack_checkpoints())


if __name__ == "__main__":
    test()
    print("success")
