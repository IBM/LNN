##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, And, Model, TRUE


def test():
    for name in ["A^B", "A∧B", "A&B"]:
        """allow unicode characters in names"""
        A = Proposition("A")
        B = Proposition("B")
        AB = And(A, B, name=name)  # This name should be accepted
        assert AB is not None

        formulae = [AB]
        facts = {"A": TRUE, "B": TRUE}
        model = Model()
        model.add_formulae(*formulae)
        model.add_facts(facts)
        model[name].upward()

        prediction = model[name].state()
        assert prediction is TRUE, "😔"


if __name__ == "__main__":
    test()
    print("success")
