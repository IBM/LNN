##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Model, truth_table


def test_1(output=False):
    """decrease weights for contradictory facts

    given And(A, B) - reduce the weight on B
    training in the upward direction
    """
    model = Model()
    A, B = model.add_propositions("A", "B")
    AB = A.Bidirectional(B, name="A <-> B")
    model.add_formulae(AB)
    for rows in truth_table(2):
        model.add_facts(
            {
                A.name: rows[0],
                B.name: rows[1],
            }
        )
        model.upward()
        if output:
            model[AB.name].print()
        model.flush()


if __name__ == "__main__":
    test_1(output=True)
    print("success")
