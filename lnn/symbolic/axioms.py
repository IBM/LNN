##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ..constants import World
from ..symbolic import Implies
from typing import Tuple


def lifted_axioms() -> dict:
    result = dict()

    def hypothetical_syllogism(formulae: Tuple[Implies, Implies]):
        """
        ((p ➞ q) ∧ (q ➞ r)) ➞ (p ➞ r)
        """
        if all(isinstance(f, Implies) for f in formulae):
            p, q_0 = formulae[0].operands
            q_1, r = formulae[1].operands
            if q_0 is q_1:
                return p.Implies(r, world=World.AXIOM)
    result[hypothetical_syllogism] = 2
    return result
