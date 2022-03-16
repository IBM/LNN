##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import pytest

from lnn import Predicate, And, Or, Implies, Bidirectional


@pytest.mark.parametrize("CompoundFormulae", [And, Or, Implies, Bidirectional])
def test_predicate_called(CompoundFormulae):
    P = Predicate("P", arity=0)

    with pytest.raises(ValueError):
        CompoundFormulae(P, P())

    with pytest.raises(ValueError):
        CompoundFormulae(P(), P)

    with pytest.raises(ValueError):
        CompoundFormulae(P, P)
