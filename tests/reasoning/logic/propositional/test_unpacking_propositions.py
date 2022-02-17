##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Proposition, Model


def test():
    model = Model()
    model["Smokes"] = Proposition()
    model["Friends"] = Proposition()

    smokes, friends = model.nodes.values()
    assert smokes.name == "Smokes", "Didn't get smokes proposition"
    assert friends.name == "Friends", "Didn't get friends proposition"
