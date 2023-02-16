##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##


def get_bindings(g):
    if isinstance(g, list):
        if isinstance(g[0], tuple):
            # A tuple with Function object and binding dict
            return g[0]
        else:  # List of strings and/or tuple[str]
            binds = [str(b) if b is not None else b for b in g]
            return binds
    else:
        if g is None:
            return g
        else:
            return str(g)
