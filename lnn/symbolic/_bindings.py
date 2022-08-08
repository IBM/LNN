##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##


import importlib


def funct_bindings(funct_binds):
    Function = getattr(importlib.import_module("lnn.symbolic.logic"), "Function")

    def check_instance(bind_key):
        if isinstance(funct_binds[1][bind_key], list):
            return funct_binds[1][bind_key]

        if isinstance(funct_binds[1][bind_key], Function):
            binds = [
                str(b) if b is not None else b
                for b in funct_binds[1][bind_key].groundings.values()
            ]
            return binds

        if isinstance(funct_binds[1][bind_key], tuple):
            return funct_bindings(funct_binds[1][bind_key])

        return None

    bindings = {
        bind_key: binding
        for bind_key in funct_binds[1]
        if (binding := check_instance(bind_key))
    }

    return [
        str(f_out)
        for f_in, f_out in funct_binds[0].groundings.items()
        if all([f_in[k] in bindings[k] for k in bindings])
    ]


def get_bindings(g):
    Function = getattr(importlib.import_module("lnn.symbolic.logic"), "Function")

    if isinstance(g, list):
        if isinstance(g[0], tuple):
            # A tuple with Function object and binding dict
            binds = funct_bindings(g[0])
            if binds:
                return binds
            else:
                return [None]
                # TODO: We should be able to handle an (initially)
                #  empty list of bindings.
        elif isinstance(g[0], Function):  # List of one Function
            binds = [str(b) if b is not None else b for b in g[0].groundings.values()]
            if binds:
                return binds
            else:
                return [None]
                # TODO: We should be able to handle an (initially)
                #  empty list of bindings.
        else:  # List of strings and/or _Grounding
            binds = [str(b) if b is not None else b for b in g]
            return binds
    else:
        if g is None:
            return g
        else:
            return str(g)
