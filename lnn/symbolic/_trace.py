##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##


def increment_param_history(self, **params):
    if not hasattr(self, "parameter_history"):
        self.parameter_history = {}
    params = params.get("parameter_history")
    for param in params:
        split = param.split(".grad")
        p = split[0]
        grad = False
        if len(split) == 2:
            grad = True
        if (params[p] is True or params[p] is self) and hasattr(self.neuron, p):
            if param not in self.parameter_history:
                self.parameter_history[param] = []
            attrib = getattr(self.neuron, p)
            val = attrib.grad if grad else attrib
            self.parameter_history[param].append(
                val.tolist()
                if val is not None
                else (attrib.clone().detach() * 0).tolist()
            )
