##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ..activations.neuron.static import _StaticActivation

import torch


class Frechet(_StaticActivation):
    """Unweighted, Propositional Frechet"""

    def _and_upward(self, operand_bounds: torch.Tensor):
        lower_bd = max(0, sum(operand_bounds[0]) - 1)
        upper_bd = min(operand_bounds[1])
        tt = torch.tensor([lower_bd, upper_bd])
        return tt

    def _and_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AB = operator_bounds[0]
        upper_bd_AB = operator_bounds[1]

        new_lower_bd_B = max(lower_bd_B, lower_bd_AB)
        new_upper_bd_B = min(upper_bd_B, upper_bd_AB - lower_bd_A + 1)
        new_lower_bd_A = max(lower_bd_A, lower_bd_AB)
        new_upper_bd_A = min(upper_bd_A, upper_bd_AB - lower_bd_B + 1)

        tt = torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )
        return tt

    def _or_upward(self, operand_bounds: torch.Tensor):
        lower_bd = max(operand_bounds[0])
        upper_bd = min(sum(operand_bounds[1]), 1)
        tt = torch.tensor([lower_bd, upper_bd])
        return tt

    def _or_downward(self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AvB = operator_bounds[0]
        upper_bd_AvB = operator_bounds[1]

        new_lower_bd_B = max(lower_bd_B, lower_bd_AvB - upper_bd_A)
        new_upper_bd_B = min(upper_bd_B, upper_bd_AvB)
        new_lower_bd_A = max(lower_bd_A, lower_bd_AvB - upper_bd_B)
        new_upper_bd_A = min(upper_bd_A, upper_bd_AvB)

        tt = torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )
        return tt

    def _implies_upward(self, operand_bounds: torch.Tensor):
        # implemented as ~A v B
        lower_bd = max(1 - operand_bounds[1, 0], operand_bounds[0, 1])
        upper_bd = min(1, 1 - operand_bounds[0, 0] + operand_bounds[1, 1])
        tt = torch.tensor([lower_bd, upper_bd])
        return tt

    def _implies_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AimplB = operator_bounds[0]
        upper_bd_AimplB = operator_bounds[1]

        new_lower_bd_B = max(lower_bd_B, lower_bd_AimplB + lower_bd_A - 1)
        new_upper_bd_B = min(upper_bd_B, upper_bd_AimplB)
        new_lower_bd_A = max(lower_bd_A, 1 - upper_bd_AimplB)
        new_upper_bd_A = min(upper_bd_A, 1 + upper_bd_B - lower_bd_AimplB)

        tt = torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )
        return tt
