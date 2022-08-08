##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ..activations.neuron.static import _StaticActivation

import torch


class Godel(_StaticActivation):
    """Unweighted, Propositional Godel"""

    def _and_upward(self, operand_bounds: torch.Tensor):
        return torch.tensor([min(operand_bounds[0]), min(operand_bounds[1])])

    def _and_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AB = operator_bounds[0]
        upper_bd_AB = operator_bounds[1]

        new_lower_bd_B = lower_bd_AB
        new_upper_bd_B = upper_bd_B
        if lower_bd_A > upper_bd_AB:
            new_upper_bd_B = upper_bd_AB

        new_lower_bd_A = lower_bd_AB
        new_upper_bd_A = upper_bd_A
        if lower_bd_B > upper_bd_AB:
            new_upper_bd_A = upper_bd_AB

        return torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )

    def _or_upward(self, operand_bounds: torch.Tensor):
        return torch.tensor([max(operand_bounds[0]), max(operand_bounds[1])])

    def _or_downward(self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AvB = operator_bounds[0]
        upper_bd_AvB = operator_bounds[1]

        new_upper_bd_B = upper_bd_AvB
        new_lower_bd_B = lower_bd_B
        if upper_bd_A < lower_bd_AvB:
            new_lower_bd_B = lower_bd_AvB

        new_upper_bd_A = upper_bd_AvB
        new_lower_bd_A = lower_bd_A
        if upper_bd_B < lower_bd_AvB:
            new_lower_bd_A = lower_bd_AvB

        return torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )

    def _implies_upward(self, operand_bounds: torch.Tensor):
        return torch.tensor(
            [
                max(1 - operand_bounds[0, 0], operand_bounds[0, 1]),
                max(1 - operand_bounds[1, 0], operand_bounds[1, 1]),
            ]
        )

    def _implies_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AimplB = operator_bounds[0]
        upper_bd_AimplB = operator_bounds[1]

        new_upper_bd_B = upper_bd_AimplB
        new_lower_bd_B = lower_bd_B
        if 1 - lower_bd_A < lower_bd_AimplB:
            new_lower_bd_B = lower_bd_AimplB

        new_upper_bd_A = upper_bd_A
        new_lower_bd_A = 1 - lower_bd_AimplB
        if upper_bd_B < lower_bd_AimplB:
            new_lower_bd_A = 1 - upper_bd_AimplB

        return torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )
