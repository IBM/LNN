##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ..activations.neuron.static import _StaticActivation

import torch
import math
from itertools import chain, combinations


class Product(_StaticActivation):
    """Unweighted, Propositional Product"""

    def _and_upward(self, operand_bounds: torch.Tensor):
        tt = torch.tensor([math.prod(operand_bounds[0]), math.prod(operand_bounds[1])])
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

        new_lower_bd_A = lower_bd_A
        new_lower_bd_B = lower_bd_B
        if upper_bd_A > 0:
            new_lower_bd_B = max(lower_bd_B, lower_bd_AB / upper_bd_A)
        if upper_bd_B > 0:
            new_lower_bd_A = max(lower_bd_A, lower_bd_AB / upper_bd_B)

        new_upper_bd_A = upper_bd_A
        new_upper_bd_B = upper_bd_B
        if lower_bd_A > 0:
            new_upper_bd_B = min(upper_bd_B, upper_bd_AB / lower_bd_A)
        if lower_bd_B > 0:
            new_upper_bd_A = min(upper_bd_A, upper_bd_AB / lower_bd_B)

        tt = torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )
        return tt

    def _or_upward(self, operand_bounds: torch.Tensor):
        lower_bd_subsets = chain.from_iterable(
            combinations(operand_bounds[0], r)
            for r in range(1, len(operand_bounds[0]) + 1)
        )
        upper_bd_subsets = chain.from_iterable(
            combinations(operand_bounds[1], r)
            for r in range(1, len(operand_bounds[1]) + 1)
        )
        lower_bd_sum = 0
        for subset in lower_bd_subsets:
            subset_sum = sum(subset)
            if len(subset) % 2 == 1:
                lower_bd_sum += subset_sum
            else:
                lower_bd_sum -= subset_sum

        upper_bd_sum = 0
        for subset in upper_bd_subsets:
            subset_sum = sum(subset)
            if len(subset) % 2 == 1:
                upper_bd_sum += subset_sum
            else:
                upper_bd_sum -= subset_sum

        tt = torch.tensor([lower_bd_sum, upper_bd_sum])
        return tt

    def _or_downward(self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AvB = operator_bounds[0]
        upper_bd_AvB = operator_bounds[1]

        new_upper_bd_B = upper_bd_B
        new_lower_bd_B = lower_bd_B
        new_upper_bd_A = upper_bd_A
        new_lower_bd_A = lower_bd_A

        if upper_bd_A < 1:
            new_lower_bd_B = max(
                lower_bd_B, (lower_bd_AvB - upper_bd_A) / (1 - upper_bd_A)
            )
        if upper_bd_B < 1:
            new_lower_bd_A = max(
                lower_bd_A, (lower_bd_AvB - upper_bd_B) / (1 - upper_bd_B)
            )

        if lower_bd_A < 1:
            new_upper_bd_B = min(
                upper_bd_B, max(0, (upper_bd_AvB - lower_bd_A) / (1 - lower_bd_A))
            )
        if lower_bd_B < 1:
            new_upper_bd_A = min(
                upper_bd_A, max(0, (upper_bd_AvB - lower_bd_B) / (1 - lower_bd_B))
            )

        tt = torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )
        return tt

    def _implies_upward(self, operand_bounds: torch.Tensor):
        lower_bd_A = operand_bounds[0, 0]
        lower_bd_B = operand_bounds[0, 1]
        upper_bd_A = operand_bounds[1, 0]
        upper_bd_B = operand_bounds[1, 1]
        lower_bd_AimplB = 0
        if upper_bd_A > 0:
            lower_bd_AimplB = min(1, lower_bd_B / upper_bd_A)
        upper_bd_AimplB = 1
        if lower_bd_A > 0:
            upper_bd_AimplB = min(1, upper_bd_B / lower_bd_A)
        tt = torch.tensor([lower_bd_AimplB, upper_bd_AimplB])
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

        new_upper_bd_B = upper_bd_B
        new_lower_bd_B = max(lower_bd_B, lower_bd_A * lower_bd_AimplB)
        if upper_bd_AimplB < 1:
            new_upper_bd_B = min(upper_bd_B, upper_bd_A * upper_bd_AimplB)

        new_upper_bd_A = upper_bd_A
        new_lower_bd_A = lower_bd_A
        if upper_bd_AimplB > 0 and upper_bd_AimplB < 1:
            new_lower_bd_A = max(lower_bd_A, min(1, lower_bd_B / upper_bd_AimplB))
        if lower_bd_AimplB > 0:
            new_upper_bd_A = min(upper_bd_A, min(1, upper_bd_B / lower_bd_AimplB))

        tt = torch.tensor(
            [[new_lower_bd_A, new_lower_bd_B], [new_upper_bd_A, new_upper_bd_B]]
        )
        return tt
