##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from ...constants import Direction
from ..._utils import negate_bounds as _not
from ..activations.neuron.static import _StaticActivation

import torch


class Product(_StaticActivation):
    """
    Weighted Product T-norm activation.

    Implements AND, OR, and Implies using the Product T-norm.
    - AND: exp(sum(w_i * log(x_i + eps))) for each bound separately.
    - OR: Derived via De Morgan's Laws using AND.
    - Implies: Direct formula (Product‑Power Implication).
    """

    def __init__(self, num_inputs=2, **kwds):
        super().__init__(**kwds)
        self.weights = torch.nn.Parameter(torch.ones(num_inputs))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------------
    # AND
    # ------------------------------------------------------------------------
    def _and_upward(self, operand_bounds: torch.Tensor):
        """
        Upward pass for Product AND.
        Computes product over inputs for both lower and upper bounds,
        returning an interval [lower, upper].
        """
        eps = 1e-8

        # Determine input shape: [batch, num_inputs, 2] or [batch, num_inputs] or ...
        if operand_bounds.dim() == 3 and operand_bounds.shape[-1] == 2:
            # Interval inputs: separate lower and upper
            lower = operand_bounds[..., 0]  # [batch, num_inputs]
            upper = operand_bounds[..., 1]  # [batch, num_inputs]
        else:
            # Point inputs: treat as degenerate interval (lower == upper)
            # Ensure we have at least 2D: (batch, num_inputs)
            if operand_bounds.dim() == 1:
                operand_bounds = operand_bounds.unsqueeze(0)
            lower = operand_bounds
            upper = operand_bounds  # same values

        # Compute log of safe values
        safe_lower = torch.clamp(lower, min=eps)
        safe_upper = torch.clamp(upper, min=eps)

        log_lower = torch.log(safe_lower)
        log_upper = torch.log(safe_upper)

        # Weighted sums over inputs
        lower_weighted = (self.weights * log_lower).sum(dim=-1)
        upper_weighted = (self.weights * log_upper).sum(dim=-1)

        # Exponentiate and clamp
        lower_res = torch.clamp(torch.exp(lower_weighted), min=0.0, max=1.0)
        upper_res = torch.clamp(torch.exp(upper_weighted), min=0.0, max=1.0)

        # Stack to form intervals: [batch, 2]
        result = torch.stack([lower_res, upper_res], dim=-1)

        # If input had no batch dimension, we might have shape [1,2]; squeeze if needed
        # but leave as [batch,2] for consistency
        return result

    def _and_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        """Exact algebraic inverse of Product AND. No clamping."""
        eps = 1e-8

        # Ensure intervals
        if operand_bounds.dim() == 3 and operand_bounds.shape[-1] == 2:
            operand_bounds = operand_bounds[..., 0]  # use lower bound for inverse
        elif operand_bounds.dim() == 2 and operand_bounds.shape[-1] == 2:
            operand_bounds = operand_bounds[..., 0]
        else:
            operand_bounds = operand_bounds

        if operator_bounds.dim() == 2 and operator_bounds.shape[-1] == 2:
            operator_bounds = operator_bounds[..., 0]
        elif operator_bounds.dim() == 1:
            operator_bounds = operator_bounds

        # Ensure batch dimension for broadcasting
        if operand_bounds.dim() == 1:
            operand_bounds = operand_bounds.unsqueeze(0)
        if operator_bounds.dim() == 1:
            operator_bounds = operator_bounds.unsqueeze(0)

        log_operand = torch.log(operand_bounds + eps)
        log_operator = torch.log(operator_bounds + eps)

        total_weighted_log = (self.weights * log_operand).sum(dim=-1, keepdim=True)
        weighted_log_current = self.weights * log_operand
        sum_excluding_current = total_weighted_log - weighted_log_current

        log_operator_expanded = log_operator.expand_as(operand_bounds)
        log_target = (log_operator_expanded - sum_excluding_current) / self.weights

        zero_weight_mask = (self.weights == 0)
        if zero_weight_mask.any():
            mask_expanded = zero_weight_mask.expand_as(operand_bounds)
            result = torch.where(mask_expanded, operand_bounds, torch.exp(log_target))
        else:
            result = torch.exp(log_target)

        # Return as interval (lower bound only; upper bound same as lower for point inversion)
        # For consistency, we return a 2D tensor [batch, 2]
        result = torch.stack([result, result], dim=-1)
        return result

    # ------------------------------------------------------------------------
    # OR (derived via De Morgan's Laws)
    # ------------------------------------------------------------------------
    def _or_upward(self, operand_bounds: torch.Tensor):
        """Upward pass: OR = 1 - AND(1 - x, 1 - y)."""
        # Negate inputs: 1 - operand_bounds, compute AND, then negate result.
        # We need to handle intervals: negate both bounds.
        if operand_bounds.dim() == 3 and operand_bounds.shape[-1] == 2:
            neg = 1 - operand_bounds  # [batch, num_inputs, 2]
        else:
            # assume point values, treat as intervals
            if operand_bounds.dim() == 1:
                operand_bounds = operand_bounds.unsqueeze(0)
            neg = torch.stack([1 - operand_bounds, 1 - operand_bounds], dim=-1)

        # Compute AND on negated bounds using the same _and_upward
        and_result = self._and_upward(neg)  # returns [batch, 2]
        # Negate result to get OR
        or_result = 1 - and_result
        return torch.clamp(or_result, min=0.0, max=1.0)

    def _or_downward(self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor):
        """Downward pass for OR via De Morgan's."""
        # Ensure intervals for negation
        if operator_bounds.dim() == 1:
            operator_bounds = operator_bounds.unsqueeze(-1).repeat(1, 2)
        if operand_bounds.dim() == 2 and operand_bounds.shape[-1] != 2:
            operand_bounds = operand_bounds.unsqueeze(-1).repeat(1, 1, 2)

        neg_operator = _not(operator_bounds)
        neg_operand = _not(operand_bounds, dim=-1)
        and_result = self._and_downward(neg_operator, neg_operand)
        return _not(and_result, dim=-1)

    # ------------------------------------------------------------------------
    # IMPLIES (Product-Power Implication)
    # ------------------------------------------------------------------------
    def _implies_upward(self, operand_bounds: torch.Tensor):
        """Upward pass: x -> y = 1 - x^w_ant * (1 - y)^w_con."""
        eps = 1e-8

        # operand_bounds shape: (batch, 2, 2) or (batch, 2) for point values
        if operand_bounds.dim() == 3 and operand_bounds.shape[-1] == 2:
            # intervals: separate lower/upper for lhs and rhs
            lhs_lower = operand_bounds[..., 0, 0]
            lhs_upper = operand_bounds[..., 0, 1]
            rhs_lower = operand_bounds[..., 1, 0]
            rhs_upper = operand_bounds[..., 1, 1]
        else:
            # point values: treat as equal bounds
            if operand_bounds.dim() == 1:
                operand_bounds = operand_bounds.unsqueeze(0)
            lhs_val = operand_bounds[..., 0]
            rhs_val = operand_bounds[..., 1]
            lhs_lower = lhs_upper = lhs_val
            rhs_lower = rhs_upper = rhs_val

        w_ant, w_con = self.weights[0], self.weights[1]

        # Compute implication for lower and upper bounds separately
        safe_lhs_lower = torch.clamp(lhs_lower, min=eps)
        safe_lhs_upper = torch.clamp(lhs_upper, min=eps)
        safe_rhs_lower = torch.clamp(rhs_lower, min=eps)
        safe_rhs_upper = torch.clamp(rhs_upper, min=eps)

        # lower result: 1 - lhs_upper^w_ant * (1 - rhs_lower)^w_con
        # upper result: 1 - lhs_lower^w_ant * (1 - rhs_upper)^w_con
        lower_res = 1 - (safe_lhs_upper ** w_ant) * ((1 - safe_rhs_lower) ** w_con)
        upper_res = 1 - (safe_lhs_lower ** w_ant) * ((1 - safe_rhs_upper) ** w_con)

        # Clamp and stack
        lower_res = torch.clamp(lower_res, min=0.0, max=1.0)
        upper_res = torch.clamp(upper_res, min=0.0, max=1.0)
        result = torch.stack([lower_res, upper_res], dim=-1)  # [batch, 2]
        return result

    def _implies_downward(
        self, operator_bounds: torch.Tensor, operand_bounds: torch.Tensor
    ):
        """Downward pass for Product-Power Implication."""
        eps = 1e-8

        # Similar to upward, extract lower bounds for inversion
        if operand_bounds.dim() == 3 and operand_bounds.shape[-1] == 2:
            lhs = operand_bounds[..., 0, 0]
            rhs = operand_bounds[..., 1, 0]
        else:
            if operand_bounds.dim() == 1:
                operand_bounds = operand_bounds.unsqueeze(0)
            lhs = operand_bounds[..., 0]
            rhs = operand_bounds[..., 1]

        if operator_bounds.dim() == 2 and operator_bounds.shape[-1] == 2:
            z = operator_bounds[..., 0]
        else:
            z = operator_bounds

        w_ant = torch.clamp(self.weights[0], min=1e-6)
        w_con = torch.clamp(self.weights[1], min=1e-6)

        safe_rhs = torch.clamp(rhs, min=eps)
        safe_lhs = torch.clamp(lhs, min=eps)
        safe_one_minus_z = torch.clamp(1 - z, min=eps)

        lhs_inv = (safe_one_minus_z / (safe_rhs ** w_con + eps)) ** (1 / w_ant)
        rhs_inv = 1 - (safe_one_minus_z / (safe_lhs ** w_ant + eps)) ** (1 / w_con)

        # Stack results as interval (lower=upper)
        result = torch.stack([lhs_inv, rhs_inv], dim=-1)
        return result

    # ------------------------------------------------------------------------
    # ALIASES for symbolic node "ProductAnd"
    # ------------------------------------------------------------------------
    def _productand_upward(self, *args, **kwargs):
        return self._and_upward(*args, **kwargs)

    def _productand_downward(self, *args, **kwargs):
        return self._and_downward(*args, **kwargs)