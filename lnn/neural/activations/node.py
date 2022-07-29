##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from typing import Tuple, List, Union

import torch
import numpy as np

from ..._utils import val_clamp
from ..parameters.node import _NodeParameters


"""
Node level activation
"""


class _NodeActivation(_NodeParameters):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.Yt = self.alpha
        self.Yf = 1 - self.alpha

    def aggregate_bounds(
        self,
        grounding_rows: List[int] = None,
        new_bounds: torch.Tensor = None,
        bound: str = None,
        **kwds,
    ) -> torch.Tensor:
        """Proof aggregation to tighten existing bounds towards new bounds

        **Parameters**

        bound: ['lower', 'upper', None]
            specify individual bound to aggregate from
            if None [default], then both 'lower' and 'upper' bounds aggregate
        """
        prev_bounds = self.get_facts(grounding_rows).clone()
        if kwds.get("logical_aggregation", False):
            raise NotImplementedError(
                "should not end here, logical" "aggregation not yet implemented"
            )
        else:
            L = (
                torch.max(prev_bounds[..., 0], new_bounds[..., 0])
                if (bound in [None, "lower"])
                else prev_bounds[..., 0]
            )
            U = (
                torch.min(prev_bounds[..., 1], new_bounds[..., 1])
                if (bound in [None, "upper"])
                else prev_bounds[..., 1]
            )
            aggregate = torch.stack([L, U], dim=-1)
        self.update_bounds(grounding_rows, val_clamp(aggregate))
        return (aggregate - prev_bounds).abs().sum()

    def output_regions(self, y: torch.Tensor) -> torch.Tensor:
        """classical region of outputs for the given node inputs

        Evaluates the classical region for a given y value (per bound)
            typically given as the output of a neuron activation function

        Regions:
            1 - False
            2 - Fuzzy False
            3 - Midpoint (0.5)
            4 - Fuzzy True
            5 - True

        """
        result = torch.zeros_like(y)
        result = result.masked_fill(y <= self.Yf, 1)
        result = result.masked_fill(bool_and(self.Yf < y, y < 0.5), 2)
        result = result.masked_fill(y == 0.5, 3)
        result = result.masked_fill(bool_and(0.5 < y, y < self.Yt), 4)
        result = result.masked_fill(self.Yt <= y, 5)
        if (result == 0).sum() > 0:
            raise Exception(f"output not in the feasible region [0, 1] for: {result}")
        return result

    def _get_state_vars(
        self, bounds: torch.Tensor = None
    ) -> Tuple[
        torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        bounds = self.get_facts() if bounds is None else bounds
        regions = self.output_regions(bounds).numpy().astype(dtype="<U3")
        L, U = regions[..., 0], regions[..., 1]
        result = np.zeros_like(L, dtype=float).astype(dtype="<U3")
        L_bounds, U_bounds = bounds[..., 0], bounds[..., 1]
        return bounds, result, L, U, L_bounds, U_bounds

    def is_contradiction(
        self,
        bounds: torch.Tensor = None,
        args: Tuple[
            torch.Tensor,
            np.ndarray,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] = None,
    ) -> torch.BoolTensor:
        """check which bounds are in contradiction
        classical contradiction removed from states: F, T"""
        *_, L, U, L_bounds, U_bounds = args if args else (self._get_state_vars(bounds))
        contradictions = bool_and(
            L_bounds > U_bounds,
            bool_and(L == "1.0", U == "1.0").logical_not(),
            bool_and(L == "5.0", U == "5.0").logical_not(),
        )
        return contradictions

    def state(self, bounds: torch.Tensor = None) -> np.ndarray:
        """classical state of formula bounds.

        Combines the output regions of Lower, Upper bounds to determine the
        overall node state and collapses the bounds dimension


        **Returns**

        numpy char array:
            classical states:
                'T' = True
                'F' = False
                'U' = Unknown
                'C' = Contradiction
            fuzzy states:
                '~T' = More True than not True
                '~F' = More False than not False
                '~U' = Unknown but not classical
                '=U' = Unknown, exact midpoint

        Warning
        -------
        Only works for output state of node bounds, not inputs to connectives

        """
        args = self._get_state_vars(bounds)
        bounds, result, L, U, *_ = args
        result = np.where(bool_and(L == "1.0", U == "5.0"), "U", result)
        result = np.where(bool_and(L == "1.0", U == "1.0"), "F", result)
        result = np.where(bool_and(L == "5.0", U == "5.0"), "T", result)
        result = np.where(bool_and(L == "3.0", U == "3.0"), "=U", result)
        result = np.where(self.is_contradiction(args=args), "C", result)
        result = np.where(bool_and(L in ["1.0", "2.0"], U == "2.0"), "~F", result)
        result = np.where(bool_and(L == "4.0", U in ["4.0", "5.0"]), "~T", result)
        result = np.where(
            bool_or(
                bool_and(L == "1.0", U in ["3.0", "4.0"]),
                bool_and(L == "2.0", U in ["3.0", "4.0", "5.0"]),
                bool_and(L == "3.0", U in ["4.0", "5.0"]),
            ),
            "~U",
            result,
        )
        if result == "0.0":
            raise Exception(f"bounds {L,U} fell in an unquantified state")
        return result


def tensorise(t: Union[bool, torch.Tensor]) -> torch.Tensor:
    return (
        t.clone().detach()
        if isinstance(t, torch.Tensor)
        else (torch.tensor(t).detach())
    )


def bool_and(*args: bool) -> torch.BoolTensor:
    return bool_tensor("and", *args)


def bool_or(*args: bool) -> torch.BoolTensor:
    return bool_tensor("or", *args)


def bool_tensor(func: str, *args: bool) -> torch.BoolTensor:
    """"""
    tensor = tensorise(args[0]).to(dtype=torch.int8)
    for a in args[1:]:
        if func == "and":
            tensor = tensor * a
        elif func == "or":
            tensor = tensor + a
    return tensor.type(torch.bool)
