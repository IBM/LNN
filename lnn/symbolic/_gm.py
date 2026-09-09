##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import itertools
import functools as ft
from typing import Set, Tuple, TypeVar, Union

from ._bindings import get_bindings
from ..constants import Direction

import torch
import pandas as pd
import numpy as np

"""
Grounding management module

All functions in this module assume a Formula level scope
"""

Formula = TypeVar("Formula")


def upward_bounds(
    operator: Formula,
    operands: Tuple[Formula],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[None, Tuple[torch.Tensor, None], Tuple[torch.Tensor, Set[tuple[str]]]]:
    r"""Returns (input_bounds, groundings) if new groundings are detected."""

    if operator.propositional and _is_contradiction(operator, operands):
        return None

    result = _operational_bounds(operator, Direction.UPWARD, operands, groundings)

    if result is None:
        return None

    if operator.propositional and operator.is_contradiction(result[0], stacked=True):
        return None

    input_bounds, groundings = result

    if groundings is None:
        return input_bounds, None

    contradictions = operator.contradicting_bounds(input_bounds, stacked=True)
    contradicting_groundings = [g for g, c in zip(groundings, contradictions) if c]

    if len(contradicting_groundings) == len(groundings):
        return None

    for g in contradicting_groundings:
        groundings.remove(g)

    input_bounds = input_bounds[~contradictions]

    return input_bounds, groundings


def downward_bounds(
    operator: Formula,
    operands: Tuple[Formula],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[
    None,
    Tuple[torch.Tensor, torch.Tensor, None],
    Tuple[torch.Tensor, torch.Tensor, Set[tuple[str]]],
]:
    r"""Returns (output_bounds, input_bounds, groundings) if new groundings detected."""
    if operator.propositional and _is_contradiction(operator, operands):
        return None

    result = _operational_bounds(operator, Direction.DOWNWARD, operands, groundings)

    if result is None:
        return None

    if operator.propositional and (
        operator.is_contradiction(result[0])
        or operator.is_contradiction(result[1], stacked=True)
    ):
        return None

    output_bounds, input_bounds, groundings = result

    if groundings is None:
        return output_bounds, input_bounds, None

    input_contradictions = operator.contradicting_bounds(input_bounds, stacked=True)
    output_contradictions = operator.contradicting_bounds(output_bounds)
    contradictions = torch.logical_or(input_contradictions, output_contradictions)
    contradicting_groundings = [g for g, c in zip(groundings, contradictions) if c]

    if len(contradicting_groundings) == len(groundings):
        return None

    for g in contradicting_groundings:
        groundings.remove(g)

    output_bounds = output_bounds[~contradictions]
    input_bounds = input_bounds[~contradictions]

    return output_bounds, input_bounds, groundings


def is_grounding_in_bindings(
    operator: Formula,
    operand_idx: int,
    operand_grounding: tuple[str],
) -> bool:
    return all(
        (
            True
            if operator.bindings[operand_idx][slot] == [None]
            else (
                operand_grounding.partial_grounding[slot]
                in operator.bindings[operand_idx][slot]
            )
        )
        for slot in range(len(operator.bindings[operand_idx]))
    )


def _operational_bounds(
    operator: Formula,
    direction: Direction,
    operands: Tuple[Formula, ...],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[
    None,
    Tuple[torch.Tensor, None],
    Tuple[torch.Tensor, torch.Tensor, None],
    Tuple[torch.Tensor, Set[tuple[str]]],
    Tuple[torch.Tensor, torch.Tensor, Set[tuple[str]]],
]:
    if operator.propositional:
        return _propositional_bounds(operator, direction, operands)

    # FOL
    is_homogenous = all([v == operator.var_remap[0] for v in operator.var_remap])

    if is_homogenous and not operator._has_bindings():
        return _fol_bounds(operator, direction, operands, groundings)

    operand_dfs, bindings = _get_operand_dfs(operator)

    if len(operand_dfs) == 0:
        return None
    elif len(operand_dfs) == 1:
        joined = operand_dfs[0]
    else:
        joined = ft.reduce(_full_outer_join, operand_dfs)

    operator_groundings = _operator_groundings(joined, operator)
    ground_objects = _operand_groundings(joined, operator, bindings)

    if ground_objects is None:
        return None

    facts = list()
    propositional_operands = list()
    max_ground_objects_len = 0
    for i, op in enumerate(operands):
        facts.append(op.get_data(*ground_objects[i]))
        if len(ground_objects[i]) > max_ground_objects_len:
            max_ground_objects_len = len(ground_objects[i])
        if op.propositional:
            propositional_operands.append(i)

    for i in propositional_operands:
        facts[i] = facts[i].repeat(max_ground_objects_len, 1)

    input_bounds = torch.stack(facts, dim=-1)

    operator._add_groundings(*operator_groundings)
    if direction is Direction.UPWARD:
        return input_bounds, operator_groundings
    output_bounds = operator.get_data(*operator_groundings)
    if len(output_bounds) == 0:
        return
    return output_bounds, input_bounds, operator_groundings


def _is_contradiction(formula: Formula, operands: Tuple[Formula, ...]) -> bool:
    operand_contradiction = any([op.is_contradiction() for op in operands])
    formula_contradiction = formula.is_contradiction()

    return operand_contradiction or formula_contradiction


def _get_operator_df(operator: Formula):
    columns = operator.unique_var_map.values()
    groundings = [(g,) if isinstance(g, str) else g for g in operator.groundings]
    return pd.DataFrame(groundings, columns=columns)


def _propositional_bounds(
    operator: Formula,
    direction: Direction,
    operands: Tuple[Formula, ...],
):
    input_bounds = torch.stack([op.get_data() for op in operands], dim=-1)
    if direction is Direction.UPWARD:
        return input_bounds, None

    return operator.get_data(), input_bounds, None


def _fol_bounds(
    operator: Formula,
    direction: Direction,
    operands: Tuple[Formula, ...],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
):
    if groundings is None:
        grounding_tables = [op.grounding_table for op in operands]
        groundings = set(itertools.chain.from_iterable(grounding_tables))

        if direction is Direction.DOWNWARD:
            groundings.update(operator.grounding_table.keys())
    else:
        groundings = set(map(operator._ground, groundings))

        # Propagate groundings to all children.
    for op in operands:
        op._add_groundings(*groundings)

    groundings = _hash_join(operands, groundings)
    input_bounds = torch.stack([op.get_data(*groundings) for op in operands], dim=-1)
    operator._add_groundings(*groundings)

    if direction is Direction.UPWARD:
        return input_bounds, groundings

    output_bounds = operator.get_data(*groundings)

    if len(output_bounds) == 0:
        return

    return output_bounds, input_bounds, groundings


def _eval_grounding(grounding):
    if grounding.name[0] != "(":
        return eval("('" + grounding.name + "',)")

    return eval(grounding.name)


def _get_operand_dfs(operator: Formula):
    operand_dfs = []
    operand_bindings = []

    for i, operand in enumerate(operator.operands):
        if operator.propositional:
            continue

        if operator.bindings[i]:
            bindings = [
                get_bindings(grounding)[0] for grounding in operator.bindings[i]
            ]
        else:
            bindings = []

        operand_bindings.append(bindings)

        if not operator.operand_map[i]:
            continue

        groundings = [(g,) if isinstance(g, str) else g for g in operand.groundings]

        if len(groundings):
            groundings = np.array(groundings)
            df = pd.DataFrame(groundings, columns=range(0, len(bindings)))
        else:
            df = pd.DataFrame(columns=range(0, len(bindings)))

        for j, binding in enumerate(bindings):
            if binding:
                df = df[df[j] == binding]
                df.drop(columns=[j], inplace=True)

        df.columns = list(operator.operand_map[i])
        operand_dfs.append(df)

    return operand_dfs, operand_bindings


def _outer_join(left, right):
    joinable_columns = left.columns.intersection(right.columns)
    if len(joinable_columns):
        joined = pd.merge(left, right, how="outer")
    else:
        joined = _full_outer_join(left, right)
    joined.dropna(inplace=True)
    joined.drop_duplicates(inplace=True)
    return joined


def _full_outer_join(first_df, next_df):
    joined = pd.merge(first_df, next_df, how="cross")

    if len(joined) == 0:
        joined = pd.concat([first_df, next_df])
        joined.dropna(inplace=True)
        return joined

    joinable_columns = first_df.columns.intersection(next_df.columns)
    if len(joinable_columns) == 0:
        return joined

    unique_columns = first_df.columns.difference(next_df.columns).union(
        next_df.columns.difference(first_df.columns)
    )

    first_df_side_columns = first_df.columns.intersection(next_df.columns)
    next_df_side_columns = next_df.columns.intersection(first_df.columns)

    first_df_side = joined[[f"{c}_x" for c in first_df_side_columns]]
    next_df_side = joined[[f"{c}_y" for c in next_df_side_columns]]

    first_df_side.columns = first_df_side_columns
    next_df_side.columns = next_df_side_columns

    first_df = pd.concat([joined[unique_columns], first_df_side], axis=1)
    next_df = pd.concat([joined[unique_columns], next_df_side], axis=1)

    joined = pd.concat([first_df, next_df])
    joined.reset_index(drop=True, inplace=True)
    joined.drop_duplicates(inplace=True)
    return joined


def _operator_groundings(joined: pd.DataFrame, operator):
    if joined.empty:
        return None

    joined.sort_index(axis=1, inplace=True)
    groundings = list(joined.itertuples(index=False, name=None))
    single_var = len(operator.unique_vars) == 1
    groundings = [
        operator._ground(g[0]) if single_var else operator._ground(g)
        for g in groundings
    ]

    return groundings


def _operand_groundings(joined: pd.DataFrame, operator: Formula, bindings):
    if joined.empty:
        return None

    grounding_objects = [[] for _ in operator.operands]

    for i, operand in enumerate(operator.operands):
        if not operator.operand_map[i]:
            continue

        columns = list(operator.operand_map[i])

        df = joined[columns].copy()

        for grounding in df.itertuples(index=False, name=None):
            adjusted = []
            j = 0
            for binding in bindings[i]:
                if binding:
                    adjusted.append(binding)
                else:
                    adjusted.append(grounding[j])
                    j += 1

            grounding = tuple(adjusted)
            grounding_object = operand._ground(grounding)
            operand._add_groundings(grounding_object)
            grounding_objects[i].append(grounding_object)

    return grounding_objects


@torch.no_grad()
def _hash_join(operands: Formula, groundings: Set) -> Set:
    """get groundings that appear in all children"""
    result = [g for g in groundings if all(g in op.grounding_table for op in operands)]
    return set(result)
