##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import itertools
from typing import List, Dict, Set, Tuple, TypeVar, Union

from ._bindings import get_bindings
from ..constants import Direction, Join

import numpy
import pandas
from itertools import product, chain

import torch

"""
Grounding management module

All functions in this module assume a Formula level scope
"""

_Grounding = TypeVar("_Grounding")
Formula = TypeVar("Formula")


def upward_bounds(
    self: Formula,
    operands: Tuple[Formula],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[None, Tuple[torch.Tensor, None], Tuple[torch.Tensor, Set[_Grounding]]]:
    r"""Returns (input_bounds, groundings) if new groundings are detected."""

    if self.propositional and _is_contradiction(self, operands):
        return None

    result = _operational_bounds(self, Direction.UPWARD, operands, groundings)

    if result is None:
        return None

    if self.propositional and self.is_contradiction(result[0], stacked=True):
        return None

    input_bounds, groundings = result

    if groundings is None:
        return input_bounds, None

    contradictions = self.contradicting_bounds(input_bounds, stacked=True)
    contradicting_groundings = [g for g, c in zip(groundings, contradictions) if c]

    if len(contradicting_groundings) == len(groundings):
        return None

    for g in contradicting_groundings:
        groundings.remove(g)

    input_bounds = input_bounds[~contradictions]

    return input_bounds, groundings


def downward_bounds(
    self: Formula,
    operands: Tuple[Formula],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[
    None,
    Tuple[torch.Tensor, torch.Tensor, None],
    Tuple[torch.Tensor, torch.Tensor, Set[_Grounding]],
]:
    r"""Returns (output_bounds, input_bounds, groundings) if new groundings detected."""
    if self.propositional and _is_contradiction(self, operands):
        return None

    result = _operational_bounds(self, Direction.DOWNWARD, operands, groundings)

    if result is None:
        return None

    if self.propositional and (
        self.is_contradiction(result[0])
        or self.is_contradiction(result[1], stacked=True)
    ):
        return None

    output_bounds, input_bounds, groundings = result

    if groundings is None:
        return output_bounds, input_bounds, None

    input_contradictions = self.contradicting_bounds(input_bounds, stacked=True)
    output_contradictions = self.contradicting_bounds(output_bounds)
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
    self: Formula,
    operand_idx: int,
    operand_grounding: _Grounding,
) -> bool:
    return all(
        True
        if self.bindings[operand_idx][slot] == [None]
        else (
            operand_grounding.partial_grounding[slot]
            in self.bindings[operand_idx][slot]
        )
        for slot in range(len(self.bindings[operand_idx]))
    )


def _operational_bounds(
    self: Formula,
    direction: Direction,
    operands: Tuple[Formula, ...],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[
    None,
    Tuple[torch.Tensor, None],
    Tuple[torch.Tensor, torch.Tensor, None],
    Tuple[torch.Tensor, Set[_Grounding]],
    Tuple[torch.Tensor, torch.Tensor, Set[_Grounding]],
]:
    if self.propositional:
        return _propositional_bounds(self, direction, operands)

    # FOL
    is_homogenous = all([v == self.var_remap[0] for v in self.var_remap])

    if is_homogenous and not self._has_bindings():
        return _fol_bounds(self, direction, operands, groundings)

    # heterogeneous
    grounding_tables = [
        {_eval_grounding(g): g for g in operand.grounding_table}
        if operand.grounding_table
        else {}
        for operand in operands
    ]

    tmp_bindings = [
        tuple([get_bindings(g) for g in op]) if op else ([None],)
        for op in self.bindings
    ]
    tmp_binding_str = [", ".join([f"{v}" for v in op]) for op in self.var_remap if op]

    outer_join_opt = True
    if self.join in [Join.INNER, Join.INNER_EXTENDED, Join.OUTER]:
        if outer_join_opt is True and self.join is Join.OUTER:
            ground_tuples, ground_objects = _hash_join_outer(self, tmp_bindings)
        else:
            ground_tuples, ground_objects = _nested_join(self, tmp_bindings)
    elif self.join is Join.OUTER_PRUNED:
        ground_tuples, ground_objects = _nested_loop_join_outer_pruned(
            grounding_tables, tmp_binding_str, tmp_bindings, operands
        )

    # TODO: This is just a quick fix. There are cases where joins
    # TODO: fail to find groundings that exist.
    if ground_objects is None or all([len(o) == 0 for o in ground_objects]):
        return

    tmp_ground_tuples = (
        [g[0] for g in ground_tuples] if (len(self.unique_vars) == 1) else ground_tuples
    )
    groundings = tuple([self._ground(t) for t in tmp_ground_tuples])

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

    self._add_groundings(*groundings)
    if direction is Direction.UPWARD:
        return input_bounds, groundings
    output_bounds = self.get_data(*groundings)
    if len(output_bounds) == 0:
        return
    return output_bounds, input_bounds, groundings


@torch.no_grad()
def _hash_join_outer(operator: Formula, bindings):
    """
     E.g  Join two Predicates P1(x,y) and P2(x,z,y) with groundings
    (x1,y1) : g11 (ground object)  and (x1,z1,y1) : g12
    (x2,y2) : g21 (ground object)  and (x2,z2,y2) : g22
    (x1,y3) : g31 (ground object)  and (x2,z3,y2) : g32

    Inputs

    arg_str : is the list of strings : ['x,y','x,z,y']
    bindings : list of bindings for each variable
    g_list : is the list [ g_list[0], g_list[1]]
    where g_list[0] is the dictionary g_list[0]
       [(x1,y1)] = g11 ; [(x2,y2)] = g21 ; [(x1,y3)] = g31
    and g_list[1] is the dictionary g_list[1]
       [(x1,z1,y1)] = g12;[(x2,z2,y2)] = g2;[(x2,z3,y3)] = g32

    **Returns**

    list_tuples:
        List of joined ground tuples: [(x1,y1,z1), (x2,y2,z2), ...]
    object: List of list of  ground objects for each operand
       [[g11,g21], [g12,g22], ...]

    """
    arg_str = [", ".join([f"{v}" for v in op]) for op in operator.var_remap if op]
    operands = operator.operands
    g_list = [
        {_eval_grounding(g): g for g in operand.grounding_table}
        for operand in operands
        if operand
    ]
    n_ops = len(g_list)
    var_list = [g.split(", ") for g in arg_str]
    _vars = set()
    var_map = {}
    var_count = 0
    var_remap = []
    is_binding = [False for _ in range(len(bindings))]
    for i_, b_ in enumerate(bindings):
        for bb_ in b_:
            is_binding[i_] = is_binding[i_] or not (len(bb_) == 1 and bb_[0] is None)
    for v in var_list:
        var_remap_t = []
        for v2 in v:
            if v2 not in _vars:
                var_map[v2] = var_count
                var_remap_t.append(var_count)
                var_count = var_count + 1
                _vars.add(v2)
            else:
                var_remap_t.append(var_map[v2])
        var_remap.append(var_remap_t)
    for v_ in var_remap:
        assert len(v_) == len(
            set(v_)
        ), "Repeated variables with Outer join not supported"

    def check_binding(inp_tup, op_index):
        if not is_binding[op_index]:
            return True
        for i, v in enumerate(inp_tup):
            if bindings[op_index][i] == [None]:
                continue
            elif v not in bindings[op_index][i]:
                return False
        return True

    def partial_outer_join(x1, y1, v_left, v_right):
        all_vars_ = v_left.copy()
        for v_ in v_right:
            if v_ not in all_vars_:
                all_vars_.append(v_)

        left_i = [all_vars_.index(v_) for v_ in v_left]
        right_b = [v_right.index(v_) for v_ in all_vars_ if v_ not in v_left]
        right_b2 = [all_vars_.index(v_) for v_ in v_right if v_ not in v_left]

        right_i = [all_vars_.index(v_) for v_ in v_right]
        left_b = [v_left.index(v_) for v_ in all_vars_ if v_ not in v_right]
        left_b2 = [all_vars_.index(v_) for v_ in v_left if v_ not in v_right]

        right_i.sort()

        left_ind = left_i + right_b2
        right_ind = left_b2 + right_i

        a_m = pandas.merge(x1, y1, how="outer", indicator=True)
        right_col = y1.to_numpy()[:, right_b].tolist()
        left_col = x1.to_numpy()[:, left_b].tolist()
        merge_list = []
        merge_len = len(all_vars_)
        temp_list = [0 for i_ in range(merge_len)]
        for index, row in a_m.iterrows():
            row = row.to_list()
            if row[-1] == "left_only":
                left_v = tuple([row[i_] for i_ in left_i])
                if len(right_b) == 0:
                    merge_list.append(left_v)
                else:
                    left_v = [left_v]
                    for p_ in list(product(left_v, right_col)):
                        pp_ = list(chain.from_iterable(p_))
                        for i_ in range(merge_len):
                            temp_list[left_ind[i_]] = pp_[i_]
                        merge_list.append(tuple(temp_list))
            elif row[-1] == "right_only":
                right_v = tuple([row[i_] for i_ in right_i])
                if len(left_b) == 0:
                    merge_list.append(right_v)
                else:
                    right_v = [right_v]
                    for p_ in list(product(left_col, right_v)):
                        pp_ = list(chain.from_iterable(p_))
                        for i_ in range(merge_len):
                            temp_list[right_ind[i_]] = pp_[i_]
                        merge_list.append(tuple(temp_list))
            else:
                left_v = tuple([row[i_] for i_ in left_i])
                if len(right_b) == 0:
                    merge_list.append(left_v)
                else:
                    left_v = [left_v]
                    for p_ in list(product(left_v, right_col)):
                        pp_ = list(chain.from_iterable(p_))
                        for i_ in range(merge_len):
                            temp_list[left_ind[i_]] = pp_[i_]
                        merge_list.append(tuple(temp_list))
                com_val = [row[i_] for i_ in left_b2]
                left_col_n = left_col.copy()
                left_col_n.remove(com_val)

                right_v = tuple([row[i_] for i_ in right_i])
                if len(left_b) == 0:
                    merge_list.append(right_v)
                else:
                    right_v = [right_v]
                    for p_ in list(product(left_col_n, right_v)):
                        pp_ = list(chain.from_iterable(p_))
                        for i_ in range(merge_len):
                            temp_list[right_ind[i_]] = pp_[i_]
                        merge_list.append(tuple(temp_list))
        merge_list = pandas.DataFrame(merge_list, columns=all_vars_).drop_duplicates()
        return merge_list

    curr_merged = g_list[0]
    # curr_map = var_remap[0]
    all_vars = []
    n_z_index = []
    for i, a_ in enumerate(var_remap):
        for aa_ in a_:
            if aa_ not in all_vars:
                all_vars.append(aa_)
        if len(g_list[i]) != 0:
            n_z_index.append(i)

    n_z_vars = []
    for i in n_z_index:
        for a_ in var_remap[i]:
            if a_ not in n_z_vars:
                n_z_vars.append(a_)

    if set(all_vars) != set(n_z_vars):
        return None, None
    reorder_pos = [None] * len(all_vars)
    for i in range(len(n_z_vars)):
        reorder_pos[i] = all_vars.index(n_z_vars[i])

    curr_merged = g_list[n_z_index[0]]
    # curr_map = var_remap[n_z_index[0]]
    # first_op = True
    op_tuples = [[] for i in range(len(operands))]
    op_tuples_new = [[] for i in range(len(operands))]

    frames = []

    for i_, g_ in enumerate(g_list):
        a_ = numpy.full([len(g_), len(var_remap[i_])], "", dtype=object)
        ii_ = 0
        for gg_ in g_:
            if check_binding(gg_, i_):
                a_[ii_] = numpy.asarray(gg_)
                ii_ = ii_ + 1
        frames.append(
            pandas.DataFrame(
                numpy.resize(a_, [ii_, len(var_remap[i_])]), columns=var_remap[i_]
            )
        )
    buff = frames[n_z_index[0]]
    v_x = var_remap[n_z_index[0]].copy()
    for i in n_z_index[1:]:
        v_y = var_remap[i]
        intersection = [v_ for v_ in v_x if v_ in v_y]
        if len(intersection) != 0:
            buff = partial_outer_join(buff, frames[i], v_x, v_y)
        else:
            buff = pandas.merge(buff, frames[i], how="cross")

        for v_ in v_y:
            if v_ not in v_x:
                v_x.append(v_)
    for i_ in range(n_ops):
        x_n = pandas.merge(buff, frames[i_], how="left", indicator=True)
        for index, row in x_n.iterrows():
            row = row.to_list()
            op_tup = tuple([row[r_] for r_ in var_remap[i_]])
            if row[-1] == "left_only":
                op_tuples[i_].append(op_tup)
                op_tuples_new[i_].append(True)
            else:
                op_tuples[i_].append(op_tup)
                op_tuples_new[i_].append(False)
    curr_merged = []
    for index, row in buff.iterrows():
        curr_merged.append(tuple(row.to_list()))

    ground_objects = [[] for i in range(len(operands))]
    for i in range(n_ops):
        if operands[i].arity == 1:
            for i_, op_tup in enumerate(op_tuples[i]):
                op_tup_new = op_tuples_new[i][i_]
                g_obj = operands[i]._ground(op_tup[0])
                if op_tup_new:
                    operands[i]._add_groundings(g_obj)
                ground_objects[i].append(g_obj)
        else:
            for i_, op_tup in enumerate(op_tuples[i]):
                op_tup_new = op_tuples_new[i][i_]
                g_obj = operands[i]._ground(op_tup)
                if op_tup_new:
                    operands[i]._add_groundings(g_obj)
                ground_objects[i].append(g_obj)

    return curr_merged, ground_objects


@torch.no_grad()
def _nested_join(operator: Formula, tmp_bindings):
    r"""
    Performs a database join on the groundings of the operator's operands with variable
    sharing. The database join to perform is determined by the operator.join field.

    Parameters
    ------------
    operator : The Formula to check for shared variables.
    tmp_bindings : list of bindings for each variable

    Example
    ------------
    Join two Predicates P1(x,y) and P2(x, z, y) with groundings
    (x1, y1) : g11 (ground object)  and (x1, z1, y1) : g12
    (x2, y2) : g21 (ground object)  and (x2, z2, y2) : g22
    (x1, y3) : g31 (ground object)  and (x2, z3, y2) : g32

    Returns
    ------------
     list_tuples:
         List of joined ground tuples: [(x1, y1, z1), (x2, y2, z2), ...]
     object: List of list of  ground objects for each operand
        [[g11, g21], [g12, g22], ...]
    """
    grounding_map = _get_grounding_map(operator)
    binding_map = _get_binding_map(tmp_bindings)
    _adjust_grounding_map_to_bindings(grounding_map, binding_map)

    shared_vars = _get_shared_vars(operator)
    groundings_product = _get_groundings_product(operator)

    joins = []
    grounding_objects = [[] for _ in operator.operands]

    # For each combination of operand groundings,
    for groundings in groundings_product:
        # Used to continue or stop joining process.
        joinable = True
        # Used to keep track of where to add groundings when working with outer joins.
        expansion_point = 0
        bindings = []
        # A table to keep track of the joins.
        candidate_joins = [[None] * len(operator.unique_vars)]

        for operand_index, operand_map in enumerate(grounding_map):
            for operand_var_index in operand_map:
                if not joinable:
                    continue

                operator_var_index = operand_map[operand_var_index]
                grounding = groundings[operand_index]

                # If the operand does not have a value for this variable, we do not
                # consider it for joining.
                if grounding is None:
                    continue

                if isinstance(grounding, tuple):
                    grounding = grounding[operand_var_index]

                satisfied = _are_bindings_satisfied(
                    tmp_bindings, grounding, operand_index, operand_var_index
                )

                bindings.append(satisfied)

                if not satisfied:
                    joinable = False
                    continue

                # This is where the join happens. We assign any new grounding to its
                # operator variable slot. TODO: This code should be revisited once the
                # pruned outer join function has been rewritten,
                if candidate_joins[0][operator_var_index] is None:
                    for candidate_join in candidate_joins:
                        candidate_join[operator_var_index] = grounding
                else:
                    # Else, there is already a grounding assigned to its operator
                    # variable slot. This is where the logic of inner joins and outer
                    # joins diverge. If it is an inner join, we check if the groundings
                    # are the same else we do not join.
                    if operator.join in [Join.INNER, Join.INNER_EXTENDED]:
                        if candidate_joins[0][operator_var_index] != grounding:
                            joinable = False
                    else:
                        # Here we check if we have space to assign the grounding
                        if len(candidate_joins) > operand_index:
                            for candidate_join in candidate_joins[expansion_point:]:
                                candidate_join[operator_var_index] = grounding
                        else:
                            # Else, we need to expand the table.
                            expansion_point = len(candidate_joins)
                            for i in range(len(candidate_joins)):
                                candidate_join_copy = candidate_joins[i].copy()
                                candidate_join_copy[operator_var_index] = grounding
                                candidate_joins.append(candidate_join_copy)

        if not joinable:
            continue

        for join in candidate_joins:
            # Check that there is at least one grounding for each operator variable
            if None in join:
                continue

            join = tuple(join)

            if join in joins:
                continue

            for operand_index, _ in enumerate(grounding_map):
                _share_groundings(
                    shared_vars, operator, groundings, operand_index, binding_map
                )

            new_grounding_objects = _construct_groundings(
                join, grounding_map, binding_map, operator.operands, grounding_objects
            )

            _propagate_groundings(operator, new_grounding_objects, join)

            joins.append(join)

    if not joins or not grounding_objects:
        return None, None

    return joins, grounding_objects


@torch.no_grad()
def _nested_loop_join_outer_pruned(g_list, arg_str, bindings, operands):
    """
    E.g  Join two Predicates P1(x,y) and P2(x,z,y) with groundings
    (x1, y1) : g11 (ground object)  and (x1, z1, y1) : g12
    (x2, y2) : g21 (ground object)  and (x2, z2, y2) : g22
    (x1, y3) : g31 (ground object)  and (x2, z3,y 2) : g32

    Inputs

    arg_str : is the list of strings : ["x, y","x, z, y"]
    bindings : list of bindings for each variable
    g_list : is the list [ g_list[0], g_list[1]]
    where g_list[0] is the dictionary g_list[0]
       [(x1, y1)] = g11 ; [(x2, y2)] = g21 ; [(x1, y3)] = g31
    and g_list[1] is the dictionary g_list[1]
       [(x1, z1, y1)] = g12;[(x2, z2, y2)] = g2;[(x2, z3, y3)] = g32

    Returns
    -------
    list_tuples:
        List of joined ground tuples: [(x1, y1, z1), (x2, y2, z2), ...]
    object: List of list of  ground objects for each operand
       [[g11, g21], [g12, g22], ...]

    """
    n_ops = len(g_list)
    var_remap = _get_var_remap(arg_str)
    is_binding = _get_is_binding(bindings)

    all_vars, n_z_index, n_z_groundings, n_z_vars = _get_non_zero_groundings(
        var_remap, g_list
    )

    if set(all_vars) != set(n_z_vars):
        return None, None

    reorder_pos = _reorder(all_vars, n_z_vars)

    curr_merged = g_list[n_z_index[0]]
    curr_map = var_remap[n_z_index[0]]
    first_op = True
    for i in n_z_index[1:]:
        colllected_g_list_obj = {}
        curr_to_merge = g_list[i]
        match_pos1, match_pos2 = _match(curr_map, var_remap[i])
        umatch_pos2 = _unmatch(curr_map, var_remap[i])
        joined_vars, scatter_pos2 = _join_and_scatter(curr_map, var_remap[i])

        for a1 in curr_merged:
            if first_op and not _check_binding(is_binding, bindings, a1, n_z_index[0]):
                continue
            for a2 in curr_to_merge:
                if not _check_binding(is_binding, bindings, a2, i):
                    continue
                m1 = [a1[i_] for i_ in match_pos1]
                m2 = [a2[i_] for i_ in match_pos2]
                if m1 == m2:
                    j1 = tuple(list(a1) + [a2[k] for k in umatch_pos2])
                    if first_op:
                        colllected_g_list_obj[j1] = [curr_merged[a1]] + [
                            curr_to_merge[a2]
                        ]
                    else:
                        colllected_g_list_obj[j1] = curr_merged[a1] + [
                            curr_to_merge[a2]
                        ]
                else:
                    j1, j2 = _get_j1_and_j2(
                        a1, a2, umatch_pos2, joined_vars, scatter_pos2
                    )
                    colllected_g_list_obj[j1] = [curr_merged[a1]] + [curr_to_merge[a2]]
                    colllected_g_list_obj[j2] = [curr_merged[a1]] + [curr_to_merge[a2]]
                    if first_op:
                        colllected_g_list_obj[j1] = [curr_merged[a1]] + [
                            curr_to_merge[a2]
                        ]
                        colllected_g_list_obj[j2] = [curr_merged[a1]] + [
                            curr_to_merge[a2]
                        ]
                    else:
                        colllected_g_list_obj[j1] = curr_merged[a1] + [
                            curr_to_merge[a2]
                        ]
                        colllected_g_list_obj[j2] = curr_merged[a1] + [
                            curr_to_merge[a2]
                        ]
        curr_map = joined_vars
        curr_merged = colllected_g_list_obj
        first_op = False

    if n_z_groundings:
        g_obj2 = {}
        for g in curr_merged:
            if first_op and not _check_binding(is_binding, bindings, g, n_z_index[0]):
                continue

            g_ = [None] * len(g)
            for i, a_ in enumerate(g):
                g_[reorder_pos[i]] = a_
            g_ = tuple(g_)
            g_obj = [curr_merged[g]]
            g_new = _get_new_grounding_objects(
                n_ops, n_z_index, all_vars, var_remap, g_, operands, g_obj
            )
            g_obj2[g_] = g_new
        curr_merged = g_obj2

    return _get_grounding_objects_from_merged(n_ops, curr_merged)


def _check_binding(is_binding, bindings, inp_tup, op_index):
    if not is_binding[op_index]:
        return True

    for i, v in enumerate(inp_tup):
        if bindings[op_index][i] != [None] and v not in bindings[op_index][i]:
            return False

    return True


def _eval_grounding(grounding):
    if grounding.name[0] != "(":
        return eval("('" + grounding.name + "',)")

    return eval(grounding.name)


def _find_rev_joined_pos(full_map, op_map):
    rev_index = []
    rev_pos = []
    for i_, o_ in enumerate(full_map):
        if o_ in op_map:
            rev_index.append(i_)
            rev_pos.append(op_map.index(o_))
    return rev_index, rev_pos


def _fol_bounds(
    self: Formula,
    direction: Direction,
    operands: Tuple[Formula, ...],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
):
    if groundings is None:
        grounding_tables = [op.grounding_table for op in operands]
        groundings = set(itertools.chain.from_iterable(grounding_tables))

        if direction is Direction.DOWNWARD:
            groundings.update(self.grounding_table.keys())
    else:
        groundings = set(map(self._ground, groundings))

        # Propagate groundings to all children.
    for op in operands:
        op._add_groundings(*groundings)

    groundings = _hash_join(operands, groundings)
    input_bounds = torch.stack([op.get_data(*groundings) for op in operands], dim=-1)
    self._add_groundings(*groundings)

    if direction is Direction.UPWARD:
        return input_bounds, groundings

    output_bounds = self.get_data(*groundings)

    if len(output_bounds) == 0:
        return

    return output_bounds, input_bounds, groundings


def _get_grounding_objects_from_merged(n_ops, merged):
    g_obj = [[gg[i] for gg in merged.values()] for i in range(n_ops)]
    return list(merged.keys()), g_obj


def _get_is_binding(bindings):
    is_binding = [False for _ in range(len(bindings))]
    for i_, b_ in enumerate(bindings):
        for bb_ in b_:
            is_binding[i_] = is_binding[i_] or not (len(bb_) == 1 and bb_[0] is None)

    return is_binding


def _get_j1_and_j2(a1, a2, umatch_pos2, joined_vars, scatter_pos2):
    j1 = tuple(list(a1) + [a2[k] for k in umatch_pos2])
    j2 = [None] * len(joined_vars)
    j2[0 : len(a1)] = a1
    for i, j in enumerate(scatter_pos2):
        j2[j] = a2[i]
    j2 = tuple(j2)
    return j1, j2


def _get_new_grounding_objects(
    n_ops, n_z_index, all_vars, var_remap, g_, operands, g_obj
):
    g_new = []
    g_obj_indx = 0
    for i in range(n_ops):
        if i not in n_z_index:
            rev_index, rev_pos = _find_rev_joined_pos(all_vars, var_remap[i])
            m1 = [None] * len(var_remap[i])
            for i_, r_ in enumerate(rev_index):
                m1[rev_pos[i_]] = g_[r_]
            m1 = tuple(m1)
            if operands[i].arity == 1:
                g_obj_n = operands[i]._ground(m1[0])
                operands[i]._add_groundings(g_obj_n)
            else:
                g_obj_n = operands[i]._ground(m1)
                operands[i]._add_groundings(g_obj_n)
            g_new.append(g_obj_n)
        else:
            g_new.append(g_obj[g_obj_indx])
            g_obj_indx = g_obj_indx + 1
    return g_new


def _get_non_zero_groundings(var_remap, g_list):
    all_vars = []
    n_z_index = []
    n_z_groundings = False
    for i, a in enumerate(var_remap):
        for aa in a:
            if aa not in all_vars:
                all_vars.append(aa)
        if len(g_list[i]) != 0:
            n_z_index.append(i)
        else:
            n_z_groundings = True

    n_z_vars = []
    for i in n_z_index:
        for a in var_remap[i]:
            if a not in n_z_vars:
                n_z_vars.append(a)

    return all_vars, n_z_index, n_z_groundings, n_z_vars


def _get_var_remap(arg_str: list[str]):
    var_list = [g.split(", ") for g in arg_str]
    _vars = set()
    var_map = {}
    var_count = 0
    var_remap = []

    for v in var_list:
        var_remap_t = []
        for v2 in v:
            if v2 not in _vars:
                var_map[v2] = var_count
                var_remap_t.append(var_count)
                var_count = var_count + 1
                _vars.add(v2)
            else:
                var_remap_t.append(var_map[v2])
        var_remap.append(var_remap_t)

    for v in var_remap:
        assert len(v) == len(set(v)), "Repeated variables with Outer join not supported"

    return var_remap


@torch.no_grad()
def _hash_join(operands: Formula, groundings: Set) -> Set:
    """get groundings that appear in all children"""
    result = [g for g in groundings if all(g in op.grounding_table for op in operands)]
    return set(result)


def _is_contradiction(formula: Formula, operands: Tuple[Formula, ...]) -> bool:
    operand_contradiction = any([op.is_contradiction() for op in operands])
    formula_contradiction = formula.is_contradiction()

    return operand_contradiction or formula_contradiction


def _join_and_scatter(var_map1, var_map2):
    joined = [v for v in set(var_map1 + var_map2)]
    scatter = [joined.index(v) for v in var_map2]
    return joined, scatter


def _match(var_map1, var_map2):
    match_pos1 = []
    match_pos2 = []
    for v in var_map1:
        if v in var_map2:
            match_pos1.append(var_map1.index(v))
            match_pos2.append(var_map2.index(v))

    return match_pos1, match_pos2


def _propositional_bounds(
    self: Formula,
    direction: Direction,
    operands: Tuple[Formula, ...],
):
    input_bounds = torch.stack([op.get_data() for op in operands], dim=-1)
    if direction is Direction.UPWARD:
        return input_bounds, None

    return self.get_data(), input_bounds, None


def _reorder(all_vars, n_z_vars):
    return [all_vars.index(n_z_vars[i]) for i in range(len(n_z_vars))]


def _unmatch(var_map1, var_map2):
    return [var_map2.index(v) for v in var_map2 if v not in var_map1]


def _adjust_grounding_map_to_bindings(
    grounding_map: List[Dict[int, int]], binding_map: List[Dict[int, int]]
) -> List[Dict[int, int]]:
    r"""
    Adjusts the keys of each dictionary in grounding_map to account for bindings.
    For example, given grounding_map is [{0: 0}, {0: 1}] and binding_map is
    [{0: 'x', 1: None}, {0: None}], this function will adjust grounding_map
    to [{1: 0}, {0: 1}].
    """
    for operand_index, operand_grounding_map in enumerate(grounding_map):
        operand_binding_map = binding_map[operand_index]

        # Is there a difference in the binding_map and the operand_binding_map? With
        # the new binding api, the operand_binding_map will be longer than the
        # operand_binding_map.
        if len(operand_binding_map) > len(operand_grounding_map):
            # This gets all the indices of the variables that do not have bindings.
            binding_mask = [
                i
                for i, key in enumerate(operand_binding_map)
                if operand_binding_map[key] is None
            ]
            for new_key, old_key in zip(binding_mask, operand_grounding_map):
                if new_key != old_key:
                    operand_grounding_map[new_key] = operand_grounding_map[old_key]
                    del operand_grounding_map[old_key]

    return grounding_map


def _get_grounding_map(operator: Formula) -> List[Dict[int, int]]:
    r"""
    Returns a list of maps (dictionaries) where each dictionary represents a map between
    the position of the operand's variables and the position of the operator's
    variables.

    # [{operand_var, operator_var_index}, ...]

    Parameters
    ------------
    operator : The Formula to produce a grounding map for.
    """

    grounding_map = []

    for operand_index, operand in enumerate(operator.operands):
        operand_map = {}
        for operator_var_index in range(len(operator.unique_vars)):
            operand_groundings = operator.operand_map[operand_index]
            if operand_groundings:
                for operand_var_slot, operand_var_index in enumerate(
                    operand_groundings
                ):
                    if operator_var_index == operand_var_index:
                        operand_map[operand_var_slot] = operator_var_index

        grounding_map.append(operand_map)

    return grounding_map


def _get_binding_map(bindings) -> List[Dict[int, int]]:
    r"""
    Returns a list of maps (dictionaries) where each dictionary represents a map
    between the position of the operand's variables and the variables bindings if
    the operand has any bindings.
    """
    res = []
    for b in bindings:
        res.append({i: x[0] for i, x in enumerate(b)})

    return res


def _get_shared_vars(operator: Formula) -> list:
    r"""
    Returns a list of shared variables between the operator's operands. Each element
    in the returned list is a dictionary. The index of each element in the returned
    list corresponds to each operand in the operator (in order of appearance). Thus,
    the keys of each dictionary are the other operands and the values are the shared
    variables.

    Parameters
    ------------
    operator : The Formula to check for shared variables.
    """
    shared_variables = []
    for operand_index_1, operand_1 in enumerate(operator.operands):
        store = {}
        for operand_index_2, operand_2 in enumerate(operator.operands):
            if operand_1 != operand_2:
                operand_1_vars = operator.operand_map[operand_index_1]
                operand_2_vars = operator.operand_map[operand_index_2]
                shared = [None]
                if operand_1_vars and operand_2_vars:
                    shared = [
                        operand_2_vars.index(var) if var in operand_2_vars else None
                        for var in operand_1_vars
                    ]

                # Only store operand for which all variables are shared.
                if None not in shared:
                    store[operand_index_2] = shared

        shared_variables.append(store)

    return shared_variables


def _get_groundings_product(operator: Formula):
    r"""
    Returns the product of operator's operand's groundings. Each element in the returned
    list is a tuple of groundings.

    Parameters
    ------------
    operator : The Formula to produce candidate joins for.
    """
    operand_groundings = [
        operand.groundings if len(operand.groundings) else [None]
        for operand in operator.operands
    ]

    return itertools.product(*operand_groundings)


def _are_bindings_satisfied(
    tmp_bindings,
    grounding: Union[str, int],
    operand_index: int,
    operand_variable_slot: int,
) -> bool:
    r"""
    Checks if the bindings of the operand at the specified variable slot is satisfied.

    Parameters
    ------------
    tmp_bindings : list of bindings for each variable.
    """
    # Check if bindings are satisfied.
    if any([b is None for b in tmp_bindings[operand_index][operand_variable_slot]]):
        # No bindings provided, so it's satisfied.
        return True
    else:
        if grounding in tmp_bindings[operand_index][operand_variable_slot]:
            return True
        else:
            return False


def _share_groundings(
    shared_vars,
    operator: Formula,
    candidate_join,
    operand_index_1: int,
    binding_map: List[Dict[int, int]],
):
    r"""
    Attempts to share groundings of the given operand amongst the other operands of the
    operator.

    Parameters
    ------------
    shared_vars : list of shared variables.
    operator : The Formula to share groundings amongst its operands.
    candidate_join : The current join attempt.
    operand_index_1 : The operand index.
    binding_map: A list of dictionaries that represent a mapping between the position
        of operand groundings and bindings at those positions.
    """
    current_operand = operator.operands[operand_index_1]

    # For every other operand and the shared variables between it and operand_index_1
    for operand_index_2, _vars in shared_vars[operand_index_1].items():
        grounding = candidate_join[operand_index_2]
        is_tuple = isinstance(grounding, tuple)

        if grounding:
            new_grounding = [
                grounding[_vars[i]] if is_tuple else grounding
                for i in range(len(_vars))
            ]

            operand_binding_map = binding_map[operand_index_1]

            if len(operand_binding_map) > len(new_grounding):
                for operand_var_index in operand_binding_map:
                    binding = operand_binding_map[operand_var_index]

                    if binding:
                        new_grounding.append(binding)

            if len(new_grounding) < 2:
                new_grounding = current_operand._ground(new_grounding[0])
            else:
                new_grounding = current_operand._ground(tuple(new_grounding))

            current_operand._add_groundings(new_grounding)


def _construct_groundings(
    join: Tuple[str],
    grounding_map: List[Dict[int, int]],
    binding_map: List[Dict[int, int]],
    operands,
    grounding_objects,
) -> List:
    r"""
    Constructs grounding objects for each operand given a join representation and a
    grounding map. Note to minimize iterations, the new grounding objects are added
    to grounding_objects as a side effect. Returns the newly constructed grounding
    objects.

    Parameters
    ------------
    join : Tuple of groundings
    grounding_map: A list of dictionaries that represent a mapping between the position
        of operand groundings and the position of the groundings in the join.
    binding_map: A list of dictionaries that represent a mapping between the position
        of operand groundings and bindings at those positions.
    operands : The operands to build groundings for.
    grounding_objects : A list of grounding objects to append the new grounding objects
        to.
    """
    new_grounding_objects = []

    for operand_index, operand_map in enumerate(grounding_map):
        operand_binding_map = binding_map[operand_index]
        grounding = [None] * max(len(operand_map), len(operand_binding_map))

        operand = operands[operand_index]
        for operand_var_index in operand_map:
            operator_var_index = operand_map[operand_var_index]
            grounding[operand_var_index] = join[operator_var_index]

        if len(operand_binding_map) > len(operand_map):
            for operand_var_index in operand_binding_map:
                binding = operand_binding_map[operand_var_index]

                if binding:
                    grounding[operand_var_index] = binding

        if len(grounding) == 1:
            grounding = grounding[0]
        else:
            grounding = tuple(grounding)

        if grounding:
            grounding_object = operand._ground(grounding)
            operand._add_groundings(grounding_object)
            grounding_objects[operand_index].append(grounding_object)
            new_grounding_objects.append(grounding_object)

    return new_grounding_objects


def _propagate_groundings(operator: Formula, grounding_objects, joins_on_operator_var):
    r"""
    If a parent grounding was successfully created during a join while some operands
    didn't have groundings, propagate appropriate groundings to those operands.

    Parameters
    ------------
    operator : The Formula to propagate groundings to its operands.
    grounding_objects : The grounding objects.
    joins_on_operator_var : The current join attempt on a specific variable. A list of
        groundings.
    """
    for i, grounding in enumerate(grounding_objects):
        if grounding is None:
            operand_groundings = operator.operand_map[i]
            new_grounding = [joins_on_operator_var[op] for op in operand_groundings]

            if len(new_grounding) < 2:
                new_grounding = new_grounding[0]
            else:
                new_grounding = tuple(new_grounding)

            grounding_object = operator.operands[i]._ground(new_grounding)
            operator.operands[i]._add_groundings(grounding_object)
            grounding_objects[i] = grounding_object
