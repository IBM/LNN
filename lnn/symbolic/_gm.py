##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import copy
from itertools import chain
from typing import Tuple, Union, List, TypeVar, Set

import torch

from ..constants import Direction, Join
from .._utils import negate_bounds


"""
Grounding management module

All functions in this module assume a _Formula level scope
"""


_Grounding = TypeVar("_Grounding")
_Formula = TypeVar("_Formula")


def upward_bounds(
    self: _Formula,
    operands: Tuple[_Formula, ...],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[None, Tuple[torch.Tensor, None], Tuple[torch.Tensor, Set[_Grounding]]]:
    """returns (input_bounds, groundings)"""
    result = _operational_bounds(self, Direction.UPWARD, operands, groundings)
    return result


def downward_bounds(
    self: _Formula,
    operands: Tuple[_Formula, ...],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[
    None,
    Tuple[torch.Tensor, torch.Tensor, None],
    Tuple[torch.Tensor, torch.Tensor, Set[_Grounding]],
]:
    """returns (output_bounds, input_bounds, groundings)"""
    result = _operational_bounds(self, Direction.DOWNWARD, operands, groundings)
    return result


def disjoint(pred_vars):
    return not any([all([v in _vars for _vars in pred_vars]) for v in pred_vars[0]])


_Variable = TypeVar("_Variable")


def unique_variables(*variables: Tuple[_Variable, ...]) -> Tuple:
    """combines all predicate variables into a unique tuple
    the tuple is sorted by the order of appearance of variables in the operands
    """
    result = list()
    for op_vars in variables:
        for v in op_vars:
            if v not in result:
                result.append(v)
    return tuple(result)


def _operational_bounds(
    self: _Formula,
    direction: Direction,
    operands: Tuple[_Formula, ...],
    groundings: Set[Union[str, Tuple[str, ...]]] = None,
) -> Union[
    None,
    Tuple[torch.Tensor, None],
    Tuple[torch.Tensor, torch.Tensor, None],
    Tuple[torch.Tensor, Set[_Grounding]],
    Tuple[torch.Tensor, torch.Tensor, Set[_Grounding]],
]:

    # propositional / hash join for homogeneous operand variables
    if self.propositional or (  # propositional
        all([v == self.var_remap[0] for v in self.var_remap])  # homogenous variables
        and not self._has_bindings()
    ):  # bindings
        if self.propositional:  # propositional bounds
            if True in (
                [op.is_contradiction() for op in operands] + [self.is_contradiction()]
            ):
                return
            input_bounds = _masked_negate(
                self, torch.stack([op.get_facts() for op in operands], dim=-1)
            )
            if direction is Direction.UPWARD:
                return input_bounds, None
            return self.get_facts(), input_bounds, None

        else:  # FOL bounds
            if groundings is None:
                groundings = set(
                    chain.from_iterable([op.grounding_table for op in operands])
                )
                if direction is Direction.DOWNWARD:
                    groundings.update(self.grounding_table.keys())
            else:
                groundings = set(map(self._ground, groundings))

            for g in groundings:
                for op in operands:
                    op._add_groundings(g)

            groundings = _hash_join(operands, groundings)
            input_bounds = _masked_negate(
                self,
                torch.stack([op.get_facts(*groundings) for op in operands], dim=-1),
            )
            self._add_groundings(*groundings)
            if direction is Direction.UPWARD:
                return input_bounds, groundings
            output_bounds = self.get_facts(*groundings)
            if len(output_bounds) == 0:
                return
            return output_bounds, input_bounds, groundings

    # nested loop join for bindings / heterogenous operand variables
    else:
        if self.propositional:
            raise TypeError("proposition should not reach here")
        grounding_tables = []
        for op in operands:
            g_t = dict()
            for g in op.grounding_table:
                if g.name[0] != "(":
                    g_t[eval("('" + g.name + "',)")] = g
                else:
                    g_t[eval(g.name)] = g  # op.grounding_table[g]
            grounding_tables.append(g_t)

        tmp_bindings = [
            tuple(
                [str(b) if b is not None else b for b in g]
                if isinstance(g, List)
                else str(g)
                if g is not None
                else g
                for g in op
            )
            for op in self.bindings
        ]
        tmp_binding_str = [", ".join([f"{v}" for v in op]) for op in self.var_remap]
        if self.join_method is Join.INNER:
            ground_tuples, ground_objects = _nested_loop_join_inner(
                grounding_tables, tmp_binding_str, tmp_bindings, operands
            )
        elif self.join_method is Join.OUTER:
            ground_tuples, ground_objects = _nested_loop_join_outer(
                grounding_tables, tmp_binding_str, tmp_bindings, operands
            )
        elif self.join_method is Join.OUTER_PRUNED:
            ground_tuples, ground_objects = _nested_loop_join_outer_pruned(
                grounding_tables, tmp_binding_str, tmp_bindings, operands
            )

        if ground_objects is None or all([len(o) == 0 for o in ground_objects]):
            return

        tmp_ground_tuples = (
            [g[0] for g in ground_tuples]
            if (len(self.unique_vars) == 1)
            else ground_tuples
        )
        groundings = tuple()
        for t in tmp_ground_tuples:
            groundings += (self._ground(t),)
        input_bounds = _masked_negate(
            self,
            torch.stack(
                [op.get_facts(*ground_objects[i]) for i, op in enumerate(operands)],
                dim=-1,
            ),
        )
        self._add_groundings(*groundings)
        if direction is Direction.UPWARD:
            return input_bounds, groundings
        output_bounds = self.get_facts(*groundings)
        if len(output_bounds) == 0:
            return
        return output_bounds, input_bounds, groundings


@torch.no_grad()
def _nested_loop_join_outer(g_list, arg_str, bindings, operands):
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
    n_ops = len(g_list)
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

    def find_rev_joined_pos(full_map, op_map):
        rev_index = []
        rev_pos = []
        for i_, o_ in enumerate(full_map):
            if o_ in op_map:
                rev_index.append(i_)
                rev_pos.append(op_map.index(o_))
        return rev_index, rev_pos

    def find_joined_vars(var_map1, var_map2):
        match_pos1 = []
        match_pos2 = []
        for v in var_map1:
            if v in var_map2:
                match_pos1.append(var_map1.index(v))
                match_pos2.append(var_map2.index(v))
        umatch_pos2 = [var_map2.index(v) for v in var_map2 if v not in var_map1]

        joined_vars = [v for v in var_map1]
        for v in var_map2:
            if v not in joined_vars:
                joined_vars.append(v)

        scatter_pos2 = []
        for v in var_map2:
            scatter_pos2.append(joined_vars.index(v))

        return match_pos1, match_pos2, umatch_pos2, joined_vars, scatter_pos2

    curr_merged = g_list[0]
    curr_map = var_remap[0]
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
    curr_map = var_remap[n_z_index[0]]
    for i in n_z_index[1:]:
        colllected_g_list_full = set()
        curr_to_merge = g_list[i]
        (
            match_pos1,
            match_pos2,
            umatch_pos2,
            joined_vars,
            scatter_pos2,
        ) = find_joined_vars(curr_map, var_remap[i])
        for a1 in curr_merged:
            for a2 in curr_to_merge:
                m1 = [a1[i] for i in match_pos1]
                m2 = [a2[i] for i in match_pos2]
                if m1 == m2:
                    j1 = tuple(list(a1) + [a2[k] for k in umatch_pos2])
                    colllected_g_list_full.add(j1)
                else:
                    j1 = tuple(list(a1) + [a2[k] for k in umatch_pos2])
                    j2 = [None] * len(joined_vars)
                    j2[0 : len(a1)] = a1
                    for i, j in enumerate(scatter_pos2):
                        j2[j] = a2[i]
                    j2 = tuple(j2)
                    colllected_g_list_full.add(j1)
                    colllected_g_list_full.add(j2)
        curr_map = joined_vars
        curr_merged = list(colllected_g_list_full)

    ground_objects = [[] for i in range(len(operands))]
    for i in range(n_ops):
        rev_index, rev_pos = find_rev_joined_pos(curr_map, var_remap[i])
        curr_op = copy.copy(g_list[i])
        for curr_tup in curr_merged:
            found = False
            m1 = [None] * len(var_remap[i])
            for i_, r_ in enumerate(rev_index):
                m1[rev_pos[i_]] = curr_tup[r_]
            m1 = tuple(m1)
            for op_tup in curr_op:
                if m1 == op_tup:
                    if operands[i].arity == 1:
                        g_obj = operands[i]._ground(op_tup[0])
                    else:
                        g_obj = operands[i]._ground(op_tup)
                    ground_objects[i].append(g_obj)
                    found = True
            if not found:
                if operands[i].arity == 1:
                    g_obj = operands[i]._ground(m1[0])
                    operands[i]._add_groundings(g_obj)
                else:
                    g_obj = operands[i]._ground(m1)
                    operands[i]._add_groundings(g_obj)
                ground_objects[i].append(g_obj)
                curr_op[m1] = g_obj

    return curr_merged, ground_objects


@torch.no_grad()
def _nested_loop_join_outer_pruned(g_list, arg_str, bindings, operands):
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
    n_ops = len(g_list)
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

    def find_rev_joined_pos(full_map, op_map):
        rev_index = []
        rev_pos = []
        for i_, o_ in enumerate(full_map):
            if o_ in op_map:
                rev_index.append(i_)
                rev_pos.append(op_map.index(o_))
        return rev_index, rev_pos

    def find_joined_vars(var_map1, var_map2):
        match_pos1 = []
        match_pos2 = []
        for v in var_map1:
            if v in var_map2:
                match_pos1.append(var_map1.index(v))
                match_pos2.append(var_map2.index(v))
        umatch_pos2 = [var_map2.index(v) for v in var_map2 if v not in var_map1]

        joined_vars = [v for v in var_map1]
        for v in var_map2:
            if v not in joined_vars:
                joined_vars.append(v)

        scatter_pos2 = []
        for v in var_map2:
            scatter_pos2.append(joined_vars.index(v))

        return match_pos1, match_pos2, umatch_pos2, joined_vars, scatter_pos2

    curr_merged = g_list[0]
    curr_map = var_remap[0]
    all_vars = []
    n_z_index = []
    n_z_groundings = False
    for i, a_ in enumerate(var_remap):
        for aa_ in a_:
            if aa_ not in all_vars:
                all_vars.append(aa_)
        if len(g_list[i]) != 0:
            n_z_index.append(i)
        else:
            n_z_groundings = True

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
    curr_map = var_remap[n_z_index[0]]
    first_op = True
    for i in n_z_index[1:]:
        colllected_g_list_obj = {}
        curr_to_merge = g_list[i]
        (
            match_pos1,
            match_pos2,
            umatch_pos2,
            joined_vars,
            scatter_pos2,
        ) = find_joined_vars(curr_map, var_remap[i])
        for a1 in curr_merged:
            for a2 in curr_to_merge:
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
                    j1 = tuple(list(a1) + [a2[k] for k in umatch_pos2])
                    j2 = [None] * len(joined_vars)
                    j2[0 : len(a1)] = a1
                    for i_, j_ in enumerate(scatter_pos2):
                        j2[j_] = a2[i_]
                    j2 = tuple(j2)
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
            g_ = [None] * len(g)
            for i, a_ in enumerate(g):
                g_[reorder_pos[i]] = a_
            g_ = tuple(g_)
            g_obj = [curr_merged[g]]
            g_new = []
            g_obj_indx = 0
            for i in range(n_ops):
                if i not in n_z_index:
                    rev_index, rev_pos = find_rev_joined_pos(all_vars, var_remap[i])
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
            g_obj2[g_] = g_new
        curr_merged = g_obj2

    g_obj = [None] * n_ops
    for i in range(n_ops):
        g_obj[i] = []
    for gg in curr_merged.values():
        for i in range(n_ops):
            g_obj[i].append(gg[i])

    return list(curr_merged.keys()), g_obj


@torch.no_grad()
def _nested_loop_join_inner(g_list, arg_str, bindings, operands):

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
    n_ops = len(g_list)
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

    def find_rev_joined_pos(full_map, op_map):
        rev_index = []
        rev_pos = []
        for i_, o_ in enumerate(full_map):
            if o_ in op_map:
                rev_index.append(i_)
                rev_pos.append(op_map.index(o_))
        return rev_index, rev_pos

    def find_joined_vars(var_map1, var_map2):
        match_pos1 = []
        match_pos2 = []
        for v in var_map1:
            if v in var_map2:
                match_pos1.append(var_map1.index(v))
                match_pos2.append(var_map2.index(v))
        umatch_pos2 = [var_map2.index(v) for v in var_map2 if v not in var_map1]

        joined_vars = [v for v in var_map1]
        for v in var_map2:
            if v not in joined_vars:
                joined_vars.append(v)

        scatter_pos2 = []
        for v in var_map2:
            scatter_pos2.append(joined_vars.index(v))

        return match_pos1, match_pos2, umatch_pos2, joined_vars, scatter_pos2

    all_vars = []
    n_z_index = []
    n_z_groundings = False
    for i, a_ in enumerate(var_remap):
        for aa_ in a_:
            if aa_ not in all_vars:
                all_vars.append(aa_)
        if len(g_list[i]) != 0:
            n_z_index.append(i)
        else:
            n_z_groundings = True

    n_z_vars = []
    for i in n_z_index:
        for a_ in var_remap[i]:
            if a_ not in n_z_vars:
                n_z_vars.append(a_)

    if set(all_vars) != set(n_z_vars):
        curr_merged = g_list[0]
        g_obj = [None] * n_ops
        for i in range(n_ops):
            g_obj[i] = []
        return list(curr_merged.keys()), g_obj

    reorder_pos = [None] * len(all_vars)
    for i in range(len(n_z_vars)):
        reorder_pos[i] = all_vars.index(n_z_vars[i])

    curr_merged = g_list[n_z_index[0]]
    curr_map = var_remap[n_z_index[0]]
    first_op = True
    for i in n_z_index[1:]:
        colllected_g_list_obj = {}
        curr_to_merge = g_list[i]
        match_pos1, match_pos2, umatch_pos2, joined_vars, _ = find_joined_vars(
            curr_map, var_remap[i]
        )
        for a1 in curr_merged:
            for a2 in curr_to_merge:
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
        curr_map = joined_vars
        curr_merged = colllected_g_list_obj
        first_op = False

    if n_z_groundings:
        g_obj2 = {}
        for g in curr_merged:
            g_ = [None] * len(g)
            for i, a_ in enumerate(g):
                g_[reorder_pos[i]] = a_
            g_ = tuple(g_)
            g_obj = [curr_merged[g]]
            g_new = []
            g_obj_indx = 0
            for i in range(n_ops):
                if i not in n_z_index:
                    rev_index, rev_pos = find_rev_joined_pos(all_vars, var_remap[i])
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
            g_obj2[g_] = g_new
        curr_merged = g_obj2

    g_obj = [None] * n_ops
    for i in range(n_ops):
        g_obj[i] = []
    for gg in curr_merged.values():
        for i in range(n_ops):
            g_obj[i].append(gg[i])

    return list(curr_merged.keys()), g_obj


def is_grounding_in_bindings(
    self: _Formula,
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


@torch.no_grad()
def _hash_join(operands: _Formula, groundings: Set) -> Set:
    """get groundings that appear in all children"""

    # limit grounding_table join to given groundings
    grounding_tables = list(
        {g: op.grounding_table[g] for g in groundings if g in op.grounding_table}
        for op in operands
    )
    result = list()
    for g in groundings:
        if all(g in grounding_tables[slot] for slot in range(len(operands))):
            result.append(g)
    return set(result)


def _masked_negate(self: _Formula, bounds: torch.Tensor, dim: int = -2):
    """negate bounds where weights are negative"""
    if hasattr(self.neuron, "weights"):
        result = bounds.where(self.neuron.weights.data >= 0, negate_bounds(bounds, dim))
        return result
    return bounds
