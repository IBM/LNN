##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from typing import TypeVar
import networkx as nx

"""
Visualization module

All functions in this module assume a Model level scope
Algorithm inspired by https://rdrr.io/bioc/Rgraphviz/man/GraphvizLayouts.html
"""

Model = TypeVar("Model")


def get_roots(nodes, edges):
    roots = set(nodes)
    for _, y in edges:
        roots.discard(y)
    roots = [r for r in roots]
    roots.sort(key=lambda x: x if isinstance(x, str) else x.name)
    return roots


def get_leaves(nodes, edges):
    leaves = set(nodes)
    for x, _ in edges:
        leaves.discard(x)
    return leaves


def get_adjacency_list(edges):
    # get neighbors
    children = {}
    for x, y in edges:
        if x not in children:
            children[x] = set()
        children[x].add(y)
    return children


def get_ranking(adj_list, ranking_dict, node, current_level):
    if node not in ranking_dict:
        ranking_dict[node] = current_level
    else:
        ranking_dict[node] = max(ranking_dict[node], current_level)
    if node not in adj_list:
        return ranking_dict
    for child in adj_list[node]:
        ranking_dict = get_ranking(adj_list, ranking_dict, child, current_level + 1)
    return ranking_dict


def init_ordering(node, seen, ranking_dict, adj_list, rank_order):
    if node in seen:
        return rank_order
    rank = ranking_dict[node]
    if rank not in rank_order:
        rank_order[rank] = []
    rank_order[rank].append(node)
    seen.add(node)
    if node not in adj_list:
        return rank_order
    children = [c for c in adj_list[node]]
    children.sort(key=lambda x: x if isinstance(x, str) else x.name)
    for child in children:
        rank_order = init_ordering(child, seen, ranking_dict, adj_list, rank_order)
    return rank_order


def get_pos(self: Model):
    original_nodes = self.graph.nodes()
    original_edges = self.graph.edges()

    # rank
    roots = get_roots(original_nodes, original_edges)
    leaves = get_leaves(original_nodes, original_edges)
    adj_list = get_adjacency_list(original_edges)

    ranking_dict = dict()

    def get_ranks(node, layers):
        if node in layers:
            return layers
        if node in leaves:
            layers[node] = 0
            return layers
        else:
            for child in adj_list[node]:
                layers = get_ranks(child, layers)
            max_child_layer = -1
            for child in adj_list[node]:
                max_child_layer = max(layers[child], max_child_layer)
            layers[node] = max_child_layer + 1
            return layers

    for root in roots:
        ranking_dict = get_ranks(root, ranking_dict)

    # add virtual nodes
    virtual_nodes = set()
    for node_x, node_y in original_edges:
        rank_x = ranking_dict[node_x]
        rank_y = ranking_dict[node_y]
        while rank_x - rank_y > 1:
            middle_node = "virtual_node_" + str(len(virtual_nodes))
            ranking_dict[middle_node] = rank_y + 1
            adj_list[node_x].add(middle_node)
            adj_list[middle_node] = set([node_y])
            node_y = middle_node
            rank_y = rank_y + 1
            virtual_nodes.add(middle_node)

    virtual_graph = nx.Graph()
    virtual_graph.add_nodes_from(virtual_nodes)
    virtual_graph.add_nodes_from(original_nodes)
    for u in adj_list:
        for v in adj_list[u]:
            virtual_graph.add_edge(u, v)

    # assume square window - can expand
    height = 1
    width = 1

    # init ordering
    rank_ordering = {}
    seen = set()
    for root in roots:
        rank_ordering = init_ordering(root, seen, ranking_dict, adj_list, rank_ordering)
    vert_gap = height / (max(rank for rank in rank_ordering) + 1)

    # skipping reordering

    # position
    pos = dict()
    for rank in rank_ordering:
        nodes_in_rank = rank_ordering[rank]
        dy = vert_gap * rank
        dx = 1 / len(nodes_in_rank)
        left = dx / 2
        for i, node in enumerate(nodes_in_rank):
            pos[node] = ((left + dx * i) * width, dy)

    return pos
