##
# Copyright 2021 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import os

import pytest
from lnn import Model, Proposition, Implies, And, Or, Not, TRUE, FALSE, truth_table_dict


# https://en.wikipedia.org/wiki/List_of_rules_of_inference


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def p(model):
    p = model["p"] = Proposition()
    return p


@pytest.fixture
def q(model):
    q = model["q"] = Proposition()
    return q


@pytest.fixture
def r(model):
    r = model["r"] = Proposition()
    return r


def assert_rule_for_truth_table(model, rule, variables):
    for test_case in truth_table_dict(*variables):
        model.add_facts(test_case)
        model.inference()
        assert rule.state() is TRUE


def test_modus_ponens(model, p, q):
    """
    (p ∧ (p ➞ q)) ➞ q
    """
    # Downwards
    p_implies_q = model["p_implies_q"] = Implies(p, q)
    modus_ponens = model["modus_ponens"] = Implies(And(p, p_implies_q), q)

    model.add_facts({p.name: TRUE, p_implies_q.name: TRUE})

    model.inference()
    assert q.state() is TRUE

    # Upwards
    assert_rule_for_truth_table(model, modus_ponens, ["p", "q"])


def test_modus_tolens(model, p, q):
    """
    (¬q ∧ (p ➞ q)) ➞ ¬p
    """
    # Downwards
    not_q = model["not_q"] = Not(q)
    p_implies_q = model["p_implies_q"] = Implies(p, q)
    modus_tollens = model["modus_tollens"] = And(not_q, p_implies_q)

    model.add_facts({q.name: FALSE, p_implies_q.name: TRUE})

    model.inference()
    assert p.state() is FALSE

    # Upwards
    assert_rule_for_truth_table(model, modus_tollens, ["p", "q"])


def test_associative(model, p, q, r):
    """
    ((p ∨ q) ∨ r) ➞ (p ∨ (q ∨ r))
    """
    lhs = model["lhs"] = Or(Or(p, q), r)
    rhs = model["rhs"] = Or(p, Or(q, r))

    associative = model["associative"] = Implies(lhs, rhs)

    # Upwards
    assert_rule_for_truth_table(model, associative, ["p", "q", "r"])


def test_commutative(model, p, q):
    """
    (p ∧ q)  ➞ (q ∧ p)
    """

    commutative = model["commutative"] = Implies(And(p, q), And(q, p))

    # Upwards
    assert_rule_for_truth_table(model, commutative, ["p", "q"])


def test_exportation(model, p, q, r):
    """
    ((p ∧ q)  ➞ r) ➞ (p ➞ (q ➞ r))
    """
    exportation = model["exportation"] = Implies(
        Implies(And(p, q), r), Implies(p, Implies(q, r))
    )

    # Upwards
    assert_rule_for_truth_table(model, exportation, ["p", "q", "r"])


def test_transposition(model, p, q):
    """
    (p ➞ q)  ➞ (¬q ➞ ¬p)
    """
    transposition = model["transposition"] = Implies(
        Implies(p, q), Implies(Not(q), Not(p))
    )

    # Upwards
    assert_rule_for_truth_table(model, transposition, ["p", "q"])


def test_hypothetical_syllogism(model, p, q, r):
    """
    ((p ➞ q) ∧ (q ➞ r)) ➞ (p ➞ r)
    """
    hypothetical_syllogism = model["hypothetical_syllogism"] = Implies(
        And(Implies(p, q), Implies(q, r)), Implies(p, r)
    )

    # Upwards
    assert_rule_for_truth_table(model, hypothetical_syllogism, ["p", "q", "r"])


def test_material_implication(model, p, q):
    """
    (p ➞ q)  ➞ (¬p ∨ q)
    """
    material_implication = model["material_implication"] = Implies(
        Implies(p, q), Or(Not(p), q)
    )

    # Upwards
    assert_rule_for_truth_table(model, material_implication, ["p", "q"])


def test_distributive(model, p, q, r):
    """
    ((p ∨ q) ∧ r) ➞ ((p ∧ r) ∨ (q ∧ r))
    """
    distributive = model["distributive"] = Implies(
        And(Or(p, q), r), Or(And(p, r), And(q, r))
    )

    # Upwards
    assert_rule_for_truth_table(model, distributive, ["p", "q", "r"])


def test_absorption(model, p, q):
    """
    (p ➞ q) ➞ (p ➞ (p ∧ q))
    """
    absorption = model["absorption"] = Implies(Implies(p, q), Implies(p, And(p, q)))

    # Upwards
    assert_rule_for_truth_table(model, absorption, ["p", "q"])


def test_disjunctive_syllogism(model, p, q):
    """
    ((p ∨ q) ∧ ¬p) ➞ q
    """
    lhs = model["lhs"] = And(Or(p, q), Not(p))
    disjunctive_syllogism = model["disjunctive_syllogism"] = Implies(lhs, q)

    # Downwards
    model.add_facts({lhs.name: TRUE})
    model.inference()
    assert q.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, disjunctive_syllogism, ["p", "q"])


def test_addition(model, p, q):
    """
    p ➞ (p ∨ q)
    """
    p_or_q = model["p_or_q"] = Or(p, q)
    addition = model["addition"] = Implies(p, p_or_q)

    # Downwards
    model.add_facts(
        {
            p.name: TRUE,
        }
    )
    model.inference()
    assert p_or_q.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, addition, ["p", "q"])


def test_simplification(model, p, q):
    """
    (p ∧ q) ➞ p
    """
    p_and_q = model["p_and_q"] = And(p, q)
    simplification = model["simplification"] = Implies(p_and_q, p)

    # Downwards
    model.add_facts({p_and_q.name: TRUE})
    model.inference()
    assert p.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, simplification, ["p", "q"])


def test_conjunction(model, p, q):
    """
    ((p) ∧ (q)) ➞ (p ∧ q)
    """
    p_and_q = model["p_and_q"] = And(p, q)
    conjunction = model["conjunction"] = Implies(p_and_q, And(p, q))

    # Downwards
    model.add_facts({p.name: TRUE, q.name: TRUE})
    model.inference()
    assert p_and_q.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, conjunction, ["p", "q"])


def test_double_negation(model, p):
    """
    p ➞ (¬¬p)
    """

    not_not_p = model["not_not_p"] = Not(Not(p))
    double_negation = model["double_negation"] = Implies(p, not_not_p)

    # Downwards
    model.add_facts(
        {
            p.name: TRUE,
        }
    )
    model.inference()
    assert not_not_p.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, double_negation, ["p"])


def test_disjunctive_simplification(model, p):
    """
    (p ∨ p) ➞ p
    """
    disjunctive_simplification = model["disjunctive_simplification"] = Implies(
        Or(p, p), p
    )

    # Upwards
    assert_rule_for_truth_table(model, disjunctive_simplification, ["p"])


def test_resolution(model, p, q, r):
    """
    ((p ∨ q) ∧ (¬p ∨ r)) ➞ (q ∨ r)
    """
    resolution = model["resolution"] = Implies(And(Or(p, q), Or(Not(p), r)), Or(q, r))

    # Upwards
    assert_rule_for_truth_table(model, resolution, ["p", "q", "r"])


def test_disjunction_elimination(model, p, q, r):
    """
    ((p ➞ q) ∧ (r ➞ q) ∧ (p ∨ r)) ➞ q
    """
    disjunction_elimination = model["disjunction_elimination"] = Implies(
        And(Implies(p, q), Implies(r, q), Or(p, r)), q
    )

    # Upwards
    assert_rule_for_truth_table(model, disjunction_elimination, ["p", "q", "r"])


if __name__ == "__main__":
    pytest.main([os.path.basename(__file__)])
