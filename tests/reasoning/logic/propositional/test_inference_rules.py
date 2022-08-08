##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import os
import pytest
from lnn import Model, Proposition, Implies, And, Or, Not, Fact, truth_table_dict

TRUE = Fact.TRUE
FALSE = Fact.FALSE

# https://en.wikipedia.org/wiki/List_of_rules_of_inference


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def p(model):
    return Proposition("p")


@pytest.fixture
def q(model):
    return Proposition("q")


@pytest.fixture
def r(model):
    return Proposition("r")


def assert_rule_for_truth_table(model, rule, variables):
    for test_case in truth_table_dict(*variables):
        model.add_data(test_case)
        model.infer()
        assert rule.state() is TRUE


def test_modus_ponens(model, p, q):
    """
    (p ∧ (p ➞ q)) ➞ q
    """
    # Downwards
    p_implies_q = Implies(p, q)
    modus_ponens = Implies(And(p, p_implies_q), q)
    model.add_knowledge(modus_ponens)

    model.add_data({p: TRUE, p_implies_q: TRUE})

    model.infer()
    assert q.state() is TRUE

    # Upwards
    assert_rule_for_truth_table(model, modus_ponens, [p, q])


def test_modus_tolens(model, p, q):
    """
    (¬q ∧ (p ➞ q)) ➞ ¬p
    """
    # Downwards
    not_q = Not(q)
    p_implies_q = Implies(p, q)
    modus_tollens = And(not_q, p_implies_q)

    model.add_knowledge(modus_tollens)

    model.add_data({q: FALSE, p_implies_q: TRUE})

    model.infer()
    assert p.state() is FALSE

    # Upwards
    assert_rule_for_truth_table(model, modus_tollens, [p, q])


def test_associative(model, p, q, r):
    """
    ((p ∨ q) ∨ r) ➞ (p ∨ (q ∨ r))
    """
    lhs = Or(Or(p, q), r)
    rhs = Or(p, Or(q, r))
    associative = Implies(lhs, rhs)
    model.add_knowledge(associative)
    # Upwards
    assert_rule_for_truth_table(model, associative, [p, q, r])


def test_commutative(model, p, q):
    """
    (p ∧ q)  ➞ (q ∧ p)
    """
    commutative = Implies(And(p, q), And(q, p))
    model.add_knowledge(commutative)

    # Upwards
    assert_rule_for_truth_table(model, commutative, [p, q])


def test_exportation(model, p, q, r):
    """
    ((p ∧ q)  ➞ r) ➞ (p ➞ (q ➞ r))
    """
    exportation = Implies(Implies(And(p, q), r), Implies(p, Implies(q, r)))
    model.add_knowledge(exportation)
    # Upwards
    assert_rule_for_truth_table(model, exportation, [p, q, r])


def test_transposition(model, p, q):
    """
    (p ➞ q)  ➞ (¬q ➞ ¬p)
    """
    transposition = Implies(Implies(p, q), Implies(Not(q), Not(p)))
    model.add_knowledge(transposition)
    # Upwards
    assert_rule_for_truth_table(model, transposition, [p, q])


def test_hypothetical_syllogism(model, p, q, r):
    """
    ((p ➞ q) ∧ (q ➞ r)) ➞ (p ➞ r)
    """
    hypothetical_syllogism = Implies(And(Implies(p, q), Implies(q, r)), Implies(p, r))
    model.add_knowledge(hypothetical_syllogism)
    # Upwards
    assert_rule_for_truth_table(model, hypothetical_syllogism, [p, q, r])


def test_material_implication(model, p, q):
    """
    (p ➞ q)  ➞ (¬p ∨ q)
    """
    material_implication = Implies(Implies(p, q), Or(Not(p), q))
    model.add_knowledge(material_implication)
    # Upwards
    assert_rule_for_truth_table(model, material_implication, [p, q])


def test_distributive(model, p, q, r):
    """
    ((p ∨ q) ∧ r) ➞ ((p ∧ r) ∨ (q ∧ r))
    """
    distributive = Implies(And(Or(p, q), r), Or(And(p, r), And(q, r)))
    model.add_knowledge(distributive)

    # Upwards
    assert_rule_for_truth_table(model, distributive, [p, q, r])


def test_absorption(model, p, q):
    """
    (p ➞ q) ➞ (p ➞ (p ∧ q))
    """
    absorption = Implies(Implies(p, q), Implies(p, And(p, q)))
    model.add_knowledge(absorption)
    # Upwards
    assert_rule_for_truth_table(model, absorption, [p, q])


def test_disjunctive_syllogism(model, p, q):
    """
    ((p ∨ q) ∧ ¬p) ➞ q
    """
    lhs = And(Or(p, q), Not(p))
    disjunctive_syllogism = Implies(lhs, q)
    model.add_knowledge(disjunctive_syllogism)
    # Downwards
    model.add_data({lhs: TRUE})
    model.infer()
    assert q.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, disjunctive_syllogism, [p, q])


def test_addition(model, p, q):
    """
    p ➞ (p ∨ q)
    """
    p_or_q = Or(p, q)
    addition = Implies(p, p_or_q)

    model.add_knowledge(addition)

    # Downwards
    model.add_data(
        {
            p: TRUE,
        }
    )
    model.infer()
    assert p_or_q.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, addition, [p, q])


def test_simplification(model, p, q):
    """
    (p ∧ q) ➞ p
    """
    p_and_q = And(p, q)
    simplification = Implies(p_and_q, p)

    model.add_knowledge(simplification)

    # Downwards
    model.add_data({p_and_q: TRUE})
    model.infer()
    assert p.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, simplification, [p, q])


def test_conjunction(model, p, q):
    """
    ((p) ∧ (q)) ➞ (p ∧ q)
    """
    p_and_q = And(p, q)
    conjunction = Implies(p_and_q, And(p, q))

    model.add_knowledge(conjunction)

    # Downwards
    model.add_data({p: TRUE, q: TRUE})
    model.infer()
    assert p_and_q.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, conjunction, [p, q])


def test_double_negation(model, p):
    """
    p ➞ (¬¬p)
    """

    not_not_p = Not(Not(p))
    double_negation = Implies(p, not_not_p)

    model.add_knowledge(double_negation)

    # Downwards
    model.add_data(
        {
            p: TRUE,
        }
    )
    model.infer()
    assert not_not_p.state() is TRUE

    model.flush()
    # Upwards
    assert_rule_for_truth_table(model, double_negation, [p])


def test_disjunctive_simplification(model, p):
    """
    (p ∨ p) ➞ p
    """
    disjunctive_simplification = Implies(Or(p, p), p)
    model.add_knowledge(disjunctive_simplification)

    # Upwards
    assert_rule_for_truth_table(model, disjunctive_simplification, [p])


def test_resolution(model, p, q, r):
    """
    ((p ∨ q) ∧ (¬p ∨ r)) ➞ (q ∨ r)
    """
    resolution = Implies(And(Or(p, q), Or(Not(p), r)), Or(q, r))
    model.add_knowledge(resolution)

    # Upwards
    assert_rule_for_truth_table(model, resolution, [p, q, r])


def test_disjunction_elimination(model, p, q, r):
    """
    ((p ➞ q) ∧ (r ➞ q) ∧ (p ∨ r)) ➞ q
    """
    disjunction_elimination = Implies(And(Implies(p, q), Implies(r, q), Or(p, r)), q)
    model.add_knowledge(disjunction_elimination)

    # Upwards
    assert_rule_for_truth_table(model, disjunction_elimination, [p, q, r])


if __name__ == "__main__":
    pytest.main([os.path.basename(__file__)])
