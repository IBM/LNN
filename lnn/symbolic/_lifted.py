##
# Copyright 2022 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

import logging
from typing import Tuple, Dict, Iterable, Union

from .. import _utils
from ..constants import World
from .logic import (
    Implies,
    Or,
    ForAll,
    And,
    Not,
    Formula,
    _Quantifier,
    Variable,
    _ConnectiveNeuron,
    _ConnectiveFormula,
    Congruent,
)


_utils.logger_setup()


def subexpression_elimination(model, formula):
    r"""Recursively replace subformulae that already exist in the model."""

    def recurse(subformula):
        if subformula.name not in model:
            for idx, operand in enumerate(subformula.operands):
                r = recurse(operand)
                subformula.operands[idx] = r
        else:
            return model[subformula]
        return subformula

    formula = recurse(formula)
    return formula


def lifted_axioms(model, rules=None) -> Dict:
    r"""Returns a dictionary of axioms, with keys as callable functions and values as
    the number of node inputs to the function."""
    axioms = dict()

    def extend_model(formula, axiom):
        r"""adds the formula to the model and logs the axiom"""
        if formula not in model:
            model.add_knowledge(formula)
            logging.info(
                f"LIFTED:'{axiom}' "
                f"ADDED:'{formula.name}' "
                f"FORMULA:{formula.formula_number}"
            )
            if isinstance(formula, Congruent):
                formula.set_congruency()
            return True
        return False

    def _u(src: Formula) -> Union[Formula, Tuple]:
        r"""Returns the subformula from the model if it exists
        may return a tuple (called formula) if src variables differ from dst variables.
        """
        if isinstance(src, Not):
            return model[src](*src.unique_vars) if src in model else src
        if src in model:
            return (
                model[src]
                if model[src].unique_vars == src.unique_vars
                else model[src](*src.unique_vars)
                if isinstance(src, _ConnectiveNeuron)
                and src.neural_equivalence(model[src])
                else src
            )
        return src

    def unify(
        replace: Iterable[Variable],
        source: Tuple[Variable, ...],
        destination: Tuple[Variable, ...],
    ) -> Tuple[Variable, ...]:
        r"""Substitutes the variables inside `replace` by applying a mapping from
        `source` to `destination`.

        Examples
        --------
        replace: (a, b, u, c, v)
        source: (b, c, a)
        destination: (y, z, x)
        returns: (x, y, u, c, v)

        """
        src_to_dst = dict(zip(source, destination))
        return tuple(src_to_dst[v] if v in src_to_dst else v for v in replace)

    def transitivity(*formulae: [Implies, Implies]) -> bool:
        r"""((p → q) → ((q → r) → (p → r))

        Transitivity axiom
        """
        p_implies_q, q_implies_r = formulae
        if all(isinstance(f, Implies) and not f.propositional for f in formulae):
            p, q0 = p_implies_q.operands
            q1, r = q_implies_r.operands
            if q0.is_equal(q1):
                src = p_implies_q.var_remap[1]
                dst = q_implies_r.var_remap[0]
                return extend_model(
                    _u(
                        ForAll(
                            _u(
                                Implies(
                                    p_implies_q(
                                        *unify(p_implies_q.unique_vars, src, dst)
                                    ),
                                    _u(
                                        Implies(
                                            q_implies_r,
                                            _u(
                                                Implies(
                                                    p(
                                                        *unify(
                                                            p_implies_q.var_remap[0],
                                                            src,
                                                            dst,
                                                        )
                                                    ),
                                                    r(*q_implies_r.var_remap[1]),
                                                )
                                            ),
                                        )
                                    ),
                                )
                            )
                        )
                    ),
                    "Transitivity",
                )
            return False
        return False

    # axioms[transitivity] = 2

    def subtractive_disjunctive_transitivity(*formulae: [Or, Or]) -> bool:
        r"""(𝜑 → (𝜓 → 𝜒))
        Implies(phi, Implies(psi, chi))

        Generalised transitivity axiom
        """

        phi, psi = formulae
        if all(
            isinstance(f, Or) and len(f.operands) == 2 and not f.propositional
            for f in formulae
        ):
            # absorb `Not` into weights for psi, phi
            operands = []
            for f in formulae:
                _operands, *_ = f.set_negative_weights(store=False)
                operands.append(_operands)

            if operands[0][1].is_equal(operands[1][0]):
                phi_weights, psi_weights = [f.neuron.weights for f in formulae]
                phi_bias, psi_bias = [f.neuron.bias for f in formulae]

                # define chi separately in case it already exists - w,b to be updated
                src = phi.var_remap[1]
                dst = psi.var_remap[0]
                chi = _u(
                    Or(
                        phi.operands[0](
                            *unify(
                                phi.var_remap[0],
                                src,
                                dst,
                            )
                        ),
                        psi.operands[1](*psi.var_remap[1]),
                    ),
                )
                chi_neuron = chi[0] if isinstance(chi, tuple) else chi
                chi_neuron.neuron.weights = phi_weights + psi_weights
                chi_neuron.neuron.bias = phi_bias + psi_bias

                return extend_model(
                    _u(
                        ForAll(
                            _u(
                                Implies(
                                    phi(*unify(phi.unique_vars, src, dst)),
                                    _u(
                                        Implies(
                                            psi,
                                            chi,
                                        )
                                    ),
                                )
                            )
                        )
                    ),
                    "Subtractive Disjunctive Transitivity",
                )
            return False
        return False

    axioms[subtractive_disjunctive_transitivity] = 2

    def equality_introduction(
        formula: _ConnectiveFormula, other: _ConnectiveFormula
    ) -> bool:
        r"""Add a symbolic equality to nodes that share equal operands."""
        formulae = [formula, other]
        if (
            formula.__class__ == other.__class__
            and formula.var_remap == other.var_remap
            and all(
                [
                    not f.propositional and isinstance(f, _ConnectiveFormula)
                    for f in formulae
                ]
            )
        ):
            equal_operands = all(
                [
                    True
                    if any(
                        [
                            formula_operand.is_equal(other_operand)
                            for other_operand in other.operands
                        ]
                    )
                    else False
                    for formula_operand in formula.operands
                ]
            )
            return (
                extend_model(
                    Congruent(formula, other),
                    "Equality Introduction",
                )
                if equal_operands
                else False
            )
        return False

    axioms[equality_introduction] = 2

    def de_morgan(formula: And) -> bool:
        r"""(p ∧ q) = ¬(¬p ∨ ¬q)"""
        if not formula.propositional and isinstance(formula, And):
            P_vars, Q_vars = formula.var_remap
            P, Q = formula.operands
            return extend_model(
                _u(
                    Congruent(
                        formula,
                        _u(
                            Not(
                                _u(
                                    Or(
                                        _u(Not(P(*P_vars))),
                                        _u(Not(Q(*Q_vars))),
                                    )
                                )
                            )
                        ),
                    )
                ),
                "De Morgan ∧ → ∨",
            )
        return False

    axioms[de_morgan] = 1

    def material_implication(formula: Implies) -> bool:
        r"""(p → q) = (¬p ∨ q)"""
        if not formula.propositional and isinstance(formula, Implies):
            P_vars, Q_vars = formula.var_remap
            P, Q = formula.operands
            return extend_model(
                _u(Congruent(formula, _u(Or(_u(Not(P(*P_vars))), Q(*Q_vars))))),
                "Material Implication",
            )
        return False

    axioms[material_implication] = 1

    def generalisation(formula: Formula) -> bool:
        r"""Axiom(p) → ∀p"""
        if (
            not isinstance(formula, _Quantifier)
            and not formula.propositional
            and formula.world_state() is World.AXIOM
        ):
            return extend_model(ForAll(formula), "Generalisation")
        return False

    axioms[generalisation] = 1

    def disjunction_elimination(*formulae: [Implies, Implies, Or]):
        """
        ((p → q) ∧ (r → q) ∧ (p ∨ r)) → q
        """
        if (
            all(isinstance(f, Implies) for f in formulae[:2])
            and isinstance(formulae[-1], Or)
            and len(formulae[-1].operands) == 2
        ):
            p_implies_q, r_implies_q, p_or_r = formulae
            p_0, q_0 = p_implies_q.operands
            r_1, q_1 = r_implies_q.operands
            p_2, r_2 = p_or_r.operands
            if p_0.is_equal(p_2) and q_0.is_equal(q_1) and r_1.is_equal(r_2):
                return extend_model(
                    _u(
                        ForAll(
                            _u(
                                Implies(
                                    _u(And(p_implies_q, r_implies_q, p_or_r)),
                                    q_0,
                                )
                            )
                        )
                    ),
                    "Disjunction Elimination",
                )
            return False
        return False

    axioms[disjunction_elimination] = 3

    if rules:
        result = dict()
        for r in rules:
            axioms[eval(r)] = axioms[eval(r)]
        return result
    else:
        return axioms
