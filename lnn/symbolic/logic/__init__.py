from .binary_neuron import Equivalent, Implies
from .connective_formula import _ConnectiveFormula
from .connective_neuron import _ConnectiveNeuron
from .formula import Formula
from .function import Function
from .leaf_formula import _LeafFormula, Predicate, Predicates, Proposition, Propositions
from .n_ary_neuron import And, Or
from .n_ary_operator import Congruent
from .neural_activation import NeuralActivation
from .unary_operator import _Quantifier, Exists, ForAll, Not
from .variable import Variable, Variables

from . import formula as _formula
from . import connective_neuron as _connective_neuron

# Internal classes (protected classes) are denoted with a prefixed underscore. These
# classes follow the convention that they will be kept as internal classes while public
# classes are exposed via the public API.


_formula.subclasses = {
    "_ConnectiveNeuron": _ConnectiveNeuron,
    "_Quantifier": _Quantifier,
    "_LeafFormula": _LeafFormula,
    "And": And,
    "Congruent": Congruent,
    "Equivalent": Equivalent,
    "Exists": Exists,
    "ForAll": ForAll,
    "Implies": Implies,
    "Not": Not,
    "Or": Or,
    "Predicate": Predicate,
    "Proposition": Proposition,
}

_connective_neuron.subclasses = {
    "And": And,
    "Implies": Implies,
    "Or": Or,
}

__all__ = [
    "_ConnectiveFormula",
    "_ConnectiveNeuron",
    "_Quantifier",
    "And",
    "Congruent",
    "Equivalent",
    "Exists",
    "ForAll",
    "Formula",
    "Function",
    "Implies",
    "NeuralActivation",
    "Not",
    "Or",
    "Proposition",
    "Propositions",
    "Predicate",
    "Predicates",
    "Variable",
    "Variables",
]
