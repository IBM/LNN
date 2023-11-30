from .binary_neuron import Iff, Implies
from .connective_formula import _ConnectiveFormula
from .connective_neuron import _ConnectiveNeuron
from .formula import Formula
from .leaf_formula import _LeafFormula, Predicate, Predicates, Proposition, Propositions
from .n_ary_neuron import And, Or, Xor
from .n_ary_operator import Equal
from .neural_activation import NeuralActivation
from .unary_operator import _Quantifier, Exists, Forall, Not
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
    "Equal": Equal,
    "Iff": Iff,
    "Exists": Exists,
    "Forall": Forall,
    "Implies": Implies,
    "Not": Not,
    "Or": Or,
    "Predicate": Predicate,
    "Proposition": Proposition,
    "Xor": Xor,
}

_connective_neuron.subclasses = {
    "And": And,
    "Iff": Iff,
    "Implies": Implies,
    "Or": Or,
    "Xor": Xor,
}

__all__ = [
    "_ConnectiveFormula",
    "_ConnectiveNeuron",
    "_Quantifier",
    "And",
    "Equal",
    "Iff",
    "Exists",
    "Forall",
    "Formula",
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
    "Xor",
]
