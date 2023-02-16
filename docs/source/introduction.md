# Logical Neural Networks

<img src="https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/lnn_structure.png" alt="LNN structure" width="400" class="aligncenter"/>

The LNN is a form of recurrent neural network with a 1-to-1 correspondence to a set of logical formulae in any of 
various systems of ___weighted, real-valued logic___, in which evaluation performs logical inference. The graph 
structure therefore directly reflects the logical formulae it represents.

Key innovations that set LNNs aside from other neural networks are
1. neural activation functions __constrained__ to implement the truth functions of the logical operations they 
represent, i.e. `And`, `Or`, `Not`, `Implies`, and, in FOL, `Forall` and `Exists`,
2. results expressed in terms of __bounds__ on truth values so as to distinguish known, approximately known, unknown, 
and contradictory states,
3. __bidirectional inference__ permitting, e.g., `x → y` to be evaluated as usual in addition to being able to prove `y`
given `x` or, just as well, `~x` given `~y`.

The nature of the modeled system of logic depends on the family of activation functions chosen for the network's 
neurons, which implement the logic's various atoms (i.e. propositions or predicates) and operations.

In particular, it is possible to constrain the network to behave exactly classically when provided classical input.
Computation is characterized by tightening ___bounds___ on truth values at neurons pertaining to subformulae in 
___upward___ and ___downward passes___ over the represented formulae's syntax trees.

Bounds tightening is monotonic; accordingly, computation cannot oscillate and necessarily converges for propositional 
logic.
Because of the network's modular construction, it is possible to partition and/or compose networks, inject formulae 
serving as logical constraints or queries, and control which parts of the network (or individual neurons) are trained or
evaluated.

Inputs are initial truth value bounds for each of the neurons in the network; in particular, neurons pertaining to 
predicate atoms may be populated with truth values taken from KB data. Additional inputs may take the form of injected 
formulae representing a query or specific inference problem.

Outputs are typically the final computed truth value bounds at one or more neurons pertaining to specific atoms or 
formulae of interest.

In other problem contexts, the outputs of interest may instead be the neural parameters themselves &mdash; serving as a
form of inductive logic programming (ILP) &mdash; after learning with a given loss function and input training data set.
