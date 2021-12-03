Python API
==========

The LNN API uses a pythonic approach for learning, reasoning and interacting 
    with neural representations of first-order logic knowledge bases.

The API follows a hierarchical model of abstraction, enabling knowledge 
  designers to construct and interact only with components of interest.

## Hierarchical Class Structure

1. Model

   A model stores the context of compute, i.e., the knowledge being reasoned over
     and the facts for which the knowledge is believed to apply.

2. Symbolic Nodes
   
   Nodes are "named" nodes that have human-interpretable behaviour
     according the type of node being specified. These nodes may directly be
     instantiated by the designer as prior knowledge  or extracted from templates
     to fit the data.

3. Neural Computations

   The underlying computations used by real-valued logical operations are 
   weighted in nature, typically using parameterised extensions of the standard 
   t-norms: Łukasiewicz, Gödel and Product logic



## Using the LNN
1. Models are Dynamic 
   - An LNN model can be initialised as an 'empty container' and populated 
     on-demand with the knowledge and data required to compute over.

     ```python
     from lnn import Model

     model = Model()
     ```
   
   - This functionality becomes of special interest in programmatic environments 
     that may have discoverable information that requires reasoning over the new 
     information while simultaneously retaining previously stored or inferred 
     facts.
   - Model knowledge and facts can be initiated with model constructors or
     populated with content on-demand


2. Knowledge as a Neural Architecture
   1. Predicates are the base building-blocks for a grounded first-order logic
      system like the LNN.
      They allow data to be populated as relations between constants or 
      more explicitly as tables of information.

      ```python
      from lnn import Predicate

      Smokes = Predicate('Smokes')
      Cancer = Predicate('Cancer')
      Friends = Predicate('Friends', arity=2)
      ```
      - The arity (default = 1) represents the number of columns in the table,
        where the rows will be filled by [facts in the table](#data)
      - Predicates require names, e.g., `'Smokes'`, which is used to identify
        the node within the scope of the model

   2. Where standard neural networks abstractly model knowledge into a neural 
      architecture, LNNs do so explicitly, using first-order logic formulae
      - An explicit representation constructs neurons directly from the formulae, 
        using a 1-to-1 mapping to construct the model neurons. The model 
        architecture is therefore symbolically defined and precise, as defined 
        by the required semantic expression.
      - First we instantiate a free variable

        ```python
        from lnn import Variable

        x = Variable('x', dtype='person')
        y = Variable('y', dtype='person')
        ```
        so that "grounding management" understands how to compose subformulae 
        during table joining operations
      - FOL formula can be constructed using neural connectives as follows:

        ```python
        from lnn import Implies, Bidirectional  # additional connectives: And, Or, Not

        model['Smoking causes Cancer'] = Implies(Smokes(x), Cancer(x))
        model['Smokers befriend Smokers'] = Implies(Friends(x, y), Bidirectional(Smokes(x), Smokes(y)))
        ```
        - Model assignment operations, i.e.,

          `model['Smoking causes Cancer'] = ...` can be used to inject new
              formulae directly into the model, including predicates
        - Each injected formula requires a unique name, as offered by the
          assignment key, e.g., `'Smoking causes Cancer'`. 
          This is used to identify the node within the `model-scope`
        - Alternatively root level formulae can be collected and inserted into 
          the model en masse:
          ```python
          from lnn import AXIOM  # additional world assumptions: OPEN, CLOSED

          formulae = [
              Implies(Smokes(x), Cancer(x), 
                      name='Smoking causes Cancer'),
              Implies(Friends(x, y), Bidirectional(Smokes(x), Smokes(y)), 
                      name='Smokers befriend Smokers')
          ]
          model.add_formulae(*formulae, world=AXIOM)
          ```
          - All nodes requiring handles for lookup or directly for symbolic 
            operations, should include a `name` via the kwarg
          - The compositional structure of the LNN will recursively build up a 
            connective tree structure per root formula and the insertion of a
            root will also include all connected subformulae.
          - Formulae can also be modified to follow a truth `world` assumption
            using the kwarg, i.e., `OPEN` or `CLOSED`
            [world assumption](https://en.wikipedia.org/wiki/Closed-world_assumption). 
            - This places restrictions on the symbolic truths to only consider
              worlds where the facts are initialised as `UNKNOWN` or `FALSE` 
              accordingly. 
            - By default, LNNs make no assumption of perfect knowledge and opt 
              for `OPEN` assumptions on all formulae unless otherwise specified.  
              e.g. A safer assumption from the above scenario may assume
              ```python
              Cancer = Predicate('Cancer', world=CLOSED)
              ```
              to prevent false negatives from downward inference, opting to trust
              the data instead of assuming the formulae perfectly describes the 
              data 
          - Alternatively a formula can be defined as an `AXIOM`, thereby 
              limiting truths to worlds where facts are assumed `TRUE`
            
   3. Connectives are "neural" due to their parameterised computation or 
      implementation as a weighted real-value t-norm. 
      - The base neural connectives in the LNN are `And`, `Or`, `Implies`
      - Note that `Not` gates, albeit a connective, is not neural in nature
      - Compound connectives, e.g., `Bidirectional`, `Xor`, etc. are implemented
        from these 4 connectives
      - The parameterised configuration of each neuron can be overloaded
         using the `neuron` kwarg. See [neuron module] documentation for more.

3. Tables of Data <a name="data"></a>
   ```python
   # add data to the model
   model.add_facts({
       'Friends': {
           ('Anna', 'Bob'): TRUE,
           ('Bob', 'Anna'): TRUE,
           ('Anna', 'Edward'): TRUE,
           ('Edward', 'Anna'): TRUE,
           ('Anna', 'Frank'): TRUE,
           ('Frank', 'Anna'): TRUE,
           ('Bob', 'Chris'): TRUE},
       Smokes.name: {
           'Anna': TRUE,
           'Edward': TRUE,
           'Frank': TRUE,
           'Gary': TRUE},
       Cancer.name: {
           'Anna': TRUE,
           'Edward': TRUE}
       })
   model.print()
   ```
   Output:
   ```raw
   *************************************************************
                                LNN Model

    AXIOM  Implies: Smokers befriend Smokers('x', 'y') 
   
    OPEN   Bidirectional: Bidirectional_0('x', 'y') 
   
    OPEN   Implies: Implies_2('y', 'x') 
   
    OPEN   Implies: Implies_1('x', 'y') 
   
    OPEN   Predicate: Friends('x0', 'x1') 
           ('Bob', 'Anna')                     TRUE (1.0, 1.0)
           ('Anna', 'Bob')                     TRUE (1.0, 1.0)
           ('Anna', 'Edward')                  TRUE (1.0, 1.0)
           ('Frank', 'Anna')                   TRUE (1.0, 1.0)
           ('Bob', 'Chris')                    TRUE (1.0, 1.0)
           ('Edward', 'Anna')                  TRUE (1.0, 1.0)
           ('Anna', 'Frank')                   TRUE (1.0, 1.0)

    AXIOM  Implies: Smokers have Cancer(x) 
   
    CLOSED Predicate: Cancer(x0) 
           'Edward'                            TRUE (1.0, 1.0)
           'Anna'                              TRUE (1.0, 1.0)
    
    OPEN   Predicate: Smokes(x0) 
           'Frank'                             TRUE (1.0, 1.0)
           'Edward'                            TRUE (1.0, 1.0)
           'Anna'                              TRUE (1.0, 1.0)
           'Gary'                              TRUE (1.0, 1.0)

    ************************************************************
   ```
   
4. Reasoning until convergence <a name="reasoning"></a>
   - Using the above formulae and specified facts, inference runs across the 
   entire model until convergence - inferring all possible facts that can be
   inferred.
   - inference can be computed over the entire model by calling
     ```python
     model.infer()
     ```

5. Learning in the presence of noise <a name="learning"></a>
   - Learning in the LNN can be done with supervisory signals but also under self
     supervision, using the rules to enforce logical consistency.
   - Supervisory learning uses labelled targets on truth bounds, which can be 
     set as follows:

       ```python
       # add supervisory targets
       model.add_labels({
           'Smokes': {
               'Ivan': TRUE,
               'Nick': TRUE}
           })
    
       # train the model and output results
       model.train(losses=['supervised'])
       model.print(params=True)
       ```
   - Self-supervisory signals can check for logical consistency, i.e., ensuring
     that the lower bounds do not cross above the upper bounds.
     This can be set using by adding the loss kwarg:
     ```python
     loss=['contradiction']
     ```
   - Additional losses that may be of interest are `logical` and `uncertainty` 
     losses
     - Logical losses 
       - Logical losses enforce the logical constraints, see 
         [Section E](https://arxiv.org/pdf/2006.13155.pdf) for more details.
         - Enforcement of constraints are required for environments that require a
           'hard-logic' interpretation of neurons, however, this presents the
           LNN with a non-trivial optimisation problem - especially in the
           classical case where noise is present
         - A soft-logic is however more desirable in most situation, allowing 
           accuracy to be a more prominent goal than interpretability - which is
           the case for most machine learning scenarios.
     - Uncertainty losses 
       - Uncertainty losses are simply the `upper - lower` bounds, allowing the 
         parameter configuration to tighten bounds and thereby minimise the amount
         of uncertainty within the output truths. 
         - This configuration is more readily used within scenarios that have
           non-classical inputs, i.e., uncertainty bounds on real-valued (0, 1)
           truths - excluding the extremes.
   - Losses can also be appended to the training list, enabling multi-task
     learning across the model

## Some design choices

  - While the LNN can handle large graphs, the current strategy of unguided 
    reasoning or reasoning over all possible points, can be time intensive.
    - Bottlenecks being addressed include: 
      - loading large numbers of formulae sequentially
      - efficiency of grounding management for large tables and deep formulae  
  - "Grounding management" or the composition of subformulae tables according to 
    the variable configuration is therefore kept to a minimal by opting for 
    inner-joins instead of outer/product joins. 
  - Albeit more efficient, these joins can still be computationally intensive 
    (2N^2 in some situations) for large n-ary tables. 
    The length of connectives should therefore be kept as compact as possible.