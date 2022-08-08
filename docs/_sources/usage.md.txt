Python API
==========

The LNN API uses an intuitive pythonic approach for learning, reasoning and interacting with neural representations of first-order logic knowledge bases.

The API follows a hierarchy of interacting components, allowing designers to construct and interact with knowledge at a desired level of granularity.

## Hierarchical Class Structure

1. [Model](lnn/LNN.html#lnn.Model)

   A model provides a context or theory of reference for which reasoning or training can be applied.

2. [Symbolic Nodes](lnn/LNN.html#symbolic-structure)

   Nodes are human-interpretable containers that follow a predefined category of behavior or compute according to the type of node being specified. These nodes may directly be instantiated by the designer as prior knowledge  or extracted from templates to fit the data.

3. [Neural Computations](#neural-configuration)

   The underlying computations used by real-valued logical operations are 
   weighted in nature, typically using parameterised extensions of the standard 
   t-norms: Łukasiewicz, Gödel and Product logic



## Using the LNN

### Symbolic Structure

1. Dynamic Models
   - An LNN model can be initialised as an "empty container" and populated
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
   

2. Knowledge Represention
   1. [Predicates](lnn/LNN.html#lnn.Predicate) are the base building-blocks for a grounded first-order logic system like the LNN.
      Predicates provide both properties of and relations between objects. Unary predicates provide properties and n(>1)-ary predicates provided relations. Predicates in LNNs explicitly store tables of information along with truth values associated with each row of the truth table.

      ```python
      from lnn import Predicates

      Smokes, Cancer = Predicates('Smokes', 'Cancer')
      Friends = Predicates('Friends', arity=2)
      ```
      - The arity (default = 1) represents the number of columns in the table,
        where the rows will be filled by [facts](#data) in the table.
      - The inputs for each row of the truth table form a unique `Grounding` that tells the predicate which row is being operated on.
      - Predicates (and [Propositions](lnn/LNN.html#lnn.Proposition)) require names, e.g., `'Smokes'`, which is used to identify
        the node within the scope of the model

   2. Where standard neural networks abstractly model knowledge into a neural 
      architecture, LNNs do so explicitly, using first-order logic formulae
      - An explicit representation constructs neurons directly from the formulae, 
        using a 1-to-1 mapping to construct the model neurons. The model 
        architecture is therefore symbolically defined and precise, as defined 
        by the required semantic expression.
      - First we instantiate a free variable

        ```python
        from lnn import Variables

        x, y = Variables('x', 'y')
        ```
        so that "grounding management" understands how to compose subformulae
        during table joining operations
        - FOL formula can be constructed using neural connectives as follows:

          ```python
          from lnn import Implies, Equivalent

          Smoking_causes_Cancer = Implies(Smokes(x), Cancer(x))
          Smokers_befriend_Smokers = Implies(Friends(x, y), Equivalent(Smokes(x), Smokes(y)))
          ```
            - All nodes requiring handles for lookup or directly for symbolic operations, should include a `name` via the kwarg
          
          </br>
          Root level formulae can be collected and inserted into the model en masse:

            ```python
            from lnn import World

            formulae = [
                Smoking_causes_Cancer
                Smokers_befriend_Smokers
            ]
            model.add_knowledge(*formulae, world=World.AXIOM)
            ```
            - Additional [World](lnn/LNN.html#lnn.World) assumptions include: `[World.OPEN, World.CLOSED]` to enforce that formulae follow the open/closed world assumption
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
                Cancer = Predicate('Cancer', world=World.CLOSED)
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
      - Compound connectives, e.g., `Equivalent`, `Xor`, etc. are implemented
        from these 4 connectives
      - The parameterised configuration of each neuron can be overloaded
         using the `neuron` kwarg. See [neuron module] documentation for more.

3. Tables of Data <a name="data"></a>
   ```python
   from lnn import Fact
   
   # add data to the model
   model.add_data({
       Friends: {
           ('Anna', 'Bob'): Fact.TRUE,
           ('Bob', 'Anna'): Fact.TRUE,
           ('Anna', 'Edward'): Fact.TRUE,
           ('Edward', 'Anna'): Fact.TRUE,
           ('Anna', 'Frank'): Fact.TRUE,
           ('Frank', 'Anna'): Fact.TRUE,
           ('Bob', 'Chris'): Fact.TRUE},
       Smokes.name: {
           'Anna': Fact.TRUE,
           'Edward': Fact.TRUE,
           'Frank': Fact.TRUE,
           'Gary': Fact.TRUE},
       Cancer.name: {
           'Anna': Fact.TRUE,
           'Edward': Fact.TRUE}
       })
   model.print()
   ```
   Output:
   ```raw
   *************************************************************
                                LNN Model

    AXIOM  Implies: Smokers befriend Smokers('x', 'y') 
   
    OPEN   Equivalent: Equivalent_0('x', 'y') 
   
    OPEN   Implies: Implies_2('y', 'x') 
   
    OPEN   Implies: Implies_1('x', 'y') 
   
    OPEN   Predicate: Friends
           ('Bob', 'Anna')                     TRUE (1.0, 1.0)
           ('Anna', 'Bob')                     TRUE (1.0, 1.0)
           ('Anna', 'Edward')                  TRUE (1.0, 1.0)
           ('Frank', 'Anna')                   TRUE (1.0, 1.0)
           ('Bob', 'Chris')                    TRUE (1.0, 1.0)
           ('Edward', 'Anna')                  TRUE (1.0, 1.0)
           ('Anna', 'Frank')                   TRUE (1.0, 1.0)

    AXIOM  Implies: Smokers have Cancer(x) 
   
    CLOSED Predicate: Cancer
           'Edward'                            TRUE (1.0, 1.0)
           'Anna'                              TRUE (1.0, 1.0)
    
    OPEN   Predicate: Smokes
           'Frank'                             TRUE (1.0, 1.0)
           'Edward'                            TRUE (1.0, 1.0)
           'Anna'                              TRUE (1.0, 1.0)
           'Gary'                              TRUE (1.0, 1.0)

    ************************************************************
   ```
   
4. Reasoning <a name="reasoning"></a>
   - Using the above formulae and specified facts, inference runs across the 
   entire model until convergence - inferring all possible facts that can be
   inferred.
   - inference can be computed over the entire model by calling
     ```python
     model.infer()
     ```

5. Learning <a name="learning"></a>
   - Learning in the LNN can be done with supervisory signals but also under self
     supervision, using the rules to enforce logical consistency.
   - Supervisory learning uses labelled targets on truth bounds, which can be
     set as follows:

       ```python
       from lnn import Loss
     
       # add supervisory targets
       model.add_labels({
           'Smokes': {
               'Ivan': Fact.TRUE,
               'Nick': Fact.TRUE}
           })
    
       # train the model and output results
       model.train(losses=Loss.SUPERVISED)
       model.print(params=True)
       ```
   - Self-supervisory signals can also enforce logical consistency, i.e., ensuring
     that the lower bounds do not cross above the upper bounds.
     This can be set by adding additional losses to the kwarg, to jointly optimise 
     multiple loss functions.
     ```python
     losses=[Loss.SUPERVISED, Loss.CONTRADICTION]
     model.train(losses=losses)
     ```
   - The following are additional [losses](lnn/LNN.html#lnn.Loss) that may be of interest:
     losses
     - Logical loss
       - Logical losses enforce the logical constraints, see 
         [Section E](https://arxiv.org/pdf/2006.13155.pdf) for more details.
         - Enforcement of constraints are required for environments that require a
           'hard-logic' interpretation of neurons, however, this presents the
           LNN with a non-trivial optimisation problem - especially in the
           classical case where noise is present
         - A soft-logic is however more desirable in most situation, allowing 
           accuracy to be a more prominent goal than interpretability - which is
           the case for most machine learning scenarios.
     - Uncertainty loss
       - Uncertainty losses are simply the `upper - lower` bounds, allowing the 
         parameter configuration to tighten bounds and thereby minimise the amount
         of uncertainty within the output truths. 
         - This configuration is more readily used within scenarios that have
           non-classical inputs, i.e., uncertainty bounds on real-valued (0, 1)
           truths - excluding the extremes.
   - Losses can also be appended to the training list, enabling multi-task
     learning across the model

6. Interpretability <a name="interpretability"></a>

   - [Printing](lnn/LNN.html#lnn.Formula.print)
     - You would have noticed that the LNN is symbolically interpretable for every 
       subformula in the model, which can easily be extracted using a print out:
       ```python
       model.print()
       ```
       This prints out the groundings and respective truths for each node in the graph. 
     - The above printout may, however, be very large for models with a large amount of 
       knowledge and data. A subset of the model can also be printed out:
       ```python
       model.print(source=Smokers_befriend_Smokers)
       ```
       This uses the source node as a root, and a print out will be done for all
       nodes that are chilrden of the root.
     - Alternatively, a print out can be done for just a single node
       ```python
       Smokers_befriend_Smokers.print()
       ```
   - [States](lnn/LNN.html#lnn.Formula.state)
     - For every truth value associated with a grounding, there is an associated 
       state (from 9 possible classes).
     - If the states of truths need to be extracted, this allows the state of a formula
       or a particular grounding to extracted
       ```python
       Cancer.state('Edward')
       ```
       returns a single state construct, whereas
       ```python
       Cancer.state()
       ```
       returns a dictionary of states, keyed by the groundings


### Activation Configuration

#### Neural Configuration
The LNN offers a generic framework for computing symbolic connectives via different
classes of neural computations according to the user specification. We can therefore
keep the symbolic containers the same and switch out the underlying activations for
different results.

Neural parameters:

| parameter | type |  |
|---|---|---|
|type|[NeuralActivation](lnn/LNN.html#lnn.NeuralActivation)|



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
