# Moving AI Beyond its Narrow View of Intelligence

> _We've all seen the headlines: "Machine Outsmarts Human", but can a
    computational system truly be smart if it lacks the ability to explain itself
    - even to the very researchers that created it?_

Despite achieving accuracies that surpass human-level performance on narrowly
    defined tasks, today's AI has become handicapped by the very methodology that
    brought it fame - Deep Learning. These deep neural structures owe their success
    to immense human ingenuity, by encoding knowledge of the world not into
    rules, but instead into a complex network of connected neurons within the
    model  architecture. This complexity mandates that researchers think
    abstractly about how the network behaves, guided by very few interpretable
    signals that characterise a model's behavior.

[<img src="https://www.macloo.com/ai/wp-content/uploads/2020/09/machine_learning_XKCD.png" alt="ML Systems" width="400" class="aligncenter">](https://xkcd.com/1838)

## So what's missing in today's AI?

While "opaque box" models are all the rage right now, they still lack the "per-neuron" symbolic interpretability required for machine learning to generalise to broader tasks. By grounding each neuron's behavior in an understandable way, researchers can change the way they engage with these complex architectures to be more principled. This also opens up the doors to using ML in regulated environments, where a deployed machine is required to justify its decision making in a way that people can understand. This is where IBM researchers come in. They have managed to merge the well-respected field of logic with neural backpropagation, allowing gradient decent to apply to hand-crafted and noisy knowledge of the world [[1](#ref_1)].

## Why a logical representation is important

Logical operations are arguably human-interpretable, constraining operations on
    inputs to behave in a manner that is both predictable and consistent with the
    type of gate being used. For example, if our knowledge of the world states
    the following:

> 1. If it rains then the grass is wet

We expect that any decision-making system should be able to apply logical rules
    such as modus ponens to our knowledge, i.e. reasoning that the grass is
    indeed wet when I know that it is raining - without needing to see a
    Bajillion examples first. Simply, the DL has no extrinsic
    knowledge, leveraging only a heap of correlations to learn that the two
    inputs may be strongly related. But honestly, how can we expect a DL system 
    to conform to such knowledge without having an explicit handle on neurons
    within the hidden layers that encode information. While it is possible to
    encode such relations implicitly, there is a computational and therefore
    financial and environmental cost to building such systems - which should
    really not be there, since our network only needs 3 neurons to compute on
    such knowledge:

<img src="https://raw.githubusercontent.com/IBM/LNN/master/docsrc/source/education/blogs/raining_implies_wet.png" alt="It is raining Implies the grass is wet" width="320" class="aligncenter">

Logic also allows us to build high-level decision makers that can reason about
    outcomes given only partial information about the world. Lets add some more
    knowledge to our model:

> 2. I need to walk on the grass to reach the rosebushes
> 3. If the grass is wet then I should not walk on it without rubber boots on

> Goal: Trim the rosebushes

Given only the following two pieces of information should be sufficient to know if I can reach my goal:
> - It is raining
> - I am not wearing rubber boots

Without the ability to reason, even large models would fail to reach simple goals in an efficient manner [[2](#ref_2)]. Using a logical system would therefore allow an agent to reason that it needs to make certain (interpretable) decisions in order to reach its goal.

## Logical Neural Networks (LNNs)

LNNs are a mathematically follow a sound and complete extension to weighted real-valued logic [[3](#ref_3)], offering a new class of NeuroSymbolic approaches according to Henry Kautz's taxonomy [[4](#ref_4)], namely `Neuro = Symbolic`. This framework offers a handful of novel features:


Knowledge can embedded in a neural network via logical statements using specialised activations to compute logical operations.
A 1-to-1 representation is used whereby each logical symbol is directly encoded by a single neuron, representing the network as an interpretable syntax tree.
The LNN is also implemented using a dynamic graph representation where reasoning is computed via message passing algorithms, allowing new facts and rules to be added to the model on the fly.
With this syntax tree, inferences such as modus ponens require facts at leaf nodes to be updated - facilitated by downward inferences, allowing the network to both handle missing inputs and verify consistent interaction between the rules and facts.
This consistency checking allows LNNs to learn purely by self-supervision, while still allowing everybody's favourite loss function to join the party.
Using bounds to represent inputs allows the LNN to operate under the open world assumption, also allowing nuance by having ranges of inputs being plausible at a single neuron.
Just like standard NNs, weights are attached to edges and learning via backprop caters to imperfect knowledge of the world.
The system is also end-to-end differentiable, allowing LNNs to play nice with multiple DL systems simultaneously. These DL models can still do what they do best - act as function approximators of hierarchical features that output to a single node, but in this case, the node may be given as an input to an LNN and governed by the logic that is expected of its symbolic behaviour.

With all this added functionality, LNNs offer a completely new class of learning-based approaches to the AI toolbox.

As the modern economy moves towards embedding real-time computation into every aspect of life, so too will AI follow. The ability to generalise beyond narrow tasks requires deployed systems to simultaneously obey rules of inference while still being robust against an ocean of noisy, unstructured data. It stands to reason [get it?], that having a differentiable white-box model that acts as a symbolic decision-maker is needed for the next generational leap in the field of AI. Perhaps advances like these will move the scientific community forward; towards an intelligence that is both generalisable and understandable.



## References


1. [Logical Neural Networks][1]<a name="ref_1"></a>
2. [Reinforcement Learning with External Knowledge by using Logical Neural Networks][2]<a name="ref_2"></a>
3. [Foundations of Reasoning with Uncertainty via Real-valued Logics][3]<a name="ref_3"></a>
4. [Robert S. Engelmore Memorial Award Lecture: The Third AI Summer - Henry Kautz][4]<a name="ref_4"></a>

[1]: https://arxiv.org/abs/2006.13155
[2]: https://arxiv.org/abs/2103.02363
[3]: https://arxiv.org/abs/2008.02429
[4]: https://vimeo.com/389560858
