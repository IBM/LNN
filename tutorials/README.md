<h1 align="center" style='text-align:center;color:#0f62fe'> IBM Neuro-Symbolic AI Essentials</h1>

# Logical Neural Networks

Logical Neural Networks are a `Neuro = Symbolic` framework designed to provide both the properties of neural networks (learning) and symbolic logic (reasoning). LNNs achieve this by extending the fundamental theory of real-valued logic to include parameters which can be mathematically incorporated and implemented using standard machine learning tools. _Knowledge_ is directly modeled as a neural network, achieving a 1-to-1 mapping between human-interpretable formulae (in the form of logical expressions) and the underlying neural architecture representation. _Data_ in the form of facts or bounds on beliefs, imposes constraints under which the model reasons (both classically and under uncertainties). Learning with these parameters allows the LNN to operate under noisy scenarios, using standard backpropagation and gradient descent techniques to adjust incomplete or inconsistent knowledge to appropriately model the supplied data. Above all else, LNNs are interpretable _per-neuron_, allowing the system to be inspected, analysed and questioned as to how it has come to believe any inferred truth.


## Introduction 
Classical AI targets problems within the knowledge representation, reasoning and planning domains. A large portion of the field is based on rigorous mathematical and logical abstractions that rely on symbolic representations to make generalisable assertions that are both compositionally-interpretable and meaningful (albeit brittle). Statistical AI, i.e. robust pattern matching, is successful in solving simple real-world problems by demanding vast amounts of data and compute to capture complex correlations between observable instances of a domain. Contrastingly, Symbolic AI allows us to represent knowledge as human-readable symbols that can be manipulated and consistently reasoned over, whereas Statistical AI learns to adapt to problems in complex environments. Combining both strategies into a hybrid Neuro-Symbolic AI framework simultaneously allows us to embed knowledge and reasoning into computer programs that learn in a data-centric manner, while maintaining consistent and interpretable beliefs.

In Neuro-Symbolic AI, we treat logic as the [lingua franca](https://en.wikipedia.org/wiki/Lingua_franca), i.e., a base language that is unambiguous in its semantic interpretation and a mechanism of connecting discrete (programmatic) components to one another in an end-to-end manner. 
The goals of using symbolic logic as this vehicle includes: (_i._) learning from less data than is needed by Statistical AI approaches; (_ii._) expressing complex theory that generalises to new domains; (_iii._) maintaining human-interpretable machine systems with a tighter coupling of having humans-in-the-loop. It is imperative that NSAI systems reason consistently and employ procedures that preserve trustworthiness when deployed outside of training environments.

## About Tutorials
These tutorials will introduce you to the LNN API, demonstrating fundamental aspects of logical reasoning and learning along with simple example problems. The tutorials will be separated into different modules, using a set of Jupyter notebooks as an interactive environment to help demonstrate the basic concepts. 

The following tutorials will be demonstrated:

<h3> Chapter 1 - Reasoning </h3>

|  | Notebook | Description |
| :---:  | :--- | :--- |
| 0 | Propositional Logic         | Reasoning in its simplest form |
| 1 | Propositional Bounds        | Extended representation of facts to bounds |
| 2 | Propositional Learning      | Introduce concept of learning via adjustment of weights |
| 3 | FOL Upward Inference        | First-Order Logic (FOL) reasoning to evaluate the truth of operators |
| 4 | FOL Downward Inference          | Inverse FOL reasoning to evaluate the truth of operands |
| 5 | Quantifiers                 | Expressing quantities that satisfy a select property |
| 6 | Contradictions | Extending the class of classical truth values |
| 7 | Real-valued Logics          | Operating under a continuous domain of truths |
| 8 | Bounds on Beliefs           | Introducing bounds to restrict truth values between a range of possibilities | 
| 9 | Grounding Management        | Propagating propositionalised data to connected nodes | 
| 10 | FOL Bindings  | Restricting variables to specific instances for which a property should apply | 


## Learning Outcomes
The learning material will guide you on how to use LNNs to solve problems that can be represented logically. Throughout the course, you will develop the following skills that will jumpstart your journey within Neuro-Symbolic AI:

- Solve problems using logic
- Represent ontological statements derived from natural language in knowledge bases (KB).
- Use the LNN API to construct solutions for KB question answering (QA).

## NSAI Badge
IBM offers an [NSAI Foundational Badge](https://www.credly.com/org/ibm/badge/neuro-symbolic-ai-essentials) to certify your completion of the content presented in our [NSAI Workshops and Summer Schools](https://ibm.github.io/neuro-symbolic-ai/events). We encourage you to go through the [badge content](./badges/README.md) to understand what the state of the art is in this field and contribute to the many streams of ongoing work. Once you catch up on the recorded sessions and complete the LNN coursework, you'll be issued a badge to add to your growing skills portfolio 🦾


### Prerequisite
In order to complete these tutorials, the only requirement is that you have:
1. Basic python programming skills.
2. Elementary knowledge in mathematics and logic.

If the terms _"[boolean logic](https://en.wikipedia.org/wiki/Boolean_algebra)"_ and _"[truth tables](https://en.wikipedia.org/wiki/Truth_table)"_ do not have any significance to you, then we suggest that you start by following our [elementary material](https://ibm.box.com/s/c0sq8d7tc01tw1c7xq4k8oedajto84n8) recommendations.


### Getting Started 

In this section, we will guide you on how to get the interactive notebooks working on your local machine.

1. [Fork the LNN repository](https://github.com/IBM/LNN/fork).

2. [Install the LNN as a library](https://github.com/IBM/LNN/#quickstart).
> We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to manage your virtual environments.

3. Run the Notebooks

```bash
cd LNN/tutorials/
pip install jupyterlab
jupyter lab
```

You should be able to see the following window open in your browser:

![Example of browser page](./Chapter%201%20-%20Reasoning/img/example.png)

You will now be able to interactively run the notebooks and go through the material.

### Earning the Badges

As you complete the tutorial material, there is an opportunity to obtain IBM approved badges in Neuro-Symbolic AI. Have a look at the [badges readme](./badges/README.md) for more information. Each badge will build on the next, testing your skills as a proficient NSAI developer in different experiments. 

```python
Implies("Correct Experiment Solution", "IBM Issued Badge")
```

Happy Coding 🍀
