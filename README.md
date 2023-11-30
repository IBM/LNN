[![Build Status](https://github.com/IBM/LNN/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/IBM/LNN/actions/workflows/build.yml?query=branch%3Amaster)
[![PRs Welcome](https://img.shields.io/static/v1?label=PRs&message=welcome&color=green&logo=git&logoColor=white)](https://makeapullrequest.com)
[![License](https://img.shields.io/static/v1?label=License&message=Apache%202.0&color=blueviolet&logo=linux&logoColor=white)](https://github.com/IBM/LNN/blob/master/LICENSE)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5926/badge)](https://bestpractices.coreinfrastructure.org/projects/5926)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Logical Neural Networks
LNNs are a novel `Neuro = Symbolic` framework designed to seamlessly provide key
properties of both neural nets (learning) and symbolic logic (knowledge and reasoning).

- Every neuron has a meaning as a component of a formula in a weighted
  real-valued logic, yielding a highly interpretable disentangled representation.
- Inference is omnidirectional rather than focused on predefined target
  variables, and corresponds to logical reasoning, including classical
  first-order logic theorem proving as a special case.
- The model is end-to-end differentiable, and learning minimizes a novel loss
  function capturing logical contradiction, yielding resilience to inconsistent
  knowledge.
- It also enables the open-world assumption by maintaining bounds on truth values
  which can have probabilistic semantics, yielding resilience to incomplete
  knowledge.

## Quickstart
To install the LNN:

 Make sure that the python version you use in line with our [setup](https://github.com/IBM/LNN/blob/master/setup.py) file. Using a fresh environment is always a good idea:
```commandline
conda create -n lnn python=3.9 -y
conda activate lnn
pip install git+https://github.com/IBM/LNN
```

## Documentation
| [Read the Docs][Docs] | [Academic Papers][Papers]	| [Educational Resources][Education] | [Neuro-Symbolic AI][Neuro-Symbolic AI] | [API Overview][API] | [Python Module][Module] |
|:-----------------------:|:---------------------------:|:-----------------:|:----------:|:-------:|:-------:|


[Docs]: https://ibm.github.io/LNN/introduction.html
[Papers]: https://ibm.github.io/LNN/papers.html
[Education]: https://ibm.github.io/LNN/education/education.html
[API]: https://ibm.github.io/LNN/usage.html
[Module]: https://ibm.github.io/LNN/lnn/LNN.html
[Neuro-Symbolic AI]: https://research.ibm.com/teams/neuro-symbolic-ai
