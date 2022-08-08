[![Build Status](https://github.com/IBM/LNN/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/IBM/LNN/actions/workflows/build.yml?query=branch%3Amaster)
[![License](https://img.shields.io/badge/license-Apache%202.0-blueviolet)](https://github.com/IBM/LNN/blob/master/LICENSE)
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
1. Install [GraphViz](https://www.graphviz.org/download/) and gmp (libgmp3-dev)
      <details>
      <summary>Tips for installing Graphviz</summary>
      <div>
      If you experience a "graphviz/cgraph.h" file not found error, try the following:
      
      ```bash
      # Using Conda
      conda install -c anaconda graphviz
      ```
      Alternatively you can install the OS specific versions as follows:
      ```bash
      # Ubuntu and Debian
      sudo apt-get install graphviz graphviz-dev
      ```
      ```bash
      # MacOS
      brew install graphviz
      ```
      ```bash
      # Windows
      sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
      ```
      </div></details>

      <details>
      <summary>Tips for installing gmp</summary>
      <div>

      If you experience a "gmp.h" file not found error, try the following:
      ```bash
      # MacOS
      brew install gmp
      env "CFLAGS=-I/usr/local/include -L/usr/local/lib" pip install pycddlib
      ```
      ```bash
      # Windows
      sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
      sudo -H apt-get install libgmp-dev python3-dev
      ```
      </div></details>

2. Make sure that the python version you use in line with our [setup](https://github.com/IBM/LNN/blob/master/setup.py) file, using a fresh environment is always a good idea:
    ```commandline
    conda create -n lnn python=3.9 -y
    conda activate lnn
    ```
3. Install the LNN as a library:
    ```commandline
    pip install git+https://github.com/IBM/LNN.git
    ```

## Contribution
Contributions to the LNN codebase are welcome!

Please have a look at the [contribution guide](https://github.com/IBM/LNN/blob/master/CONTRIBUTING.md) for more information on how to set up the LNN for contributing and how to follow our development standards.

## Documentation
| [Read the Docs][Docs] | [Academic Papers][Papers]	| [Educational Resources][Education] | [Neuro-Symbolic AI][Neuro-Symbolic AI] | [API Overview][API] | [Python Module][Module] |
|:-----------------------:|:---------------------------:|:-----------------:|:----------:|:-------:|:-------:|
| [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/doc.png alt="Docs" width="60"/>][Docs] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/academic.png alt="Academic Papers" width="60"/>][Papers] |  [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/help.png alt="Getting Started" width="60"/>][Education] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/nsai.png alt="Neuro-Symbolic AI" width="60"/>][Neuro-Symbolic AI] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/api.png alt="API" width="60"/>][API] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/python.png alt="Python Module" width="60"/>][Module] |

## Citation
If you use Logical Neural Networks for research, please consider citing the
reference paper:
```raw
@article{riegel2020logical,
  title={Logical neural networks},
  author={Riegel, Ryan and Gray, Alexander and Luus, Francois and Khan, Naweed and Makondo, Ndivhuwo and Akhalwaya, Ismail Yunus and Qian, Haifeng and Fagin, Ronald and Barahona, Francisco and Sharma, Udit and others},
  journal={arXiv preprint arXiv:2006.13155},
  year={2020}
}
```

[Docs]: https://ibm.github.io/LNN/introduction.html
[Papers]: https://ibm.github.io/LNN/papers.html
[Education]: https://ibm.github.io/LNN/education/education.html
[API]: https://ibm.github.io/LNN/usage.html
[Module]: https://ibm.github.io/LNN/lnn/LNN.html
[Neuro-Symbolic AI]: https://research.ibm.com/teams/neuro-symbolic-ai
