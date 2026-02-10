# `luno` - Linearized Uncertainty for Neural Operators

This repository contains the main algorithm of the paper "Linearization Turns Neural Operators into Function-Valued Gaussian Processes" by Magnani et al. (2025).

## Description

`luno` is a Python package that implements linearized uncertainty quantification for neural operators. It provides tools for:
- Fourier Neural Operators with uncertainty quantification
- Jacobian computations for Fourier Neural Operators
- Covariance structures for function-valued Gaussian processes

## Installation

The package can be installed via pip:

```bash
pip install git+https://github.com/MethodsOfMachineLearning/luno.git
```

For development installation with all dependencies:

```bash
pip install -e ".[dev]"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{magnani2025linearizationturnsneuraloperators,
      title={Linearization Turns Neural Operators into Function-Valued Gaussian Processes}, 
      author={Emilia Magnani and Marvin Pförtner and Tobias Weber and Philipp Hennig},
      year={2025},
      eprint={2406.05072},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.05072}
}
```

## Requirements

- Python >= 3.10
- NumPy >= 1.21.2
- JAX >= 0.4.20
- [linox](https://github.com/2bys/linox) (from GitHub)

Optional dependencies for development and testing are available in the `dev` extra.
