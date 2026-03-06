# Agility Forge

A GPU-accelerated finite element framework for modeling incremental forging processes with automatic differentiation capabilities.

## Overview

**Agility Forge** is built on [JAX-FEM](https://github.com/deepmodeling/jax-fem) and provides a comprehensive suite of tools for simulating metal forging operations. The framework combines:

- **GPU Acceleration**: JAX-based implementation for efficient large-scale simulations
- **Plasticity Modeling**: Advanced material models including J2 plasticity with hardening
- **Multi-physics**: Fully-coupled thermo-mechanical analysis
- **Inverse Analysis**: Built-in support for parameter identification and design optimization

## Key Features

- **Constitutive Models**: J2 plasticity with isotropic/kinematic hardening
- **Element Types**: HEX8, QUAD4 with full integration
- **Boundary Conditions**: Automatic handling of large deformation kinematics
- **Mesh Generation**: Integrated with Gmsh for flexible geometry creation
- **Post-processing**: VTK output and visualization tools
- **Time Stepping**: Automatic and manual load stepping strategies

## Installation

### Prerequisites
- Python >= 3.12
- CUDA 12 (for GPU support, optional)
- JAX-FEM main repository

### Directory Setup

Before installation, ensure the following directory structure:

```
jax-fem/
├── applications/
│   └── Agility_Forge/          # Place this repository here
│       └── jax-forge-main/
├── jax_fem_checkpoint/         # Place this folder here
└── ... (other JAX-FEM files)
```

**Step 1**: Place the `Agility_Forge` directory into `jax-fem/applications/`

```bash
cp -r Agility_Forge /path/to/jax-fem/applications/
```

**Step 2**: Place the `jax_fem_checkpoint` directory into the `jax-fem` root directory

```bash
cp -r jax_fem_checkpoint /path/to/jax-fem/
```

### Running the Program

To run the Agility Forge application:

```bash
python -m applications.Agility_Forge.jax_forge
```

## Citation

If you use JAX-FEM, please cite:

```bibtex
@article{xue2023jaxfem,
  title={JAX-FEM: A Differentiable Framework for Finite Element Method},
  author={Xue, Tianju and others},
  journal={arXiv preprint arXiv:2212.00964},
  year={2022}
}
```


## References

- [Original JAX-FEM Repository](https://github.com/deepmodeling/jax-fem)
- [JAX-FEM Paper](https://arxiv.org/abs/2212.00964)
- Simo & Rifai (1990) - Enhanced strain methods



## Support

For issues and questions, please refer to the [JAX-FEM documentation](https://github.com/deepmodeling/jax-fem/tree/main/demos) and cite relevant academic references.
