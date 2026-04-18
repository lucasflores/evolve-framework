# Governance — evolve/agents.md

## evolve/ Package Rules

- **No ML framework imports** in core modules (PyTorch, JAX, TensorFlow)
- **NumPy only** for numeric operations in `representation/` and `core/`
- All public classes/functions MUST have docstrings
- Activation functions registered in `representation/network.py::ACTIVATIONS` dict
- Decoders registered in `registry/decoders.py` via lazy-init factory pattern
- New decoder files go in `representation/` alongside existing `decoder.py`
- Genomes are immutable frozen dataclasses
- Networks (`NEATNetwork`, `NumpyNetwork`) are callable via `__call__`
- `Decoder[G, P]` protocol in `representation/phenotype.py` is the interface contract
