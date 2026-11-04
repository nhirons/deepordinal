# CLAUDE.md

## Project Overview

DeepOrdinal is a Python library for deep ordinal regression with PyTorch, TensorFlow/Keras, and JAX/Flax NNX. It provides an `OrdinalOutput` layer (logit -> ordinal class probabilities via sorted thresholds) and loss functions from Rennie & Srebro (2005).

## File Layout

```
deepordinal/
  __init__.py          # version, __all__
  torch.py             # PyTorch: OrdinalOutput(nn.Module), ordinal_loss, ordistic_loss
  tf.py                # TF/Keras: OrdinalOutput(Layer), SortedInitializer, ordinal_loss, ordistic_loss
  jax.py               # JAX/Flax NNX: OrdinalOutput(nnx.Module), ordinal_loss, ordistic_loss
tests/
  test_torch.py        # PyTorch tests (skipped if torch not installed)
  test_tf.py           # TF tests (skipped if tensorflow not installed)
  test_jax.py          # JAX tests (skipped if jax/flax not installed)
examples/
  example_torch.ipynb  # PyTorch training loop with ordinal_loss
  example_tf.ipynb     # TF/Keras GradientTape training loop with ordinal_loss
pyproject.toml         # build config, version, extras [torch], [tf], and [jax]
```

## Key Architectural Details

- All backends expose the same public API: `OrdinalOutput`, `ordinal_loss`, `ordistic_loss`
- `OrdinalOutput` projects input to a single logit, then produces K class probabilities from K-1 sorted thresholds
- Math: `P(y=k|x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)`, with `t(0) = -inf` and `t(K) = inf` fixed
- Loss functions operate on raw logits + thresholds, NOT on probability output from the layer
- PyTorch `OrdinalOutput(input_dim, output_dim)` — thresholds accessed via `layer.interior_thresholds` (shape `(K-1,)`), logit via `layer.linear(h)`
- TF/Keras `OrdinalOutput(output_dim)` — thresholds shape is `(1, K-1)`, use `layer.interior_thresholds[0]` for loss; logit via `tf.matmul(h, layer.kernel) + layer.bias`
- JAX/Flax NNX `OrdinalOutput(input_dim, output_dim, *, rngs)`: thresholds via `layer.interior_thresholds[...]` (shape `(K-1,)`), logit via `layer.linear(h)`
- JAX `_penalty` clips z at 1 inside quadratic branches so `immediate` construction gradients stay finite at the +/-inf boundary thresholds
- TF module additionally exports `SortedInitializer` (wraps a Keras initializer, sorts output)

## Build & Test

```bash
pip install -e ".[torch,tf,jax]"   # dev install with all backends
pytest -v                       # run all tests
```

- Uses `pyproject.toml` with setuptools (no setup.py)
- Python >=3.10 required (the `[jax]` extra needs 3.11+, since Flax NNX >=0.11 does)
- Base dependency: `numpy`. Backends are optional extras.
- Tests use `pytest.importorskip` to skip backend-specific tests when that backend is not installed

## Conventions

- PyTorch is the primary backend — list it first in docs, examples, and code
- All backends must maintain API parity
- Version is tracked in both `pyproject.toml` and `deepordinal/__init__.py` — keep them in sync
