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
  _common.py           # backend-free numpy helpers (threshold init, argument checks)
tests/
  test_torch.py        # PyTorch tests (skipped if torch not installed)
  test_tf.py           # TF tests (skipped if tensorflow not installed)
  test_jax.py          # JAX tests (skipped if jax/flax not installed)
  test_parity.py       # cross-backend numeric parity (skipped with fewer than two backends)
examples/
  example_torch.ipynb  # PyTorch training loop with ordinal_loss
  example_tf.ipynb     # TF/Keras GradientTape training loop with ordinal_loss
CHANGELOG.md           # Keep a Changelog; breaking changes carry migration notes
pyproject.toml         # build config, version, extras [torch], [tf], and [jax]
```

## Key Architectural Details

- All backends expose the same public API: `OrdinalOutput`, `ordinal_loss`, `ordistic_loss`
- `OrdinalOutput` projects input to a single logit, then produces K class probabilities from K-1 sorted thresholds
- Math: `P(y=k|x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)`, where `sigmoid(t(0) - logit)` is 0 and `sigmoid(t(K) - logit)` is 1
- No backend materializes infinite thresholds: the layer pads the K-1 cumulative probabilities with literal 0 and 1, and the `immediate` construction is the `all` construction with a mask over the two thresholds bounding the target class. Infinite thresholds produce NaN gradients for quadratic penalties
- Loss functions operate on raw logits + thresholds, NOT on probability output from the layer
- Constructors: PyTorch `OrdinalOutput(input_dim, output_dim)`, TF/Keras `OrdinalOutput(output_dim)`, JAX/Flax NNX `OrdinalOutput(input_dim, output_dim, *, rngs)`
- Every backend: `layer.logit(h)` returns the raw logit `(..., 1)` and `layer.interior_thresholds` returns the ordered `(K-1,)` thresholds. Pass both to the losses
- Thresholds are ordered by construction: the trainable parameter is `raw_thresholds` (first threshold, then inverse softplus of each gap) and `interior_thresholds` is a computed property `cumsum([raw[0], softplus(raw[1:])])`
- The projection to the logit has no bias, because the thresholds absorb it. Thresholds initialize at `logit(k/K)` via `deepordinal/_common.py`, shared by all backends
- TF module additionally exports `SortedInitializer` (wraps a Keras initializer, sorts output). `OrdinalOutput` no longer uses it; it stays exported for compatibility

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
