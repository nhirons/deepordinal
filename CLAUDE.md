# CLAUDE.md

DeepOrdinal: an `OrdinalOutput` layer (features -> one logit -> K ordered class probabilities) and the Rennie & Srebro (2005) ordinal losses, with identical APIs in PyTorch, TensorFlow/Keras, and JAX/Flax NNX.

## Layout

```
deepordinal/
  __init__.py          # __version__ (the only place the version lives)
  _common.py           # backend-free NumPy helpers: threshold init, output_dim check
  torch.py, tf.py, jax.py
tests/
  conftest.py          # `backend` fixture: one adapter per framework, skips if not installed
  reference.py         # independent float64 NumPy reference, written per sample from the paper
  test_deepordinal.py  # every shared behavior, run once per backend against reference.py
  test_tf.py, test_jax.py  # framework-specific behavior only (Keras config/Sequential, jit)
examples/              # one notebook per backend, same data and model
CHANGELOG.md           # Keep a Changelog; breaking changes carry migration notes
```

## Design invariants

- Public API per backend: `OrdinalOutput`, `ordinal_loss`, `ordistic_loss`, `cumulative_link_loss`. Constructors differ only where the framework requires it (`input_dim` in PyTorch and JAX, `rngs` in JAX).
- Each backend file has the same order: `_penalty`, `_nll`, the three losses, `_log1mexp`, `_log_probs`, `OrdinalOutput`. Keep the three files line-for-line comparable.
- Losses take the raw logit (`layer.logit(h)`) and `layer.interior_thresholds`, never probabilities.
- No infinite thresholds anywhere, because they produce NaN gradients. `immediate` is the `all` margin matrix with a mask.
- `_log_probs` is the one place cumulative-logit log-probabilities are computed (softplus + `log1mexp`). The layer's forward pass is `exp(log_probs)`.
- Thresholds are ordered by construction: the parameter is `raw_thresholds` (first threshold, then inverse softplus of each gap), and `interior_thresholds` is a computed property.
- The logit projection has no bias, because the thresholds absorb it.
- Loss dtype follows the logits.

## Build and test

```bash
pip install -e ".[torch,tf,jax]"
pytest
```

- A new shared behavior gets one test in `test_deepordinal.py` through the `backend` adapter, compared with `reference.py` when it is numeric. Only framework-specific behavior goes in `test_tf.py` or `test_jax.py`.
- Tested minimum versions: torch 2.0, tensorflow 2.15, jax 0.6 with flax 0.11. The `[jax]` extra needs Python 3.11+ because Flax 0.11 does.
- Notebooks are committed without outputs. Execute them to verify before committing.

## Conventions

- PyTorch is the primary backend: list it first in docs, examples, and code.
- All backends keep API parity.
