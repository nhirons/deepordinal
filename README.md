# DeepOrdinal

[![PyPI version](https://img.shields.io/pypi/v/deepordinal)](https://pypi.org/project/deepordinal/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Ordinal regression for PyTorch, TensorFlow/Keras, and JAX/Flax: an `OrdinalOutput` layer that turns features into K ordered class probabilities, and the loss functions of [Rennie & Srebro (2005)](https://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf) to train it. The three backends share one API.

## Install

```bash
pip install "deepordinal[torch]"  # PyTorch
pip install "deepordinal[tf]"     # TensorFlow/Keras
pip install "deepordinal[jax]"    # JAX/Flax NNX, Python 3.11+
```

## Quick start

Every backend follows the same pattern. Train on the raw logit `head.logit(h)` and the thresholds `head.interior_thresholds`, and read probabilities from `head(h)`.

### PyTorch

```python
import torch
import torch.nn as nn
from deepordinal.torch import OrdinalOutput, ordinal_loss

features = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
head = OrdinalOutput(input_dim=16, output_dim=4)
optimizer = torch.optim.Adam([*features.parameters(), *head.parameters()])

loss = ordinal_loss(head.logit(features(x)), y, head.interior_thresholds)
optimizer.zero_grad()
loss.backward()
optimizer.step()

probs = head(features(x))  # (batch, 4)
```

### TensorFlow/Keras

```python
import tensorflow as tf
from deepordinal.tf import OrdinalOutput, ordinal_loss

features = tf.keras.layers.Dense(16, activation="relu")
head = OrdinalOutput(output_dim=4)
optimizer = tf.keras.optimizers.Adam()

with tf.GradientTape() as tape:
    loss = ordinal_loss(head.logit(features(x)), y, head.interior_thresholds)
params = features.trainable_variables + head.trainable_variables
optimizer.apply_gradients(zip(tape.gradient(loss, params), params))

probs = head(features(x))  # (batch, 4)
```

### JAX/Flax NNX

```python
import jax
import optax
from flax import nnx
from deepordinal.jax import OrdinalOutput, ordinal_loss

class Model(nnx.Module):
    def __init__(self, rngs):
        self.features = nnx.Sequential(nnx.Linear(8, 16, rngs=rngs), jax.nn.relu)
        self.head = OrdinalOutput(input_dim=16, output_dim=4, rngs=rngs)

    def __call__(self, x):
        return self.head(self.features(x))

model = Model(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        return ordinal_loss(model.head.logit(model.features(x)), y, model.head.interior_thresholds)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

loss = train_step(model, optimizer, x, y)
probs = model(x)  # (batch, 4)
```

## The layer

`OrdinalOutput` projects its input to one logit `z` and compares it with K-1 ordered thresholds `t`:

```
P(y = k | x) = sigmoid(t[k] - z) - sigmoid(t[k-1] - z)
```

The first term is 1 for the last class, and the second is 0 for the first class.

- **The thresholds are ordered by construction.** The trainable `raw_thresholds` hold the first threshold and the inverse softplus of each gap, so no training step can reorder them or produce a negative probability.
- **The projection has no bias.** Shifting `z` and every threshold together leaves the model unchanged, so the thresholds absorb it.
- **The thresholds start at `logit(k/K)`**, so a zero logit gives every class probability `1/K`.

| Constructor | |
| --- | --- |
| PyTorch | `OrdinalOutput(input_dim, output_dim)` |
| TensorFlow/Keras | `OrdinalOutput(output_dim)` |
| JAX/Flax NNX | `OrdinalOutput(input_dim, output_dim, rngs=rngs)` |

| Member | Shape | |
| --- | --- | --- |
| `layer(h)` | `(..., K)` | class probabilities |
| `layer.log_probs(h)` | `(..., K)` | log class probabilities, accurate at any logit |
| `layer.logit(h)` | `(..., 1)` | raw logit, the first argument to every loss |
| `layer.interior_thresholds` | `(K-1,)` | ordered thresholds, the third argument to `ordinal_loss` and `cumulative_link_loss` |

## Losses

Every loss takes the raw logit (`(batch,)` or `(batch, 1)`), integer targets in `[0, K)`, and returns the mean over the batch.

| Loss | What it fits | Use it when |
| --- | --- | --- |
| `ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic")` | A margin penalty for each threshold on the wrong side of the logit (paper eqs 12 and 13) | You care about the predicted label. The default all-threshold logistic loss performed best in the paper. |
| `cumulative_link_loss(logits, targets, thresholds)` | The likelihood of the probabilities `OrdinalOutput` outputs | You use the probabilities themselves, for example for calibration or expected-cost decisions. |
| `ordistic_loss(logits, targets, means, log_priors=None)` | The likelihood of the paper's ordistic model, with one mean per class (section 4) | You want the paper's probabilistic alternative and supply the `(K,)` class means yourself. The paper fixes the first mean at -1 and the last at 1. |

### `ordinal_loss` options

- `construction="all"` sums a penalty over every threshold, so the loss grows with the distance between the predicted and true labels (eq 13).
- `construction="immediate"` penalizes only the two thresholds that bound the target class (eq 12).

| `penalty` | `f(m)` for margin `m` | Paper |
| --- | --- | --- |
| `"logistic"` | `log(1 + exp(-m))` | eq 9 |
| `"hinge"` | `max(0, 1 - m)` | eq 5 |
| `"smooth_hinge"` | `0` if `m >= 1`, `(1 - m)^2 / 2` if `0 < m < 1`, `0.5 - m` otherwise | eq 6 |
| `"modified_least_squares"` | `0` if `m >= 1`, `(1 - m)^2` otherwise | eq 7 |

### `cumulative_link_loss` numerics

The log-probabilities come from softplus and `log1mexp` terms, never from a difference of sigmoids. In float32, that difference rounds to exactly 0 once `|logit|` passes about 17. `layer.log_probs` and `layer(h)` use the same computation.

For interior classes, the loss equals the immediate-threshold logistic loss plus `-log(1 - exp(-gap))`, where `gap` is the width of the target class's interval. For the first and last classes, the two losses are equal.

## Examples

Each notebook trains the same model on the same synthetic data:

- [`examples/example_torch.ipynb`](examples/example_torch.ipynb): PyTorch
- [`examples/example_tf.ipynb`](examples/example_tf.ipynb): TensorFlow/Keras
- [`examples/example_jax.ipynb`](examples/example_jax.ipynb): JAX/Flax NNX (also needs `optax`)

## Development

```bash
pip install -e ".[torch,tf,jax]"
pytest
```

Each test in `tests/test_deepordinal.py` runs once per installed backend and checks the result against an independent float64 NumPy reference in `tests/reference.py`.

## Citation

> Rennie, J. D. M. & Srebro, N. (2005). Loss Functions for Preference Levels: Regression with Discrete Ordered Labels. *Proceedings of the IJCAI Multidisciplinary Workshop on Advances in Preference Handling*.
