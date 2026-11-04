# DeepOrdinal

[![PyPI version](https://img.shields.io/pypi/v/deepordinal)](https://pypi.org/project/deepordinal/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Ordinal output layers and loss functions for PyTorch, TensorFlow/Keras, and JAX/Flax, based on [Rennie & Srebro (2005)](https://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf).

DeepOrdinal provides an `OrdinalOutput` layer that converts a learned logit into ordinal class probabilities via ordered thresholds, plus loss functions designed specifically for ordinal regression.

## Installation

```bash
pip install deepordinal
```

Install with a specific backend:

```bash
pip install "deepordinal[torch]"  # PyTorch
pip install "deepordinal[tf]"     # TensorFlow/Keras
pip install "deepordinal[jax]"    # JAX/Flax NNX (Python 3.11+)
```

For development:

```bash
pip install -e ".[torch,tf,jax]"
```

## Quick Start

### PyTorch

```python
import torch
import torch.nn as nn
from deepordinal.torch import OrdinalOutput, ordinal_loss

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    OrdinalOutput(input_dim=16, output_dim=4),
)
ordinal_layer = model[2]
optimizer = torch.optim.Adam(model.parameters())

# Training step
h = model[1](model[0](x_batch))              # hidden features
logits = ordinal_layer.logit(h)               # raw logit
thresholds = ordinal_layer.interior_thresholds
loss = ordinal_loss(logits, y_batch, thresholds)

optimizer.zero_grad()
loss.backward()
optimizer.step()

# Inference
probs = model(x_batch)        # (batch, K) class probabilities
preds = probs.argmax(dim=1)
```

### TensorFlow/Keras

```python
import tensorflow as tf
from deepordinal.tf import OrdinalOutput, ordinal_loss

dense = tf.keras.layers.Dense(16, activation="relu")
ordinal = OrdinalOutput(output_dim=4)

# Training step
with tf.GradientTape() as tape:
    h = dense(x_batch)
    logits = ordinal.logit(h)
    thresholds = ordinal.interior_thresholds
    loss = ordinal_loss(logits, y_batch, thresholds)
grads = tape.gradient(loss, dense.trainable_variables + ordinal.trainable_variables)

# Inference
probs = ordinal(dense(x_batch))  # (batch, K) class probabilities
preds = tf.argmax(probs, axis=1)
```

### JAX/Flax NNX

```python
import jax
import optax
from flax import nnx
from deepordinal.jax import OrdinalOutput, ordinal_loss

class Model(nnx.Module):
    def __init__(self, rngs):
        self.hidden = nnx.Linear(8, 16, rngs=rngs)
        self.ordinal = OrdinalOutput(input_dim=16, output_dim=4, rngs=rngs)

    def features(self, x):
        return jax.nn.relu(self.hidden(x))

    def __call__(self, x):
        return self.ordinal(self.features(x))

model = Model(nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        logits = model.ordinal.logit(model.features(x))  # raw logit
        thresholds = model.ordinal.interior_thresholds
        return ordinal_loss(logits, y, thresholds)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

loss = train_step(model, optimizer, x_batch, y_batch)

# Inference
probs = model(x_batch)  # (batch, K) class probabilities
preds = probs.argmax(axis=1)
```

## API

### `OrdinalOutput` Layer

Projects an input down to a single logit, with no bias, and converts it into K class probabilities using K-1 learned, ordered thresholds:

```
P(y = k | x) = sigmoid(t(k+1) - logit) - sigmoid(t(k) - logit)
```

where `sigmoid(t(0) - logit)` is 0 and `sigmoid(t(K) - logit)` is 1.

- **Ordered by construction.** The layer stores `raw_thresholds`, the first threshold followed by the inverse softplus of each gap. `interior_thresholds` computes `cumsum([raw[0], softplus(raw[1:])])`, so training can never reorder the thresholds or produce negative probabilities.
- **No bias.** Adding a constant to the logit and to every threshold gives the same model, so the thresholds absorb it.
- **Uniform start.** Thresholds initialize at `logit(k/K)`, so a zero logit assigns every class probability `1/K`.

| | PyTorch | TensorFlow/Keras | JAX/Flax NNX |
|---|---|---|---|
| Import | `from deepordinal.torch import OrdinalOutput` | `from deepordinal.tf import OrdinalOutput` | `from deepordinal.jax import OrdinalOutput` |
| Constructor | `OrdinalOutput(input_dim, output_dim)` | `OrdinalOutput(output_dim)` | `OrdinalOutput(input_dim, output_dim, rngs=rngs)` |
| Logit | `layer.logit(h)` | `layer.logit(h)` | `layer.logit(h)` |
| Log-probabilities | `layer.log_probs(h)` | `layer.log_probs(h)` | `layer.log_probs(h)` |
| Thresholds | `layer.interior_thresholds`, shape `(K-1,)` | `layer.interior_thresholds`, shape `(K-1,)` | `layer.interior_thresholds`, shape `(K-1,)` |
| Trainable parameters | `linear.weight`, `raw_thresholds` | `kernel`, `raw_thresholds` | `linear.kernel`, `raw_thresholds` |

### `ordinal_loss`

Threshold-based ordinal loss from Rennie & Srebro (2005). Operates on raw logits and thresholds rather than probability output.

```python
ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic")
```

**Parameters:**

- **logits** — `(batch,)` or `(batch, 1)`, raw predictor output
- **targets** — `(batch,)`, integer labels in `[0, K)`
- **thresholds** — `(K-1,)`, sorted interior thresholds
- **construction** — `"all"` (default) or `"immediate"`
- **penalty** — `"logistic"` (default), `"hinge"`, `"smooth_hinge"`, or `"modified_least_squares"`

**Returns:** scalar mean loss over the batch.

#### Constructions

- **All-threshold** (default, eq 13) — penalizes violations of every threshold, weighted by direction. Bounds mean absolute error. Best performer in the paper's experiments.
- **Immediate-threshold** (eq 12) — only penalizes violations of the two thresholds bounding the correct class segment.

#### Penalty Functions

| Name | Formula | Reference |
|---|---|---|
| `"logistic"` | `log(1 + exp(-z))` | eq 9 |
| `"hinge"` | `max(0, 1-z)` | eq 5 |
| `"smooth_hinge"` | `0` if z>=1, `(1-z)^2/2` if 0<z<1, `0.5-z` if z<=0 | eq 6 |
| `"modified_least_squares"` | `0` if z>=1, `(1-z)^2` if z<1 | eq 7 |

The paper recommends **all-threshold + logistic** as the best-performing combination (the default).

### `ordistic_loss`

Probabilistic generalization of logistic regression to K-class ordinal problems (Section 4).

```python
ordistic_loss(logits, targets, means, log_priors=None)
```

**Parameters:**

- **logits** — `(batch,)` or `(batch, 1)`, raw predictor output
- **targets** — `(batch,)`, integer labels in `[0, K)`
- **means** — `(K,)`, class means (convention: mu_1=-1, mu_K=1; interior means learned)
- **log_priors** — `(K,)` or `None`, optional log-prior terms

**Returns:** scalar mean negative log-likelihood over the batch.

### `cumulative_link_loss`

Negative log-likelihood of the model `OrdinalOutput` represents (the cumulative logit, proportional-odds, or ordered logit model). Train with this loss when you use the layer's probabilities directly, for example for calibrated predictions or expected-cost decisions.

```python
cumulative_link_loss(logits, targets, thresholds)
```

**Parameters:**

- **logits**: `(batch,)` or `(batch, 1)`, raw predictor output
- **targets**: `(batch,)`, integer labels in `[0, K)`
- **thresholds**: `(K-1,)`, strictly increasing interior thresholds, such as `layer.interior_thresholds`

**Returns:** scalar mean negative log-likelihood over the batch.

The log-probabilities come from softplus and `log1mexp` terms instead of a difference of sigmoids. A difference of sigmoids rounds to exactly 0 in float32 once `|logit|` passes about 17. `layer.log_probs(h)` exposes the same computation, and the layer's forward pass returns `exp(layer.log_probs(h))`.

For interior classes, the loss equals the immediate-threshold logistic loss plus `-log(1 - exp(-gap))`, where `gap` is the width of the target class's interval. That extra term penalizes narrow intervals. For the first and last classes, the two losses are equal.

#### Choosing a loss

- `ordinal_loss` (all-threshold, logistic) is the paper's best performer for minimizing absolute error between predicted and true labels.
- `cumulative_link_loss` fits the probabilities the layer outputs. Choose it when the probabilities themselves matter.

## Examples

Complete training loops with synthetic ordinal data:

- [`examples/example_torch.ipynb`](examples/example_torch.ipynb) — PyTorch
- [`examples/example_tf.ipynb`](examples/example_tf.ipynb) — TensorFlow/Keras

## Testing

```bash
pip install -e ".[torch,tf,jax]"
pytest -v
```

## License

MIT

## Citation

> Rennie, J. D. M. & Srebro, N. (2005). Loss Functions for Preference Levels: Regression with Discrete Ordered Labels. *Proceedings of the IJCAI Multidisciplinary Workshop on Advances in Preference Handling*.
