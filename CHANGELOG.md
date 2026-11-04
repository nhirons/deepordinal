# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-04

### Added
- **A JAX backend**, `deepordinal.jax`, with the same API as the PyTorch backend. `OrdinalOutput` is a Flax NNX module. The losses use only `jax.numpy`, so any JAX framework can call them. Install with `pip install "deepordinal[jax]"` on Python 3.11 or later.
- **`OrdinalOutput.logit(x)`** in every backend, returning the raw logit to pass to the losses.
- **`cumulative_link_loss`** in every backend: the negative log-likelihood of the model `OrdinalOutput` represents. The Rennie & Srebro losses train the logit and thresholds, but none of them is the likelihood of the probabilities the layer outputs.
- **`OrdinalOutput.log_probs(x)`** in every backend, computed from softplus and `log1mexp` terms so it stays accurate for large `|logit|`.

### Changed
- **Thresholds are ordered by construction.** The trainable parameter is `raw_thresholds`: the first threshold, then the inverse softplus of each gap. `interior_thresholds` is a computed, differentiable property. Unconstrained thresholds could cross during training, and the layer then returned negative probabilities.
- **The projection to the logit has no bias.** A bias shifts the logit and every threshold by the same amount, so it duplicated the thresholds and gave the optimizer a flat direction.
- **Thresholds initialize at `logit(k/K)`** instead of sorted Glorot-uniform draws, so a zero logit gives every class probability `1/K`.
- **TF `interior_thresholds` has shape `(K-1,)`**, matching the other backends.
- **`OrdinalOutput` raises `ValueError` for `output_dim < 2`.**
- **TF losses compute in the dtype of the logits** instead of casting everything to float32, matching the other backends.
- **The `[tf]` extra requires `tensorflow>=2.15`**, the oldest version the test suite covers. The previous `>=2.0` floor was never tested.
- **The forward pass returns `exp(log_probs(x))`** instead of a difference of sigmoids. In float32, the difference rounded every class but one to exactly 0 once `|logit|` passed about 17. The two agree to float precision elsewhere.

### Removed
- **`deepordinal.tf.SortedInitializer`.** It existed to initialize sorted thresholds, and the layer no longer has a sorted-threshold variable to initialize. It cannot initialize `raw_thresholds` either, because that vector stores a first threshold and inverse-softplus gaps, not sorted values. Sorting an initializer's output never kept parameters ordered during training anyway. The class was TF-only and undocumented. It never appeared in a saved layer config, so removing it breaks no saved model. If you used it elsewhere, sort the output of any Keras initializer with `tf.sort(init(shape), axis=-1)`.

### Fixed
- **`from deepordinal import *`** no longer imports every backend, which failed unless all three were installed.
- **NaN gradients in `ordinal_loss(..., construction="immediate")`** for targets in the first or last class. PyTorch failed with `smooth_hinge` and `modified_least_squares`, and TensorFlow failed with all four penalties. No backend builds infinite thresholds any more.

### Migrating from 0.2

| 0.2 | 0.3 |
| --- | --- |
| PyTorch `layer.linear(h)` | `layer.logit(h)` (`layer.linear(h)` still works) |
| TF `tf.matmul(h, layer.kernel) + layer.bias` | `layer.logit(h)` |
| TF `layer.interior_thresholds[0]` | `layer.interior_thresholds` |
| Optimizing `layer.interior_thresholds` directly | Optimize `layer.raw_thresholds`. `model.parameters()` and `trainable_variables` already include it |

Checkpoints from 0.2 do not load. The parameter names and the meaning of the stored threshold values both changed.
