"""A ``backend`` fixture that runs each test once per installed backend.

Each adapter converts NumPy arrays in and out and wraps the framework's
gradient and layer APIs, so tests are written once.
"""

import importlib.util

import numpy as np
import pytest


class Torch:
    def __init__(self):
        import torch
        import deepordinal.torch as mod

        self.torch, self.mod = torch, mod

    def tensor(self, a):
        return self.torch.as_tensor(np.asarray(a))

    def numpy(self, x):
        return x.detach().numpy()

    def grad(self, fn, *arrays):
        args = [self.tensor(a).requires_grad_() for a in arrays]
        fn(*args).backward()
        return [a.grad.numpy() for a in args]

    def layer(self, input_dim, output_dim):
        return self.mod.OrdinalOutput(input_dim, output_dim)

    def set_raw_thresholds(self, layer, values):
        with self.torch.no_grad():
            layer.raw_thresholds.copy_(self.tensor(values))

    def param_shapes(self, layer):
        return sorted(tuple(p.shape) for p in layer.parameters())

    def layer_grads(self, layer, fn):
        fn(layer).backward()
        return [p.grad.numpy() for p in layer.parameters()]


class TF:
    def __init__(self):
        import tensorflow as tf
        import deepordinal.tf as mod

        self.tf, self.mod = tf, mod

    def tensor(self, a):
        return self.tf.constant(np.asarray(a))

    def numpy(self, x):
        return x.numpy()

    def grad(self, fn, *arrays):
        args = [self.tf.Variable(np.asarray(a)) for a in arrays]
        with self.tf.GradientTape() as tape:
            out = fn(*args)
        return [g.numpy() for g in tape.gradient(out, args)]

    def layer(self, input_dim, output_dim):
        layer = self.mod.OrdinalOutput(output_dim)
        layer.build((None, input_dim))
        return layer

    def set_raw_thresholds(self, layer, values):
        layer.raw_thresholds.assign(np.asarray(values))

    def param_shapes(self, layer):
        return sorted(tuple(w.shape) for w in layer.trainable_weights)

    def layer_grads(self, layer, fn):
        with self.tf.GradientTape() as tape:
            out = fn(layer)
        return [g.numpy() for g in tape.gradient(out, layer.trainable_weights)]


class JAX:
    def __init__(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        import deepordinal.jax as mod

        self.jax, self.jnp, self.nnx, self.mod = jax, jnp, nnx, mod

    def tensor(self, a):
        return self.jnp.asarray(np.asarray(a))

    def numpy(self, x):
        return np.asarray(x)

    def grad(self, fn, *arrays):
        argnums = tuple(range(len(arrays)))
        return [np.asarray(g) for g in self.jax.grad(fn, argnums=argnums)(*map(self.tensor, arrays))]

    def layer(self, input_dim, output_dim):
        return self.mod.OrdinalOutput(input_dim, output_dim, rngs=self.nnx.Rngs(0))

    def set_raw_thresholds(self, layer, values):
        layer.raw_thresholds[...] = self.tensor(values)

    def _params(self, state):
        return self.jax.tree_util.tree_leaves(self.nnx.state(state, self.nnx.Param))

    def param_shapes(self, layer):
        return sorted(tuple(p.shape) for p in self._params(layer))

    def layer_grads(self, layer, fn):
        return [np.asarray(g) for g in self._params(self.nnx.grad(fn)(layer))]


BACKENDS = {"torch": (Torch, "torch"), "tf": (TF, "tensorflow"), "jax": (JAX, "flax")}


@pytest.fixture(params=list(BACKENDS))
def backend(request):
    adapter, requirement = BACKENDS[request.param]
    if importlib.util.find_spec(requirement) is None:
        pytest.skip(f"{requirement} is not installed")
    return adapter()
