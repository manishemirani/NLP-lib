from typing import Iterable

import jax.numpy as jnp
import jax.random as random
import flax.linen as nn


class ActNorm(nn.Module):
    input_shape: Iterable[int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        scale_fn = lambda rng, dim: self.scale_init_fn(dim, rng)
        bias_fn = lambda rng, scale, dim: self.bias_init_fn(scale, dim, rng)
        self.scale = self.param('scale_weights', scale_fn, self.input_shape)
        self.bias = self.param('bias_weights', bias_fn,
                               self.scale, self.input_shape)

    def scale_init_fn(self, shape, rng):
        init_input = random.normal(rng, shape=shape)
        if len(shape) > 1:
            init_input = self._transpose(init_input, len(shape))
        scale = 1. / jnp.std(init_input.reshape((-1, init_input.shape[-1])), axis=0,
                             dtype=self.dtype)
        return scale

    def bias_init_fn(self, scale, shape, rng):
        init_input = random.normal(rng, shape=shape)
        if len(shape) > 1:
            init_input = self._transpose(init_input, len(shape))
        bias = jnp.multiply(-scale, jnp.mean(init_input.reshape(-1, init_input.shape[-1]), axis=0))
        return bias

    def _transpose(self, inputs, dim):
        permutations = [i for i in range(dim)]
        a, b = permutations[1], permutations[-1]
        permutations[1], permutations[-1] = b, a
        return jnp.transpose(inputs, permutations)

    def __call__(self, x: jnp.ndarray, operation: str = 'inverse') -> jnp.ndarray:

        if operation == 'forward':
            if x.ndim > 1:
                x = self._transpose(x, x.ndim)
            x = jnp.multiply(self.scale, x) + self.bias
            if x.ndim > 1:
                x = self._transpose(x, x.ndim)
            return x
        if operation == 'inverse':
            if x.ndim > 1:
                x = self._transpose(x, x.ndim)
            x = jnp.multiply(x - self.bias, 1. / self.scale)
            if x.ndim > 1:
                x = self._transpose(x, x.ndim)
            return x
        else:
            raise ValueError(f'Unknown operation type {operation}. Available operations are [forward, inverse]')
