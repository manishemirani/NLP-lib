from typing import Iterable

import jax.numpy as jnp
import jax.random as random
import flax.linen as nn


class ActNorm(nn.Module):
    input_shape: Iterable[int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        scale_fn = lambda rng: self._scale_init_fn(rng)
        bias_fn = lambda rng, scale: self._bias_init_fn(scale, rng)
        self.scale = self.param('scale_weights', scale_fn)
        self.bias = self.param('bias_weights', bias_fn, self.scale)

    def _scale_init_fn(self, rng):
        init_input = random.normal(rng, shape=self.input_shape)
        if init_input.ndim > 1:
            init_input = self._transpose(init_input)
        scale = 1. / jnp.std(init_input.reshape((-1, init_input.shape[-1])), axis=0,
                             dtype=self.dtype)
        return scale

    def _bias_init_fn(self, scale, rng):
        init_input = random.normal(rng, shape=self.input_shape)
        if init_input.ndim > 1:
            init_input = self._transpose(init_input)
        bias = jnp.multiply(-scale, jnp.mean(init_input.reshape(-1, init_input.shape[-1]), axis=0))
        return bias

    def _transpose(self, inputs):
        permutations = [i for i in range(inputs.ndim)]
        a, b = permutations[1], permutations[-1]
        permutations[1], permutations[-1] = b, a
        return jnp.transpose(inputs, permutations)

    def __call__(self, x: jnp.ndarray, operation: str = 'inverse') -> jnp.ndarray:

        if operation == 'forward':
            if x.ndim > 1:
                x = self._transpose(x)
            x = jnp.multiply(self.scale, x) + self.bias
            if x.ndim > 1:
                x = self._transpose(x)
            return x
        if operation == 'inverse':
            if x.ndim > 1:
                x = self._transpose(x)
            x = jnp.multiply(x - self.bias, 1. / self.scale)
            if x.ndim > 1:
                x = self._transpose(x)
            return x
        else:
            raise ValueError(f'Unknown operation type {operation}. Available operations are [forward, inverse]')