from typing import Tuple

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp

Initializer = jax.nn.initializers


@struct.dataclass
class Variables():
    counter = 0
    layers = []
    normalized_weights = None


class DynamicLinearCombination(nn.Module):
    num_layers: int
    input_shape: Tuple[int]
    weight_dim: int = 1
    weight_normalizing_type: str = 'avg'
    window_size: int = -1
    normalized_before: bool = False
    norm_weights: bool = False
    dropout_rate: int = 0.0
    bias_init: Initializer = nn.initializers.zeros
    scale_init: Initializer = nn.initializers.ones
    use_bias: bool = True
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.shape = (self.num_layers, self.num_layers)
        self.layer_norms = [nn.LayerNorm(dtype=self.dtype, use_bias=self.use_bias,
                                         epsilon=self.epsilon, bias_init=self.bias_init,
                                         scale_init=self.scale_init) for _ in range(self.num_layers)]
        self.init_fn = lambda rng, shape, window_size, weihgt_dim, norm_type: self.weight_normalizing(
            self.weight_mask(shape, window_size), weihgt_dim, norm_type
        )
        self.weights = self.param('dlcl_weights', self.init_fn, self.shape,
                                  self.window_size, self.weight_dim, self.weight_normalizing_type)

        self._init(jnp.ones(shape=(self.input_shape[-1], )))

    def weight_normalizing(self, weights: jnp.ndarray, dim: int, norm_type: str):
        if norm_type == 'avg':
            normed_weights = weights.at[:, :].set(weights / jnp.sum(weights, axis=-1,
                                                                    keepdims=True))
        elif norm_type == 'one_only':
            normed_weights = weights.at[:, :].set(1.)
        else:
            raise ValueError(f'Unknown norm type, {norm_type}')

        normed_weights = normed_weights[..., jnp.newaxis]
        if dim > 1:
            normed_weights = jnp.tile(normed_weights, (1, 1, dim))

        return normed_weights

    def weight_mask(self, shape, window_size):
        mask = jnp.zeros(shape=shape, dtype=self.dtype)
        if window_size == -1:
            for i in range(mask.shape[0]):
                mask = mask.at[i, :i + 1].set(1.)
        else:
            for i in range(mask.shape[0]):
                mask = mask.at[i, max(0, i + 1 - window_size):i + 1].set(1.)

        return mask

    def _init(self, inputs):
        for i in range(self.num_layers):
            self.layer_norms[i](inputs)
        return

    def add_to_memory(self, layer):
        Variables.counter += 1

        if Variables.counter == 1:
            Variables.layers.append(self.layer_norms[0](layer))
            if self.norm_weights:
                mask = jnp.expand_dims((self.weight_mask(self.shape, window_size=self.window_size) == 0), -1)
                weights = jnp.where(mask,
                                    self.weights,
                                    jnp.finfo(self.dtype).min)
                Variables.normalized_weights = jax.nn.softmax(weights, axis=-1)
            return

        if self.normalized_before:
            layer = self.layer_norms[Variables.counter - 1](layer)

        Variables.layers.append(layer)

    def select_weights(self):
        weights = Variables.normalized_weights if Variables.normalized_weights is not None else self.weights
        weights = weights[Variables.counter - 1:, :Variables.counter, :]
        weights = jnp.reshape(weights, (-1, 1, 1, self.weight_dim))
        return weights

    def forward(self, deterministic):
        assert len(Variables.layers) > 0, ('No value added to memroy!')

        weights = self.select_weights()
        layers = jnp.stack(Variables.layers, axis=0)
        lw_product = (layers * weights).sum(0)

        if self.normalized_before:
            if self.dropout_rate > 0.:
                return nn.Dropout(rate=self.dropout_rate)(lw_product, deterministic=deterministic)
            else:
                return lw_product

        if self.dropout_rate > 0.:
            return nn.Dropout(rate=self.dropout_rate)(self.layer_norms[Variables.counter - 1](lw_product),
                                                      deterministic=deterministic)
        else:
            return self.layer_norms[Variables.counter - 1](lw_product)

    def __call__(self, inputs: jnp.ndarray, operation_type: str = 'init'
                 , deterministic: bool = True):

        if operation_type == 'add_to_memory':
            self.add_to_memory(inputs)
        elif operation_type == 'forward':
            return self.forward(deterministic)
        else:
            raise ValueError(
                f'Unknown operation type, {operation_type}. Available operations are [forward, add_to_memory]')