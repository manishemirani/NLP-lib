from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn



class Highway(nn.Module):
    features: int
    num_layers: int
    use_y: bool = True
    use_bias: bool = True
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: jax.nn.initializers = nn.initializers.xavier_normal()
    bias_init: jax.nn.initializers = nn.initializers.zeros
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.use_y:
            self.layers = [nn.Dense(
                features=self.features,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            ) for _ in range(self.num_layers)]
        else:
            self.layers = [nn.Dense(
                features=self.features * 2,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            ) for _ in range(self.num_layers)]

    def __call__(self, inputs) -> jnp.ndarray:

        if self.use_y:
            assert len(inputs) == 2, (f'Expected 2 inputs but got {len(inputs)}')
            x, y = inputs
            for layer in self.layers:
                gate = layer(x)
                gate = nn.sigmoid(gate)
                x = jnp.multiply((1 - gate), x) + jnp.multiply(gate, y)
            return x
        else:
            x = inputs
            for layer in self.layers:
                proj_x = layer(x)
                proj_x, gate = jnp.split(proj_x, 2, axis=-1)
                proj_x = self.activation_fn(proj_x)
                gate = nn.sigmoid(gate)
                x = jnp.multiply(1 - gate, proj_x) + jnp.multiply(gate, proj_x)
            return x
