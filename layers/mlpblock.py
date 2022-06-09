from typing import Optional, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

Initializer = jax.nn.initializers

class MLPBlock(nn.Module):
    mlp_dim: int
    out_dim: int
    dropout_rate: float = 0.1
    use_bias: bool = True
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    precision: Optional[jax.lax.Precision] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, deterministic):
        out_dim = inputs.shape[-1] or self.out_dim
        x = nn.Dense(
            self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision
        )(inputs)

        x = self.activation_fn(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision
        )(x)

        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output