from typing import Sequence, Callable

import flax.linen as nn
import jax.numpy as jnp



def concat_state_time(state, time):
    new_time_shape = jnp.ones_like(state[..., :1]) * time
    return jnp.concatenate([new_time_shape, state], -1)


class Dens_time(nn.Module):
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, state, time):
        return nn.Dense(features=self.features, use_bias=self.use_bias,
                        kernel_init=nn.initializers.xavier_normal())(concat_state_time(state, time))


class Conv_time(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: int
    padding: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, state, time):
        return nn.Conv(features=self.features, kernel_size=self.kernel_size,
                       use_bias=self.use_bias,
                       kernel_init=nn.initializers.xavier_uniform())(concat_state_time(state, time))

class Convtr_t(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: int
    padding: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, state, time):
        return nn.ConvTranspose(features=self.features, kernel_size=self.kernel_size,
                                use_bias=self.use_bias,
                                kernel_init=nn.initializers.xavier_uniform())(concat_state_time(state, time))


class Conv_flow(nn.Module):
    input_features: int
    output_features: int
    num_layers: int
    kernel_size: Sequence[int]
    strides: int
    padding: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.activation.selu
    use_bias: bool = True

    @nn.compact
    def __call__(self, state, time):
        x = state
        for _ in range(self.num_layers - 1):
            x = Conv_time(features=self.input_features, kernel_size=self.kernel_size,
                          use_bias=self.use_bias, strides=self.strides, padding=self.padding)(x, time)
            x = self.activation_fn(x)
        return Convtr_t(features=self.output_features, kernel_size=self.kernel_size,
                        strides=self.strides, padding=self.padding,
                        use_bias=self.use_bias)(x, time)


class Linear_flow(nn.Module):
    embedding_dim: int
    num_layers: int
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, state, time):
        x = state
        for _ in range(self.num_layers - 1):
            x = Dens_time(features=self.embedding_dim)(x, time)
            x = self.activation_fn(x)
        return Dens_time(features=self.embedding_dim)(x, time)
