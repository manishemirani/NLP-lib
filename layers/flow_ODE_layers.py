from typing import Callable
from functools import partial

import jax.numpy as jnp
from jax import random
from jax.experimental.ode import odeint
import flax.linen as nn



class ODEBlock(nn.Module):
    flow_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    max_seq_len: int
    delta_t: float

    @nn.compact
    def __call__(self, x, params):
        print(x.shape)
        time = jnp.arange(0, self.max_seq_len, dtype=jnp.float32) * self.delta_t
        final_states = odeint(partial(self.flow_func.apply, {'params': params}),
                              x, time)
        return final_states


class ODEBlockVmap(nn.Module):
    flow_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    max_seq_len: int
    delta_t: float

    @nn.compact
    def __call__(self, x, params):
        batched_ode = nn.vmap(
            ODEBlock,
            variable_axes={'params': 0, 'nfe': None},
            split_rngs={'params': True, 'nfe': False},
            in_axes=(0, None))
        return batched_ode(flow_func=self.flow_func, max_seq_len=self.max_seq_len,
                           delta_t=self.delta_t, name='odeblock')(x, params)


class Flow_ODENet(nn.Module):
    flow_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    max_seq_len: int = 255
    delta_t: float = 0.01

    @nn.compact
    def __call__(self, x):
        init_fn = lambda rng, x: self.flow_func.init(random.split(rng)[-1], x, 0.)['params']
        odenet_params = self.param('odenet_params', init_fn, jnp.ones_like(x[0]))
        final_states = ODEBlockVmap(flow_func=self.flow_func, max_seq_len=self.max_seq_len
                                    , delta_t=self.delta_t)(x, odenet_params)

        return final_states
