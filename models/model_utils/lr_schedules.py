from typing import Callable, Sequence

from chex import Numeric
from ml_collections import ConfigDict

import jax.numpy as jnp
import optax


def create_traingular_schedule(lr_min, lr_max, step_per_cycle):
    top = (step_per_cycle + 1) // 2

    def learning_rate_fn(step):
        cycle_step = step % step_per_cycle
        if cycle_step < top:
            lr = lr_min + (cycle_step / top) * (lr_max - lr_min)
        else:
            lr = lr_max - ((cycle_step - top) / top) * (lr_max - lr_min)
        return lr

    return learning_rate_fn


def create_transformer_schedule(d_model, warmup_steps=4000):
    def learning_rate_fn(step):
        value1 = jnp.sqrt(step)
        value2 = step * (warmup_steps ** -1.5)
        return jnp.sqrt(d_model) * jnp.minimum(value1, value2)

    return learning_rate_fn


def rsqrt_scheduler(step):
    return 1. / jnp.sqrt(step)


def reverse_rsqrt_scheduler(initial_value: float):
    def learning_rate_fn(step):
        return initial_value / jnp.sqrt(step)

    return learning_rate_fn


def combine_lr_schedules(lr_schedules: Sequence[Callable[[Numeric], Numeric]], boundaries: Sequence[int]):
    learning_rate_fn = optax.join_schedules(schedules=lr_schedules,
                                            boundaries=boundaries)
    return learning_rate_fn


def get_lr_schedule(config: ConfigDict):
    if config.lr_config.get('lr_schedule') == 'clr':
        return create_traingular_schedule(config.lr_config.get('lr_min'),
                                          config.lr_config.get('lr_max'),
                                          config.lr_config.get('step_per_cycle'))

    if config.lr_config.get('lr_schedule') == 'rsqrt':
        return rsqrt_scheduler

    if config.lr_config.get('lr_schedule') == 'reverse_rsqrt':
        return reverse_rsqrt_scheduler(config.lr_config.get('init_value'))

    if config.lr_config.get('lr_schedule') == 'transformer':
        return create_transformer_schedule(config.d_model, config.lr_config.get('warmup_steps'))

    if config.lr_config.get('lr_schedule') == 'cosine_decay':
        return optax.cosine_decay_schedule(config.lr_config.get('init_value'),
                                           config.lr_config.get('decay_steps'),
                                           config.lr_config.get('alpha'))
    if config.lr_config.get('lr_schedule') == 'polynomial':
        return optax.polynomial_schedule(config.lr_config.get('init_value'),
                                         config.lr_config.get('end_value'),
                                         config.lr_config.get('power'),
                                         config.lr_config.get('transition_steps'),
                                         config.lr_config.get('transition_begin'))

    if config.lr_config.get('lr_schedule') == 'piecewise_constant':
        return optax.piecewise_constant_schedule(config.lr_config.get('init_value'),
                                                 config.lr_config.get('boundaries_and_scales'))

    if config.lr_config.get('lr_schedule') == 'linear':
        return optax.linear_schedule(config.lr_config.get('init_value'),
                                     config.lr_config.get('end_value'),
                                     config.lr_config.get('transition_steps'),
                                     config.lr_config.get('transition_begin'))
    else:
        raise NotImplementedError(f"{config.lr_config.get('lr_schedule')} not implemented!")
