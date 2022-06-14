from typing import Tuple
from layers.flow_ODE_layers import Flow_ODENet

import flax.linen as nn
import jax.numpy as jnp
from jax import random
import numpy as np
import ml_collections


def sinusoidal_init(max_len: int, min_scale=1.0, max_scale=10000.0):
    """1D Sinusoidal Position Embedding Initializer"""

    def init(key, shape: Tuple,
             dtype=jnp.float32):
        """
            Sinusoidal init
        """
        del key, dtype

        d_feature = shape[-1]
        pe_matrix = np.zeros((max_len, d_feature), dtype=np.float32)
        positions = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_feature, 2) *
                          (-np.log(max_scale / min_scale)) / d_feature)
        pe_matrix[:, ::2] = np.sin(positions * div_term)
        pe_matrix[:, 1::2] = np.cos(positions * div_term)
        pe_matrix = pe_matrix[np.newaxis, :, :]

        return jnp.array(pe_matrix)

    return init


class SinusoidalPosEmbedding(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, positions=None):
        assert inputs.ndim == 3, (f'Number of dimension should be 3, but got {inputs.ndim}')

        seq_length = inputs.shape[1]
        max_len = self.config.max_len if self.config.max_len is not None else inputs.shape[0]

        pos_embedding_shape = (1, max_len, inputs.shape[-1])

        if self.config.posemb_initializer is None:
            pos_embedding = sinusoidal_init(max_len)(
                None, pos_embedding_shape, None
            )
        else:
            pos_embedding = self.param('pos_embedding',
                                       self.posemb_initializer,
                                       pos_embedding_shape)

        pe = pos_embedding[:, :seq_length, :]

        if positions is None:
            return inputs + pe
        else:
            return inputs + jnp.take(pe[0], positions, axis=0)


class FlowPosEmbedding(nn.Module):
    config: ml_collections.ConfigDict

    def setup(self):

        assert self.config.num_flow_layers != None, ("Expecting number of layers, but given 'None'")

        if self.config.flow_model == 'linear':
            assert self.config.embedding_dim != None, (
                "Expecting embedding dimension, but given 'None'"
            )
            from layers.flow_layers import Linear_flow
            self.flow_odenet = Flow_ODENet(
                Linear_flow(embedding_dim=self.config.embedding_dim,
                            num_layers=self.config.num_flow_layers,
                            activation_fn=self.config.flow_activation_fn)
            )
        elif self.flow_model == 'conv':
            from layers.flow_layers import Conv_flow
            self.flow_odenet = Flow_ODENet(
                Conv_flow(input_features=self.config.flow_conv_features,
                          output_features=self.config.embedding_dim,
                          num_layers=self.config.num_flow_layers,
                          kernel_size=self.config.flow_conv_kernel_size,
                          strides=self.config.flow_conv_strides, padding=self.config.flow_conv_padding)
            )
        else:
            raise NotImplementedError(f'Flow model {self.flow_model} has not been implemented!')

        init_fn = lambda rng, shape: random.normal(rng, shape=shape)

        self.flow_weights = self.param('flow_weights',
                                       init_fn,
                                       (3, self.config.flow_layers, self.config.embedding_dim))

    def __call__(self, inputs: jnp.ndarray):
        bias_vector = self.flow_odenet(self.flow_weights)

        return bias_vector
