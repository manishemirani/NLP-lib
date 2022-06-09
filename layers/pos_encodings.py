from typing import Tuple, Optional, Callable, Iterable, Sequence
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
    def __call__(self, inputs: jnp.ndarray, input_position=None):
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

        if input_position is None:
            return inputs + pe
        else:
            return inputs + jnp.take(pe[0], input_position, axis=0)


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
#
#
# config = ml_collections.ConfigDict()
#
# config.share_embeddings = None
# config.embedding_initializer = nn.initializers.normal(stddev=1.0)
# config.max_len = None
# config.posemb_initializer = None
# config.embedding_dim = 512
# config.vocab_size = 10
# config.output_vocab_size = 10
# config.dropout_rate = 0.1
# config.deterministic = False
# config.dtype = jnp.float32
# config.start_point = 0.
# config.end_point = 8.
# config.epsilon = 1e-4
# config.kernel_init = nn.initializers.lecun_normal()
# config.bias_init = nn.initializers.zeros
# config.label_smoothing = 0.01
# config.metric_mode = 'weighted'
# config.lr_config = ml_collections.ConfigDict()
# config.lr_config.lr_schedule = 'rsqrt'
# config.res_operation = 'add'
# config.num_heads = 8
# config.qkv_features = 512
# config.out_features = None
# config.use_same_qk_weights = False
# # config.attention_fn = dot_product_attention
# config.broadcast_dropout = False
# config.mha_dropout_rate = 0.1
# config.deterministic = False
# config.mha_kernel_init = nn.initializers.lecun_normal()
# config.mha_bias_init = nn.initializers.zeros
# config.use_bias_mha = False
# config.mlp_dim = 1024
# config.mlp_out_dim = None
# config.mlp_use_bias = True
# config.mlp_kernel_init = nn.initializers.lecun_normal()
# config.mlp_bias_init = nn.initializers.zeros
# config.mlp_activation_fn = nn.relu
# config.mlp_deterministic = False
# config.decode = False
# config.use_attend = False
# config.num_flow_layers = 1
# config.flow_layers = 10
# config.flow_model = 'linear'
# config.flow_activation_fn = nn.relu
#
#
# x = random.normal(random.PRNGKey(1), shape=(10, 10, 512))
#
# model = FlowPosEmbedding(config)
# params = model.init(random.PRNGKey(0), x)
