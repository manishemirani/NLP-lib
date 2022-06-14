from layers.attention_layers import dot_product_attention
from layers.base_layers import EncoderLayer, DecoderLayer
from layers.dlcl import DynamicLinearCombination
from models.model_utils.metrics import get_metrics
from models.model_utils.lr_schedules import get_lr_schedule
from models.base_models import *

import flax.linen as nn
import jax.numpy as jnp
from jax.experimental.ode import odeint
import jax.random as random
import ml_collections


class Encoder(BaseEncoder):

    @nn.compact
    def __call__(self, source_tokens: jnp.ndarray,
                 source_positions: Optional[Any] = None,
                 encoder_mask: Optional[Any] = None):
        assert source_tokens.ndim == 2, 'Source tokens should have 2 dimensions (batch, source_vocab_size)'

        x = source_tokens.astype('int32')
        x = self.embedding(x)
        x = self.pos_encoding(x, positions=source_positions)
        x = nn.Dropout(rate=self.config.dropout_rate)(
            x, deterministic=self.config.deterministic)
        x = x.astype(self.config.dtype)
        dlcl = DynamicLinearCombination(self.config.num_layers, x.shape)

        for i in range(self.config.num_layers):
            x = EncoderLayer(config=self.config, name=f'Encoder_layer{i}')(x, encoder_mask=encoder_mask)
            dlcl(x, 'add_to_memory')

        x = dlcl(x, 'forward')

        encoded_out = nn.LayerNorm(epsilon=self.config.epsilon,
                                   dtype=self.config.dtype)(x)

        return encoded_out


class Decoder(BaseDecoder):

    @nn.compact
    def __call__(self, encoded: jnp.ndarray,
                 targets: jnp.ndarray,
                 targets_positions: Optional[Any] = None,
                 decoder_mask: Optional[Any] = None,
                 encoder_decoder_mask: Optional[Any] = None):
        x = targets.astype('int32')
        if not self.config.decode:
            x = self._shift_right(x)

        x = self.embedding(x)
        x = self.pos_encoding(x, positions=targets_positions)
        x = nn.Dropout(rate=self.config.dropout_rate)(
            x, deterministic=self.config.deterministic)

        x = x.astype(self.config.dtype)


        dlcl = DynamicLinearCombination(self.config.num_layers, x.shape)
        dlcl.clean()

        for i in range(self.config.num_layers):
            x = DecoderLayer(self.config, name=f'Decoder_layer{i}')(x, None,
                                                                    encoded,
                                                                    decoder_mask,
                                                                    encoder_decoder_mask)
            dlcl(x, 'add_to_memory')

        x = dlcl(x, 'forward')

        decoded_out = nn.LayerNorm(epsilon=self.config.epsilon,
                                   dtype=self.config.dtype)(x)

        return decoded_out

class DlclTransformer(Translation):

    @nn.compact
    def __call__(self, inputs: jnp.ndarray,
                 targets: jnp.ndarray,
                 inputs_positions=None,
                 targets_positions=None,
                 inputs_segmentation=None,
                 targets_segmentation=None
                 ):
        super(DlclTransformer, self).__call__(inputs, targets, inputs_positions, targets_positions,
                                             inputs_segmentation, targets_segmentation)


class DlclTranslation(BaseModel):

    def __init__(self, config: Optional[ml_collections.ConfigDict] = None):
        if config is None:
            config = self.default_config()

        self.config = config

    def build_model(self):
        return DlclTransformer(Encoder, Decoder, self.config)

    def loss_fn(self, logits: jnp.ndarray, targets: jnp.ndarray, weight: Optional[jnp.ndarray] = None,
                label_weights: Optional[jnp.ndarray] = None):
        return get_metrics(logits, targets, weight, label_weights, self.config.label_smoothing,
                           self.config.metric_mode)

    def learning_rate_fn(self):
        return get_lr_schedule(self.config)

    def default_config(self):
        config = ml_collections.ConfigDict()

        config.share_embeddings = None
        config.embedding_initializer = nn.initializers.normal(stddev=1.0)
        config.max_len = None
        config.posemb_initializer = None
        config.emb_dim = 512
        config.vocab_size = 10
        config.output_vocab_size = 10
        config.dropout_rate = 0.1
        config.deterministic = False
        config.dtype = jnp.float32
        config.start_point = 0.
        config.end_point = 8.
        config.epsilon = 1e-4
        config.kernel_init = nn.initializers.lecun_normal()
        config.bias_init = nn.initializers.zeros
        config.label_smoothing = 0.01
        config.metric_mode = 'weighted'
        config.lr_config = ml_collections.ConfigDict()
        config.lr_config.lr_schedule = 'rsqrt'
        config.res_operation = 'add'
        config.num_heads = 8
        config.qkv_features = 512
        config.out_features = None
        config.use_same_qk_weights = False
        config.attention_fn = dot_product_attention
        config.broadcast_dropout = False
        config.mha_dropout_rate = 0.1
        config.deterministic = False
        config.mha_kernel_init = nn.initializers.lecun_normal()
        config.mha_bias_init = nn.initializers.zeros
        config.mha_use_bias = False
        config.mlp_dim = 1024
        config.mlp_out_dim = None
        config.mlp_use_bias = True
        config.mlp_kernel_init = nn.initializers.lecun_normal()
        config.mlp_bias_init = nn.initializers.zeros
        config.mlp_activation_fn = nn.relu
        config.mlp_deterministic = False
        config.decode = False
        config.use_attend = False
        config.num_layers = 30

        return config