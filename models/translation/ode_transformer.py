from functools import partial

from layers.attention_layers import dot_product_attention
from layers.base_layers import EncoderLayer, DecoderLayer
from models.model_utils.metrics import get_metrics
from models.model_utils.lr_schedules import get_lr_schedule
from models.base_models import *
import flax.linen as nn
import jax.numpy as jnp
from jax.experimental.ode import odeint
import jax.random as random
import ml_collections


def _computes_time_series(start_point: float, end_point: float, dtype: jnp.dtype = jnp.float32):
    return jnp.arange(start_point, end_point, dtype=dtype)


class ODEEnocderBlock(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, inputs, encoder_mask, params):
        time_series = _computes_time_series(self.config.start_point,
                                            self.config.end_point,
                                            self.config.dtype)
        final_outputs = odeint(partial(EncoderLayer(self.config).apply, {'params': params},
                                       encoder_mask=encoder_mask,
                                       rngs={'dropout': random.PRNGKey(2)}), jnp.expand_dims(inputs, axis=0),
                               time_series)
        return final_outputs


class ODEDecoderBlock(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, targets,
                 encoded,
                 decoder_maks,
                 encoder_decoder_mask,
                 params):
        time_series = _computes_time_series(self.config.start_point,
                                            self.config.end_point,
                                            self.config.dtype)
        final_outputs = odeint(partial(DecoderLayer(self.config).apply, {'params': params},
                                       encoded_tokens=jnp.expand_dims(encoded, 0),
                                       decoder_mask=decoder_maks,
                                       encoder_decoder_mask=encoder_decoder_mask,
                                       rngs={'dropout': random.PRNGKey(2)}), jnp.expand_dims(targets, axis=0),
                               time_series)
        return final_outputs


class EncoderODEVmap(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, inputs, encoder_mask, params):
        batched_ode = nn.vmap(
            ODEEnocderBlock,
            variable_axes={'params': 0, 'nfe': None},
            split_rngs={'params': False, 'nfe': False},
            in_axes=(0, 0, None),
            out_axes=0)
        return batched_ode(config=self.config)(inputs, encoder_mask, params)


class DecoderODEVmap(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, targets,
                 encoded,
                 decoder_mask,
                 encoder_decoder_mask,
                 params):
        batched_ode = nn.vmap(
            ODEDecoderBlock,
            variable_axes={'params': 0, 'nfe': None},
            split_rngs={'params': True, 'nfe': False},
            in_axes=(0, 0, 0, 0, None),
            out_axes=0)
        return batched_ode(config=self.config)(targets, encoded, decoder_mask,
                                               decoder_mask, params)


class ODEEncoder(BaseEncoder):
    config: ml_collections.ConfigDict

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
        init_fn = lambda rng, x: EncoderLayer(self.config).init({'params': random.split(rng)[-1],
                                                                 'dropout': random.split(rng)[0]},
                                                                jnp.ones_like(x[0])[jnp.newaxis, ...], 0.
                                                                )['params']
        odenet_params = self.param('odenet_params', init_fn, x)

        x = EncoderODEVmap(config=self.config)(x, encoder_mask=encoder_mask,
                                               params=odenet_params)[:, 0, 0, :, :]

        encoded_out = nn.LayerNorm(epsilon=self.config.epsilon,
                                   dtype=self.config.dtype)(x)

        return encoded_out


class ODEDecoder(BaseDecoder):

    @nn.compact
    def __call__(self, encoded: jnp.ndarray,
                 targets: jnp.ndarray,
                 targets_positions: Optional[Any] = None,
                 decoder_mask: Optional[Any] = None,
                 encoder_decoder_mask: Optional[Any] = None):
        assert targets.ndim == 2, 'Target tokens must have 2 dimensions (batch, target_vocab_size)'
        assert encoded.ndim == 3, 'Encoded tokens must have 3 dimensions (batch, len, depth)'

        x = targets.astype('int32')
        if not self.config.decode:
            x = self._shift_right(x)

        x = self.embedding(x)
        x = self.pos_encoding(x, positions=targets_positions)
        x = nn.Dropout(rate=self.config.dropout_rate)(
            x, deterministic=self.config.deterministic)

        x = x.astype(self.config.dtype)

        init_fn = lambda rng, x: DecoderLayer(self.config).init({'params': random.split(rng)[-1],
                                                                 'dropout': random.split(rng)[0]},
                                                                jnp.ones_like(x[0])[jnp.newaxis, ...],
                                                                0.,
                                                                jnp.ones_like(encoded[0])[jnp.newaxis, ...],
                                                                )['params']
        odenet_params = self.param('odenet_params', init_fn, x)

        x = DecoderODEVmap(config=self.config)(x, encoded,
                                               decoder_mask,
                                               encoder_decoder_mask,
                                               odenet_params)[:, 0, 0, :, :]

        x = nn.LayerNorm(epsilon=self.config.epsilon,
                         dtype=self.config.dtype)(x)

        if self.config.use_attend:
            logits = self.embedding.attend(x)

            logits = logits / jnp.sqrt(x.shape[-1])
        else:
            logits = nn.Dense(
                features=self.config.output_vocab_size,
                dtype=self.config.dtype,
                kernel_init=self.config.kernel_init,
                bias_init=self.config.bias_init,
                name='logits'
            )(x)

        return logits


class ODETransformer(Translation):

    @nn.compact
    def __call__(self, inputs: jnp.ndarray,
                 targets: jnp.ndarray,
                 inputs_positions=None,
                 targets_positions=None,
                 inputs_segmentation=None,
                 targets_segmentation=None
                 ):
        super(ODETransformer, self).__call__(inputs, targets, inputs_positions, targets_positions,
                                             inputs_segmentation, targets_segmentation)


class ODETranslation(BaseModel):

    def __init__(self, config: Optional[ml_collections.ConfigDict] = None):
        if config is None:
            config = self.default_config()

        self.config = config

    def build_model(self):
        return ODETransformer(ODEEncoder, ODEDecoder, self.config)

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
        config.use_bias_mha = False
        config.mlp_dim = 1024
        config.mlp_out_dim = None
        config.mlp_use_bias = True
        config.mlp_kernel_init = nn.initializers.lecun_normal()
        config.mlp_bias_init = nn.initializers.zeros
        config.mlp_activation_fn = nn.relu
        config.mlp_deterministic = False
        config.decode = False
        config.use_attend = False

        return config
