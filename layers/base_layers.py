from typing import Optional, Any

from layers.attention_layers import MultiHeadAttention
from layers.resnets import ResNet
from layers.mlpblock import MLPBlock
import flax.linen as nn
import jax.numpy as jnp
import ml_collections


class EncoderLayer(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, source_tokens: jnp.ndarray,
                 time: Optional[Any] = None,
                 encoder_mask: Optional[Any] = None) -> jnp.ndarray:
        assert source_tokens.ndim == 3, (f'Nmber of input dimensions must be 3 but got {source_tokens.ndim}')
        resnet = ResNet(self.config)

        x = nn.LayerNorm(epsilon=self.config.epsilon,
                         dtype=self.config.dtype)(source_tokens)

        x = MultiHeadAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_features,
            out_features=self.config.out_features,
            use_same_qk_weights=self.config.use_same_qk_weights,
            attention_fn=self.config.attention_fn,
            broadcast_dropout=self.config.broadcast_dropout,
            dropout_rate=self.config.mha_dropout_rate,
            deterministic=self.config.deterministic,
            kernel_init=self.config.mha_kernel_init,
            bias_init=self.config.mha_bias_init,
            use_bias=self.config.mha_use_bias,
            dtype=self.config.dtype
        )(x, x, time, encoder_mask)

        x = nn.Dropout(rate=self.config.dropout_rate)(
            x, deterministic=self.config.deterministic)

        x = resnet(source_tokens, x)

        x = nn.LayerNorm(epsilon=self.config.epsilon,
                         dtype=self.config.dtype)(x)
        y = MLPBlock(
            mlp_dim=self.config.mlp_dim,
            out_dim=self.config.mlp_out_dim,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.mlp_use_bias,
            kernel_init=self.config.mlp_kernel_init,
            bias_init=self.config.mlp_bias_init,
            activation_fn=self.config.mlp_activation_fn,
            dtype=self.config.dtype
        )(x, deterministic=self.config.mlp_deterministic)

        out = nn.LayerNorm(epsilon=self.config.epsilon,
                           dtype=self.config.dtype)(resnet(x, y))
        return out


class DecoderLayer(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, target_tokens: jnp.ndarray,
                 time,
                 encoded_tokens: jnp.ndarray,
                 decoder_mask: Optional[Any] = None,
                 encoder_decoder_mask: Optional[Any] = None) -> jnp.ndarray:
        assert target_tokens.ndim == 3, (f'Nmber of input dimensions must be 3 but got {target_tokens.ndim}')
        resnet = ResNet(self.config)

        x = nn.LayerNorm(epsilon=self.config.epsilon,
                         dtype=self.config.dtype)(target_tokens)

        x = MultiHeadAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_features,
            out_features=self.config.out_features,
            use_same_qk_weights=self.config.use_same_qk_weights,
            attention_fn=self.config.attention_fn,
            broadcast_dropout=self.config.broadcast_dropout,
            dropout_rate=self.config.mha_dropout_rate,
            deterministic=self.config.deterministic,
            kernel_init=self.config.mha_kernel_init,
            bias_init=self.config.mha_bias_init,
            use_bias=self.config.mha_use_bias,
            dtype=self.config.dtype
        )(x, x, time, decoder_mask)

        x = nn.Dropout(rate=self.config.dropout_rate)(
            x, deterministic=self.config.deterministic)

        x = resnet(target_tokens, x)

        x = nn.LayerNorm(epsilon=self.config.epsilon,
                         dtype=self.config.dtype)(x)

        y = MultiHeadAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_features,
            out_features=self.config.out_features,
            use_same_qk_weights=self.config.use_same_qk_weights,
            attention_fn=self.config.attention_fn,
            broadcast_dropout=self.config.broadcast_dropout,
            dropout_rate=self.config.mha_dropout_rate,
            deterministic=self.config.deterministic,
            kernel_init=self.config.mha_kernel_init,
            bias_init=self.config.mha_bias_init,
            use_bias=self.config.mha_use_bias,
            dtype=self.config.dtype
        )(x, encoded_tokens, time, encoder_decoder_mask)

        y = MLPBlock(
            mlp_dim=self.config.mlp_dim,
            out_dim=self.config.mlp_out_dim,
            dropout_rate=self.config.dropout_rate,
            use_bias=self.config.mlp_use_bias,
            kernel_init=self.config.mlp_kernel_init,
            bias_init=self.config.mlp_bias_init,
            activation_fn=self.config.mlp_activation_fn,
            dtype=self.config.dtype
        )(y, deterministic=self.config.mlp_deterministic)

        out = nn.LayerNorm(epsilon=self.config.epsilon,
                           dtype=self.config.dtype)(resnet(x, y))

        return out