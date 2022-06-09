from typing import Optional, Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn


def attention_dropout(
        attention_weights: jnp.ndarray,
        dropout_rate: float,
        broadcast: bool,
        dropout_rng: jnp.ndarray) -> jnp.ndarray:
    keep_prob = 1. - dropout_rate

    if broadcast:
        attn_shape = list(attention_weights.shape)
        attn_shape[0] = 1
        attn_shape[1] = 1
        keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_shape)
    else:
        keep = jax.random.bernoulli(dropout_rng, keep_prob, attention_weights.shape)
    multiplier = (
            keep.astype(attention_weights.dtype) /
            jnp.asarray(keep_prob, dtype=attention_weights.dtype)
    )

    return attention_weights * multiplier


def dot_product_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        *,
        query_bias: jnp.ndarray = None,
        key_bias: jnp.ndarray = None,
        value_bias: jnp.ndarray = None,
        bias: Optional[jnp.ndarray] = None,
        mask: Optional[Any] = None,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.1,
        dropout_rng: Optional[jnp.ndarray] = None,
        deterministic: bool,
        dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = None) -> jnp.ndarray:
    assert query.ndim == key.ndim == value.ndim, 'q, k, v must have same rank'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        'q, k, v batch dims must macth'
    )
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        'q, k, v num_heads must match'
    )
    assert key.shape[-3] == value.shape[-3], 'q, v lengths must match'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must macth'

    if query_bias is not None:
        query += query_bias

    if key_bias is not None:
        key += key_bias

    if value_bias is not None:
        value += value_bias

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    attention_weights = jnp.einsum('...qhd,...khd->...hqk', query, key,
                                   precision=precision)

    if bias is not None:
        attention_weights += bias

    if mask is not None:
        neg_value = jnp.finfo(dtype).min
        attention_weights = jnp.where(mask, attention_weights, neg_value)

    attention_weights = jax.nn.softmax(attention_weights).astype(dtype)

    if not deterministic and dropout_rate > 0.:
        if dropout_rng is None:
            raise ValueError('dot_product_attention() got no rng')

        attention_weights = attention_dropout(
            attention_weights,
            dropout_rate,
            broadcast_dropout,
            dropout_rng
        )

    return jnp.einsum('...hqk,...khd->...qhd', attention_weights, value,
                      precision=precision)


def l2_attention(
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        *,
        query_bias: jnp.ndarray = None,
        key_bias: jnp.ndarray = None,
        value_bias: jnp.ndarray = None,
        attention_bias: Optional[jnp.ndarray] = None,
        mask: Optional[Any] = None,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.1,
        dropout_rng: Optional[jnp.ndarray] = None,
        deterministic: bool,
        dtype: jnp.dtype = jnp.float32,
        precision: Optional[jax.lax.Precision] = None) -> jnp.ndarray:
    assert query.ndim == key.ndim == value.ndim, 'q, k, v must have same rank'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        'q, k, v batch dims must macth'
    )
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        'q, k, v num_heads must match'
    )
    assert key.shape[-3] == value.shape[-3], 'q, v lengths must match'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must macth'

    scale = query.shape[-1]

    if query_bias is not None:
        query += query_bias

    if key_bias is not None:
        key += key_bias

    if value_bias is not None:
        value += value_bias

    query_norm = jnp.linalg.norm(query, axis=-1, keepdims=True)
    key_norm = jnp.linalg.norm(key, axis=-1, keepdims=True)

    qk_product = jnp.einsum('...qhd,...qkd->...qkh', query, key,
                            precision=precision)

    attention_weights = query_norm - 2 * qk_product + jnp.transpose(key_norm,
                                                                    (0, 1, 3, 2)) / scale
    if attention_bias is not None:
        attention_weights += attention_bias

    if mask is not None:
        neg_value = jnp.finfo(dtype).min
        attention_weights = jnp.where(mask, attention_weights, neg_value)

    attention_weights = jax.nn.softmax(-attention_weights).astype(dtype)

    if not deterministic and dropout_rate > 0.:
        if dropout_rng is None:
            raise ValueError('l2_attention() got no rng!')

        attention_weights = attention_dropout(
            attention_weights,
            dropout_rate,
            broadcast_dropout,
            dropout_rng
        )

    return jnp.einsum('...kqh,...khd->...kqd', attention_weights, value,
                      precision=precision)


def _concat_state_time(state, time):
    new_time_shape = jnp.ones_like(state[..., :1]) * time
    return jnp.concatenate([new_time_shape, state], -1)


class MultiHeadAttention(nn.Module):
    num_heads: int
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    attention_fn: Callable[..., jnp.ndarray] = dot_product_attention
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    precision: Optional[jax.lax.Precision] = None
    kernel_init: jax.nn.initializers = nn.initializers.lecun_normal()
    bias_init: jax.nn.initializers = nn.initializers.zeros
    use_same_qk_weights: Optional[bool] = False
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self,
                 inputs_q: jnp.ndarray,
                 inputs_kv: jnp.ndarray,
                 time: Optional[Any] = None,
                 mask: Optional[Any] = None,
                 deterministic: Optional[bool] = None) -> jnp.ndarray:
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_kv.shape[-1]

        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')

        head_dim = qkv_features // self.num_heads

        if self.use_same_qk_weights:
            dense_qk = nn.DenseGeneral(
                axis=-1,
                features=(self.num_heads, head_dim),
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name='query_key'
            )

            dense_v = nn.DenseGeneral(
                axis=-1,
                features=(self.num_heads, head_dim),
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name='value'
            )

            query, key, value = (dense_qk(inputs_q),
                                 dense_qk(inputs_kv),
                                 dense_v(inputs_kv))

        else:
            dense = partial(nn.DenseGeneral,
                            axis=-1,
                            dtype=self.dtype,
                            param_dtype=self.param_dtype,
                            features=(self.num_heads, head_dim),
                            kernel_init=self.kernel_init,
                            bias_init=self.bias_init,
                            use_bias=self.use_bias,
                            precision=self.precision)
            query, key, value = (dense(name='query')(inputs_q),
                                 dense(name='key')(inputs_kv),
                                 dense(name='value')(inputs_kv))

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng('dropout')

        if time is not None:
            value = _concat_state_time(value, time)

        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision
        )

        out = nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name='out'
        )(x)

        return out
