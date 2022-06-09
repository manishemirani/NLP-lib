from typing import Callable, Optional, Any

from layers.pos_encodings import SinusoidalPosEmbedding
import flax.linen as nn
import jax.numpy as jnp
import ml_collections


class BaseEncoder(nn.Module):
    config: ml_collections.ConfigDict

    def setup(self):

        if self.config.share_embeddings is None:
            self.embedding = nn.Embed(
                num_embeddings=self.config.vocab_size,
                features=self.config.emb_dim,
                embedding_init=self.config.embedding_initializer)
        else:
            self.embedding = self.config.shared_embeddings

        self.pos_encoding = SinusoidalPosEmbedding(config=self.config, name='input_posemb')

    def __call__(self, source_tokens: jnp.ndarray,
                 source_positions: Optional[Any] = None,
                 encoder_mask: Optional[Any] = None) -> jnp.ndarray:
        raise NotImplementedError('No forward operation has been implemented!')


class BaseDecoder(nn.Module):
    config: ml_collections.ConfigDict

    def setup(self):
        if self.config.share_embeddings is None:
            self.embedding = nn.Embed(
                num_embeddings=self.config.vocab_size,
                features=self.config.emb_dim,
                embedding_init=self.config.embedding_initializer)
        else:
            self.embedding = self.config.share_embeddings

        self.pos_encoding = SinusoidalPosEmbedding(config=self.config, name='target_posemb')

    def _shift_right(self, x, axis=1):
        pad_widths = [(0, 0)] * x.ndim
        pad_widths[axis] = (1, 0)
        padded = jnp.pad(
            x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
        return padded[:, :-1]

    def __call__(self, targets: jnp.ndarray,
                 encoded: jnp.ndarray,
                 targets_positions: Optional[Any] = None,
                 decoder_mask: Optional[Any] = None,
                 encoder_decoder_mask: Optional[Any] = None) -> jnp.ndarray:
        raise NotImplementedError('No forward operation has been implemented!')


class Translation(nn.Module):
    encoder_model: Callable
    decoder_model: Callable
    config: ml_collections.ConfigDict

    def setup(self):
        self.encoder = self.encoder_model(self.config)
        self.decoder = self.decoder_model(self.config)

    def encode(self, inputs: jnp.ndarray,
               inputs_positions=None,
               inputs_segmentation=None):
        encoder_mask = nn.make_attention_mask(
            inputs > 0, inputs > 0, dtype=self.config.dtype
        )

        if inputs_segmentation is not None:
            encoder_mask = nn.combine_masks(
                encoder_mask,
                nn.make_attention_mask(
                    inputs_segmentation,
                    inputs_segmentation,
                    jnp.equal,
                    dtype=self.config.dtype
                )
            )

        return self.encoder(
            inputs,
            encoder_mask=encoder_mask,
            source_positions=inputs_positions
        )

    def decode(self, encoded: jnp.ndarray,
               inputs: jnp.ndarray,
               targets: jnp.ndarray,
               targets_positions: Optional[Any] = None,
               inputs_segmentation: Optional[Any] = None,
               targets_segmentation: Optional[Any] = None):

        decoder_mask = nn.combine_masks(
            nn.make_attention_mask(targets > 0, targets > 0, dtype=self.config.dtype),
            nn.make_causal_mask(targets, dtype=self.config.dtype)
        )

        encoder_decoder_mask = nn.make_attention_mask(
            targets > 0, inputs > 0, dtype=self.config.dtype
        )

        if inputs_segmentation is not None:
            decoder_mask = nn.combine_masks(
                decoder_mask,
                nn.make_attention_mask(targets_segmentation,
                                       targets_segmentation,
                                       jnp.equal,
                                       dtype=self.config.dtype)
            )

            encoder_decoder_mask = nn.combine_masks(
                encoder_decoder_mask,
                nn.make_attention_mask(
                    targets_segmentation,
                    inputs_segmentation,
                    jnp.equal,
                    dtype=self.config.dtype))

        return self.decoder(encoded, targets,
                            targets_positions=targets_positions,
                            decoder_mask=decoder_mask,
                            encoder_decoder_mask=encoder_decoder_mask)

    def __call__(self, inputs: jnp.ndarray,
                 targets: jnp.ndarray,
                 inputs_positions=None,
                 targets_positions=None,
                 inputs_segmentation=None,
                 targets_segmentation=None
                 ):

        encoded = self.encode(inputs,
                              inputs_positions=inputs_positions,
                              inputs_segmentation=inputs_segmentation)

        return self.decode(
            encoded,
            inputs,
            targets,
            targets_positions=targets_positions,
            inputs_segmentation=inputs_segmentation,
            targets_segmentation=targets_segmentation
        )


class BaseModel(object):

    def __int__(self, config: Optional[ml_collections.ConfigDict] = None):
        if config is None:
            config = self.default_config()

        self.config = config

    def build_model(self):
        raise NotImplementedError('No model has been implemented!')

    def loss_fn(self, logits: jnp.ndarray,
                targets: jnp.ndarray,
                weights: jnp.ndarray = None):
        raise NotImplementedError('No loss function has been implemented!')

    def learning_rate_fn(self):
        raise NotImplementedError('No learning rate function has been implemented!')

    def default_config(self):
        raise NotImplementedError('No config has been added!')
