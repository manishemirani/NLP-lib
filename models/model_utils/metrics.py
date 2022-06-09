from typing import Optional

import jax
import jax.numpy as jnp
from flax.training import common_utils
import numpy as np
from ml_collections import config_dict


def compute_weighted_cross_entropy(logits: jnp.ndarray,
                                   targets: jnp.ndarray,
                                   weights: Optional[jnp.ndarray] = None,
                                   label_weights: Optional[jnp.ndarray] = None,
                                   label_smoothing: float = 0.0,
                                   normalize_logits: bool = True):
    if logits.ndim != targets.ndim:
        raise ValueError(
            f'Incorrect shapes. Logists --> {logits.shape} and targest --> {targets.shape} must have same shapes!')

    num_classes = logits.shape[-1]
    on_value = 1 - label_smoothing
    off_value = (1 - on_value) / (num_classes - 1)
    normalizing_value = -(on_value * jnp.log(on_value) +
                          (num_classes - 1) * off_value * jnp.log(off_value + 1e-20))
    one_hot_labels = common_utils.onehot(targets, num_classes,
                                         on_value=on_value, off_value=off_value)

    if label_weights is not None:
        one_hot_labels *= weights

    if normalize_logits:
        logits = jax.nn.log_softmax(logits)

    loss = -jnp.einsum('...x,...x->...', one_hot_labels, logits)
    loss = loss - normalizing_value

    normalizing_factor = np.prod(targets.shape)
    if weights is not None:
        loss *= weights
        normalizing_factor = weights.sum()

    mean_loss = loss.sum() / normalizing_factor

    return mean_loss, normalizing_factor


def compute_cross_entropy(logits: jnp.ndarray,
                          targets: jnp.ndarray,
                          normalize_logits: bool = True):
    if logits.ndim != targets.ndim:
        raise ValueError(
            f'Incorrect shapes. Logists --> {logits.shape} and targest --> {targets.shape} must have same shape!')

    if normalize_logits:
        logits = jax.nn.log_softmax(logits)

    one_hot_labels = common_utils.onehot(targets, num_classes=logits.shape[-1])

    loss = -jnp.mean(jnp.einsum('...x,...x->...', one_hot_labels, logits))
    return loss


def compute_accuracy(logits: jnp.ndarray, targets: jnp.ndarray):
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    return acc


def compute_weighted_accuracy(logits: jnp.ndarray, targets: jnp.ndarray,
                              weight: Optional[jnp.ndarray] = None):
    acc = jnp.mean(jnp.argmax(logits, -1) == targets)
    normalizing_factor = np.prod(logits.shape[:-1])

    if weight is not None:
        acc *= weight
        normalizing_factor = weight.sum()

    return acc, normalizing_factor


def get_metrics(logits: jnp.ndarray, targets: jnp.ndarray, weight: Optional[jnp.ndarray] = None,
                label_weights: Optional[jnp.ndarray] = None, label_smoothing: float = 0.0,
                mode: str = 'weighted'):
    metrics = config_dict.ConfigDict()

    if mode == 'weighted':
        loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weight, label_weights, label_smoothing)
        accuracy, _ = compute_weighted_accuracy(logits, targets, weight)
        metrics.loss = loss
        metrics.accuracy = accuracy
        metrics.denominator = weight_sum

    elif mode == 'none-weighted':
        loss = compute_cross_entropy(logits, targets)
        accuracy = compute_accuracy(logits, targets)
        metrics.a = loss
        metrics.b = accuracy

    else:
        raise NotImplementedError(f'{mode} is not implemented!')

    metrics = jax.lax.psum(metrics, axis_name='batch')

    return metrics
