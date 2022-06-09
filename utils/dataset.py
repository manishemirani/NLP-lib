from typing import Dict, Optional, List, Union, Callable, \
    Any

from clu import deterministic_data

import tensorflow_datasets as tfds
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_text as text
import tensorflow as tf
import jax
import ml_collections
import os

AUTOTUNE = tf.data.AUTOTUNE


def rename_features(ds_info: tfds.core.DatasetInfo,
                    new_keys: List[str] = ['inputs', 'targets'],
                    reverse_translation: bool = False):
    input_language, target_language = ds_info.supervised_keys

    if reverse_translation:
        input_language, target_language = target_language, input_language

    def rename(features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        features[new_keys[0]] = features.pop(input_language)
        features[new_keys[1]] = features.pop(target_language)
        return features

    return rename


def max_tokens_filter(max_length):
    def filter_fn(features):
        num_tokens = tf.maximum(tf.shape(features['inputs'])[0], tf.shape(features['targets'])[0])
        return tf.less(num_tokens, max_length + 1)

    return filter_fn


def get_raw_wmt_dataset(builder: tfds.core.DatasetBuilder,
                        batch_size: int,
                        num_epochs: int,
                        split: str,
                        preprocess_fn: Callable,
                        rng: jax.random.PRNGKey,
                        shuffle: bool = False,
                        shuffle_buffer_size: int = 10_000,
                        drop_remainder: bool = False):
    number_of_examples = builder.info.splits[split].num_examples
    ds_split = deterministic_data.get_read_instruction_for_host(
        split, number_of_examples, drop_remainder=drop_remainder
    )

    ds = deterministic_data.create_dataset(
        builder,
        split=ds_split,
        rng=rng,
        batch_dims=[jax.local_device_count(), batch_size // jax.device_count()],
        preprocess_fn=preprocess_fn,
        num_epochs=num_epochs,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size)

    return ds


def pack_dataset(dataset: tf.data.Dataset,
                 key2length: Union[int, Dict[str, int]],
                 keys: Optional[List[str]] = None) -> tf.data.Dataset:
    """Creates a 'packed' version of a dataset on-the-fly.

    Adapted from the mesh-tf implementation.

    This is meant to replace the irritation of having to create a separate
    "packed" version of a dataset to train efficiently on TPU.
    Each example in the output dataset represents several examples in the
    input dataset.
    For each key in the input dataset, two additional keys are created:
    <key>_segmentation: an int32 tensor identifying the parts
       representing the original example.
    <key>_position: an int32 tensor identifying the position within the original
       example.
    Example:
    Two input examples get combined to form an output example.
    The input examples are:
    {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
    {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
    The output example is:
    {
                   "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
      "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
          "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                  "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
     "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
         "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
    }
    0 represents padding in both the inputs and the outputs.
    Sequences in the incoming examples are truncated to length "length", and the
    sequences in the output examples all have fixed (padded) length "length".

    Args:
      dataset: a tf.data.Dataset
      key2length: an integer, or a dict from feature-key to integer
      keys: a list of strings (e.g. ["inputs", "targets"])

    Returns:
      a tf.data.Dataset
    # """
    shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
    if keys is None:
        keys = list(shapes.keys())
    for k in keys:
        if k not in shapes:
            raise ValueError('Key %s not found in dataset.  Available keys are %s' %
                             (k, shapes.keys()))
        if not shapes[k].is_compatible_with(tf.TensorShape([None])):
            raise ValueError('Tensors to be packed must be one-dimensional.')
    # make sure that the length dictionary contains all keys as well as the
    # keys suffixed by "_segmentation" and "_position"
    if isinstance(key2length, int):
        key2length = {k: key2length for k in keys}
    for k in keys:
        for suffix in ['_segmentation', '_position']:
            key2length[k + suffix] = key2length[k]

    # trim to length
    dataset = dataset.map(
        lambda x: {k: x[k][:key2length[k]] for k in keys},
        num_parallel_calls=AUTOTUNE)
    # Setting batch_size=length ensures that the concatenated sequences (if they
    # have length >=1) are sufficient to fill at least one packed example.
    batch_size = max(key2length.values())
    dataset = dataset.padded_batch(
        batch_size, padded_shapes={k: [-1] for k in keys})
    dataset = _pack_with_tf_ops(dataset, keys, key2length)

    # Set the Tensor shapes correctly since they get lost in the process.
    def my_fn(x):
        return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}

    return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(dataset: tf.data.Dataset, keys: List[str],
                      key2length: Dict[str, int]) -> tf.data.Dataset:
    """Helper-function for packing a dataset which has already been batched.

    Helper for pack_dataset()  Uses tf.while_loop.

    Args:
      dataset: a dataset containing padded batches of examples.
      keys: a list of strings
      key2length: an dict from feature-key to integer

    Returns:
      a dataset.
    """
    empty_example = {}
    for k in keys:
        empty_example[k] = tf.zeros([0], dtype=tf.int32)
        empty_example[k + '_position'] = tf.zeros([0], dtype=tf.int32)
    keys_etc = empty_example.keys()

    def write_packed_example(partial, outputs):
        new_partial = empty_example.copy()
        new_outputs = {}
        for k in keys_etc:
            new_outputs[k] = outputs[k].write(
                outputs[k].size(),
                tf.pad(partial[k], [[0, key2length[k] - tf.size(partial[k])]]))
        return new_partial, new_outputs

    def map_fn(x):
        """Internal function to flat_map over.

        Consumes a batch of input examples and produces a variable number of output
        examples.
        Args:
          x: a single example

        Returns:
          a tf.data.Dataset
        """
        partial = empty_example.copy()
        i = tf.zeros([], dtype=tf.int32)
        dynamic_batch_size = tf.shape(x[keys[0]])[0]
        outputs = {}
        for k in keys:
            outputs[k] = tf.TensorArray(
                tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
            outputs[k + '_position'] = tf.TensorArray(
                tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])

        def body_fn(i, partial, outputs):
            """Body function for while_loop.

            Args:
              i: integer scalar
              partial: dictionary of Tensor (partially-constructed example)
              outputs: dictionary of TensorArray

            Returns:
              A triple containing the new values of the inputs.
            """
            can_append = True
            one_example = {}
            for k in keys:
                val = tf.cast(x[k][i], tf.int32)
                val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
                one_example[k] = val
            for k in keys:
                can_append = tf.logical_and(
                    can_append,
                    tf.less_equal(
                        tf.size(partial[k]) + tf.size(one_example[k]), key2length[k]))

            def false_fn():
                return write_packed_example(partial, outputs)

            def true_fn():
                return partial, outputs

            partial, outputs = tf.cond(can_append, true_fn, false_fn)
            new_partial = {}
            for k in keys:
                new_seq = one_example[k][:key2length[k]]
                new_seq_len = tf.size(new_seq)
                new_partial[k] = tf.concat([partial[k], new_seq], 0)
                new_partial[k + '_position'] = tf.concat(
                    [partial[k + '_position'],
                     tf.range(new_seq_len)], 0)
            partial = new_partial
            return i + 1, partial, outputs

        # For loop over all examples in the batch.
        i, partial, outputs = tf.while_loop(
            cond=lambda *_: True,
            body=body_fn,
            loop_vars=(i, partial, outputs),
            shape_invariants=(
                tf.TensorShape([]),
                {k: tf.TensorShape([None]) for k in keys_etc},
                {k: tf.TensorShape(None) for k in keys_etc},
            ),
            maximum_iterations=dynamic_batch_size)
        _, outputs = write_packed_example(partial, outputs)
        packed = {k: outputs[k].stack() for k in keys_etc}
        for k in keys:
            packed[k + '_segmentation'] = (
                    tf.cumsum(
                        tf.cast(tf.equal(packed[k + '_position'], 0), tf.int32), axis=1) *
                    tf.cast(tf.not_equal(packed[k], 0), tf.int32))
        return packed

    dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
    return dataset.unbatch()


def get_tokenizers(dataset: tf.data.Dataset,
                   keys: List[str],
                   input_lang_vocab_path: str,
                   target_lang_vocab_path: str,
                   vocab_size: int,
                   reversed_tokens: List[str],
                   bert_tokenizer_params=None,
                   learn_params=None,
                   lower_case: bool = True):
    def create_vocab_file(file_path: str, vocab):
        with open(file_path, 'w') as file:
            for token in vocab:
                print(token, file=file)
        file.close()

    input_lang_ds = dataset.map(lambda features: features[keys[0]])
    target_lang_ds = dataset.map(lambda features: features[keys[1]])

    input_lang_vocab = bert_vocab.bert_vocab_from_dataset(
        input_lang_ds,
        vocab_size=vocab_size,
        reserved_tokens=reversed_tokens,
        bert_tokenizer_params=bert_tokenizer_params,
        learn_params=learn_params,
    )
    target_lang_vocab = bert_vocab.bert_vocab_from_dataset(
        target_lang_ds,
        vocab_size=vocab_size,
        reserved_tokens=reversed_tokens,
        bert_tokenizer_params=bert_tokenizer_params,
        learn_params=learn_params
    )

    if not os.path.exists(input_lang_vocab_path):
        create_vocab_file(input_lang_vocab_path, input_lang_vocab)

    if not os.path.exists(target_lang_vocab_path):
        create_vocab_file(target_lang_vocab_path, target_lang_vocab)

    input_lang_tokenizer = text.BertTokenizer(input_lang_vocab_path, lower_case=lower_case)
    target_lang_tokenizer = text.BertTokenizer(target_lang_vocab_path, lower_case=lower_case)

    return input_lang_tokenizer, target_lang_tokenizer


class Tokenize:

    def __init__(self, input_tokenizer: Any, target_tokenizer: Any,
                 ds_keys: List[str]):
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.inputs = ds_keys[0]
        self.targets = ds_keys[1]

    def __call__(self, features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        tokenized_input = self.input_tokenizer.tokenize(features[self.inputs])[0]
        tokenized_target = self.target_tokenizer.tokenize(features[self.targets])[0]
        features[self.inputs] = tf.cast(tokenized_input.to_tensor()[0], dtype=tf.int32)[0]
        features[self.targets] = tf.cast(tokenized_target.to_tensor()[0], dtype=tf.int32)[0]
        return features


def preprocess_wmt_dataset(dataset: tf.data.Dataset,
                           batch_size: int,
                           num_epochs: int,
                           max_length: int,
                           pack_examples: bool = True,
                           drop_remainder: bool = False,
                           prefetch_size: int = AUTOTUNE):
    if max_length > 0:
        dataset = dataset.filter(max_tokens_filter(max_length))

    dataset = dataset.repeat(num_epochs)

    if pack_examples:
        dataset = pack_dataset(dataset, max_length)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    else:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                'inputs': max_length,
                'targets': max_length
            },
            padding_values={
                'inputs': 0,
                'targets': 0
            },

            drop_remainder=drop_remainder
        )

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset


def get_wmt_dataset(config: ml_collections.ConfigDict):
    train_dataset_builder = tfds.builder(config.dataset_name)
    if not os.path.exists(config.download_dir + f'/{config.dataset_name}'):
        train_dataset_builder.download_and_prepare(download_dir=config.download_dir)
    train_ds = get_raw_wmt_dataset(builder=train_dataset_builder,
                                   batch_size=config.batch_size,
                                   num_epochs=config.num_epochs,
                                   split=config.train_split,
                                   preprocess_fn=rename_features(train_dataset_builder.info,
                                                                 reverse_translation=config.reverse_translation),
                                   shuffle=config.shuffle,
                                   drop_remainder=config.drop_remainder,
                                   rng=None)

    if config.eval_dataset is not None:
        eval_dataset_builder = tfds.builder(config.eval_dataset_name)
    else:
        eval_dataset_builder = train_dataset_builder

    eval_ds = get_raw_wmt_dataset(builder=eval_dataset_builder,
                                  batch_size=config.batch_size,
                                  num_epochs=config.num_epochs,
                                  split=config.eval_split,
                                  preprocess_fn=rename_features(ds_info=eval_dataset_builder.info,
                                                                new_keys=config.ds_keys,
                                                                reverse_translation=config.reverse_translation),
                                  shuffle=config.shuffle,
                                  drop_remainder=config.drop_remainder,
                                  rng=None)

    input_lang_tokenizer, target_lang_tokenizer = get_tokenizers(
        dataset=train_ds,
        keys=config.ds_keys,
        input_lang_vocab_path=config.input_lang_vocab,
        target_lang_vocab_path=config.target_lang_vocab,
        vocab_size=config.vocab_size,
        reversed_tokens=config.reversed_tokens,
        bert_tokenizer_params=config.bert_tokenizer_params,
        learn_params=config.learn_params,
        lower_case=config.lower_case,
    )

    train_data = train_ds.map(Tokenize(input_lang_tokenizer, target_lang_tokenizer,
                                       config.ds_keys),
                            num_parallel_calls=AUTOTUNE)

    eval_data = eval_ds.map(Tokenize(input_lang_tokenizer, target_lang_tokenizer,
                                     config.ds_keys),
                          num_parallel_calls=AUTOTUNE)

    train_ds = preprocess_wmt_dataset(
        dataset=train_data,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        max_length=config.max_target_length,
        pack_examples=config.pack_examples,
        drop_remainder=config.drop_remainder)

    eval_ds = preprocess_wmt_dataset(
        dataset=eval_data,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        max_length=config.max_eval_length,
        pack_examples=False,
        drop_remainder=config.drop_remainder
    )

    predict_ds = preprocess_wmt_dataset(
        dataset=eval_data,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        max_length=config.max_predict_length,
        pack_examples=False,
        drop_remainder=config.drop_remainder
    )

    return train_ds, eval_ds, predict_ds, input_lang_tokenizer, target_lang_tokenizer


# config = ml_collections.ConfigDict
# config.ds_keys = ['inputs', 'targets']
# config.eval_dataset = None
# config.max_target_length = 250
# config.max_eval_length = 250
# config.max_predict_length = 250
# config.eval_split = 'test'
# config.download_dir = '/home/mani/tensorflow_datasets'
# config.input_lang_vocab = './input_vocab.txt'
# config.target_lang_vocab = './target_vocab.txt'
# config.vocab_size = 8000
# config.reversed_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
# config.bert_tokenizer_params = dict(lower_case=True)
# config.learn_params = {}
# config.lower_case = True
# config.dataset_name = 'ted_hrlr_translate/pt_to_en'
# config.batch_size = 32
# config.num_epochs = 10
# config.train_split = 'train'
# config.shuffle = False
# config.shuffle_buffer_size = 10_000
# config.drop_remainder = False
# config.reverse_translation = False
# config.pack_examples = True
# config.ds_keys = ['inputs', 'targets']

