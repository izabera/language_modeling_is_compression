# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a language model on the Enwik8 dataset."""

import dataclasses
import functools
import math
import os
import pathlib
import random
from typing import Any

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree

from language_modeling_is_compression import constants
from language_modeling_is_compression import data_loaders
from language_modeling_is_compression import model_checkpoint
from language_modeling_is_compression import transformer


_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 1, 'Number of sequences in each training batch.'
)
_SEED = flags.DEFINE_integer(
    'seed', 0, 'Random seed for parameter initialization and batch sampling.'
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-4, 'Optimizer learning rate.'
)
_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    'adam',
    ('adam', 'adamw', 'sgd', 'rmsprop'),
    'Optimizer to use.',
)
_MOMENTUM = flags.DEFINE_float(
    'momentum',
    0.9,
    'Momentum for SGD and RMSProp; ignored by Adam and AdamW.',
)
_WEIGHT_DECAY = flags.DEFINE_float(
    'weight_decay',
    1e-4,
    'Weight decay for AdamW; ignored by other optimizers.',
)
_GRADIENT_CLIP_NORM = flags.DEFINE_float(
    'gradient_clip_norm',
    0.0,
    'Clip gradients to this global norm; 0 disables clipping.',
)
_NORMALIZE_GRADIENTS = flags.DEFINE_bool(
    'normalize_gradients',
    True,
    'Normalize gradients by sequence length before applying updates.',
)
_TRAINING_STEPS = flags.DEFINE_integer(
    'training_steps', 100, 'Number of parameter-update steps.'
)
_LOG_EVERY = flags.DEFINE_integer(
    'log_every', 10, 'Log metrics every N steps; 0 disables step logging.'
)
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length',
    constants.CHUNK_SIZE_BYTES,
    'Training sequence length in bytes.',
)
_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '200k',
    transformer.MODEL_SIZE_PRESETS.keys(),
    'Rounded parameter-count preset for the Transformer.',
)
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim',
    None,
    'Embedding width override; by default the model-size preset is used.',
)
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers',
    None,
    'Transformer layer-count override; by default the preset is used.',
)
_NUM_HEADS = flags.DEFINE_integer(
    'num_heads',
    None,
    'Attention-head-count override; by default the preset is used.',
)
_WIDENING_FACTOR = flags.DEFINE_integer(
    'widening_factor',
    None,
    'Feed-forward widening override; by default the preset is used.',
)
_ATTENTION_BLOCK_SIZE = flags.DEFINE_integer(
    'attention_block_size',
    256,
    'Query/key tile size for exact blockwise attention; 0 uses dense '
    'attention.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', 'params.npz', 'Path for the trained model checkpoint.'
)
_USE_TQDM = flags.DEFINE_bool(
    'use_tqdm', True, 'Show a progress bar during training.'
)


def _make_optimizer(
    name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> optax.GradientTransformation:
  """Constructs an optimizer from command-line-compatible settings."""
  if name == 'adam':
    optimizer = optax.adam(learning_rate)
  elif name == 'adamw':
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
  elif name == 'sgd':
    optimizer = optax.sgd(learning_rate, momentum=momentum)
  elif name == 'rmsprop':
    optimizer = optax.rmsprop(learning_rate, momentum=momentum)
  else:
    raise ValueError(f'Unknown optimizer {name!r}.')

  if gradient_clip_norm > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm), optimizer
    )
  return optimizer


def _resolve_model_config(
    model_size: str,
    embedding_dim: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    widening_factor: int | None = None,
    attention_block_size: int | None = 256,
) -> transformer.TransformerConfig:
  """Resolves a model-size preset with optional architecture overrides."""
  config = transformer.config_for_model_size(
      model_size=model_size,
      vocab_size=constants.ALPHABET_SIZE,
  )
  overrides = {
      'embedding_dim': embedding_dim,
      'num_layers': num_layers,
      'num_heads': num_heads,
      'widening_factor': widening_factor,
  }
  config = dataclasses.replace(
      config,
      **{name: value for name, value in overrides.items() if value is not None},
  )
  return dataclasses.replace(config, attention_block_size=attention_block_size)


def _validate_training_arguments(
    *,
    training_steps: int,
    log_every: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    optimizer_name: str,
    seed: int,
    momentum: float,
    weight_decay: float,
    gradient_clip_norm: float,
    config: transformer.TransformerConfig,
) -> None:
  """Validates training and architecture settings."""
  positive_values = {
      'training_steps': training_steps,
      'batch_size': batch_size,
      'sequence_length': sequence_length,
      'learning_rate': learning_rate,
      'embedding_dim': config.embedding_dim,
      'num_layers': config.num_layers,
      'num_heads': config.num_heads,
      'widening_factor': config.widening_factor,
  }
  for name, value in positive_values.items():
    if value <= 0:
      raise ValueError(f'{name} must be positive; got {value}.')
  finite_values = {
      'learning_rate': learning_rate,
      'momentum': momentum,
      'weight_decay': weight_decay,
      'gradient_clip_norm': gradient_clip_norm,
  }
  for name, value in finite_values.items():
    if not math.isfinite(value):
      raise ValueError(f'{name} must be finite; got {value}.')
  if log_every < 0:
    raise ValueError(f'log_every must be non-negative; got {log_every}.')
  if optimizer_name not in ('adam', 'adamw', 'sgd', 'rmsprop'):
    raise ValueError(f'Unknown optimizer {optimizer_name!r}.')
  if seed < 0:
    raise ValueError(f'seed must be non-negative; got {seed}.')
  if seed > 2**32 - 1:
    raise ValueError(f'seed must fit in 32 bits; got {seed}.')
  if not 0 <= momentum < 1:
    raise ValueError(f'momentum must be in [0, 1); got {momentum}.')
  if weight_decay < 0:
    raise ValueError(
        f'weight_decay must be non-negative; got {weight_decay}.'
    )
  if gradient_clip_norm < 0:
    raise ValueError(
        'gradient_clip_norm must be non-negative; got '
        f'{gradient_clip_norm}.'
    )
  if sequence_length > constants.ENWIK8_SIZE_BYTES:
    raise ValueError(
        'sequence_length cannot exceed the Enwik8 training-set size '
        f'({constants.ENWIK8_SIZE_BYTES} bytes).'
    )
  if config.embedding_dim % config.num_heads:
    raise ValueError(
        'embedding_dim must be divisible by num_heads; got '
        f'{config.embedding_dim} and {config.num_heads}.'
    )
  if (
      config.attention_block_size is not None
      and config.attention_block_size <= 0
  ):
    raise ValueError(
        'attention_block_size must be positive or None; got '
        f'{config.attention_block_size}.'
    )


def _validate_output_path(path: str) -> None:
  """Checks obvious checkpoint-path errors before expensive training."""
  checkpoint_path = pathlib.Path(path)
  parent = checkpoint_path.parent
  if not parent.is_dir():
    raise ValueError(f'Output directory does not exist: {str(parent)!r}.')
  if checkpoint_path.exists() and not checkpoint_path.is_file():
    raise ValueError(f'output_path is not a file: {path!r}.')
  writable_path = checkpoint_path if checkpoint_path.exists() else parent
  if not os.access(writable_path, os.W_OK):
    raise ValueError(f'output_path is not writable: {path!r}.')


def _loss_statistics(
    loss: Any, sequence_length: int
) -> tuple[float, float, float]:
  """Converts summed sequence loss to normalized compression metrics."""
  nats_per_byte = float(jax.device_get(loss)) / sequence_length
  bits_per_byte = nats_per_byte / math.log(2)
  try:
    perplexity = math.exp(nats_per_byte)
  except OverflowError:
    perplexity = math.inf
  return nats_per_byte, bits_per_byte, perplexity


def _to_marginals(
    predictions: jax.Array,
    sequences: jax.Array,
) -> jax.Array:
  """Converts a conditional array to a marginals array."""
  true_predictions = jnp.take_along_axis(
      predictions, sequences[..., None], axis=-1
  )
  true_predictions = true_predictions[..., 0]  # Shape (B, T).
  return jnp.sum(true_predictions, axis=1)  # Shape (B,).


def _make_loss_fn(model: hk.Transformed) -> Any:
  """Returns the loss function for update_parameters."""

  def loss_fn(
      params: hk.Params,
      sequences: jax.Array,
  ) -> jnp.float32:
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      sequences: The input of sequences to evaluate. See neural_predictors.py.
    """
    conditionals = model.apply(
        params=params,
        targets=sequences,
        rng=None,
    )
    marginals = _to_marginals(conditionals, sequences)
    return -jnp.mean(marginals)

  return loss_fn


def _initialize_parameters(model: hk.Transformed, seed: int) -> hk.Params:
  """Initializes shape-independent Transformer parameters cheaply.

  None of the model's parameter shapes depend on the input batch or sequence
  dimensions. Tracing initialization with one token therefore creates the same
  parameter tree without running attention over the full training batch.
  """
  dummy_sequence = np.zeros((1, 1), dtype=np.uint8)
  return model.init(jax.random.PRNGKey(seed), dummy_sequence)


@functools.partial(
    jax.jit,
    static_argnames=('optimizer', 'grad_fn', 'normalize_gradients'),
    donate_argnums=(0, 1),
)
def _update_parameters(
    params: hk.Params,
    opt_state: optax.OptState,
    sequences: jax.Array,
    grad_fn: Any,
    optimizer: optax.GradientTransformation,
    normalize_gradients: bool = True,
    compute_grad_norm: bool = True,
) -> tuple[hk.Params, optax.OptState, dict[str, Any]]:
  """Returns updated params and extra logs (like loss, last state etc).

  Backpropagation is done on the whole sequence. The whole function is jitted.

  Args:
    params: The current parameters of the network.
    opt_state: The optimizer state.
    sequences: The input of sequences to evaluate. See base_predictor.py.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the
      predictor. It returns the gradient of the parameters for this batch of
      data, and extra values.
    optimizer: An optax optimizer.
    normalize_gradients: Whether to divide the gradients by the length of the
      sequences, or keep them as is. Using this option guarantees to have the
      same scale across various sequence lengths, and therefore tasks.
    compute_grad_norm: Whether the returned gradient norm is evaluated. This
      is a dynamic predicate, so logging and non-logging steps share one
      compiled update. The unused value is NaN on non-logging steps.

  Note:
    ``params`` and ``opt_state`` are donated to the compiled update. Callers
    must use the returned trees and must not reuse the input buffers.
  """
  loss, grad = grad_fn(params, sequences)
  if normalize_gradients:
    length_sequence = float(sequences.shape[1])
    grad = tree.map_structure(lambda x: x / length_sequence, grad)
  updates, new_opt_state = optimizer.update(grad, opt_state, params)
  new_params = optax.apply_updates(params, updates)

  grad_norm = jax.lax.cond(
      compute_grad_norm,
      lambda _: optax.global_norm(grad),
      lambda _: jnp.asarray(jnp.nan, dtype=loss.dtype),
      operand=None,
  )
  log_dict = {
      'loss': loss,
      'grad_norm_unclipped': grad_norm,
  }

  return new_params, new_opt_state, log_dict


def train_transformer_decoder(
    training_steps: int,
    log_every: int,
    batch_size: int = 128,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
    use_tqdm: bool = True,
    *,
    model_config: transformer.TransformerConfig | None = None,
    seed: int = 0,
    learning_rate: float = 1e-4,
    optimizer_name: str = 'adam',
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 0.0,
    normalize_gradients: bool = True,
) -> tuple[hk.Params, float]:
  """Trains a language model on Enwik8 data.

  Fixed-length byte sequences are extracted from Enwik8, and then randomly
  sampled. We train a decoder-only transformer on batches, minimizing the
  log-loss objective. The exact architecture can be modified using the
  TransformerConfig object (defined in transformer.py)

  Args:
    training_steps: Number of batches to train on.
    log_every: How often to log the loss. Set to 0 to disable step logging.
    batch_size: The number of sequences in a batch.
    sequence_length: The length of the sequences to train on, in number of ASCII
      characters.
    use_tqdm: Whether to use a progress bar or not.
    model_config: Transformer architecture to train.
    seed: Random seed for parameter initialization and batch sampling.
    learning_rate: Optimizer learning rate.
    optimizer_name: One of adam, adamw, sgd, or rmsprop.
    momentum: Momentum used by SGD and RMSProp.
    weight_decay: Weight decay used by AdamW.
    gradient_clip_norm: Global gradient clipping norm, or 0 to disable it.
    normalize_gradients: Whether to normalize gradients by sequence length.

  Returns:
    The final parameters and loss.
  """
  config = model_config or transformer.TransformerConfig(
      vocab_size=constants.ALPHABET_SIZE
  )
  _validate_training_arguments(
      training_steps=training_steps,
      log_every=log_every,
      batch_size=batch_size,
      sequence_length=sequence_length,
      learning_rate=learning_rate,
      optimizer_name=optimizer_name,
      seed=seed,
      momentum=momentum,
      weight_decay=weight_decay,
      gradient_clip_norm=gradient_clip_norm,
      config=config,
  )
  expected_parameter_count = transformer.parameter_count(config)
  logging.info(
      'Model: %s parameters, embedding_dim=%d, num_layers=%d, '
      'num_heads=%d, widening_factor=%d, attention=%s',
      f'{expected_parameter_count:,}',
      config.embedding_dim,
      config.num_layers,
      config.num_heads,
      config.widening_factor,
      (
          f'blockwise-{config.attention_block_size}'
          if config.attention_block_size is not None
          else 'dense'
      ),
  )
  logging.info(
      'Training: optimizer=%s, learning_rate=%g, batch_size=%d, '
      'sequence_length=%d, seed=%d, gradient_clip_norm=%g',
      optimizer_name,
      learning_rate,
      batch_size,
      sequence_length,
      seed,
      gradient_clip_norm,
  )
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )

  data_generator = data_loaders.get_enwik9_iterator(
      num_chunks=constants.ENWIK8_SIZE_BYTES // sequence_length,
      sequence_length=sequence_length,
  )
  dataset = list(data_generator)
  if not dataset:
    raise ValueError('The Enwik8 training dataset did not contain any chunks.')

  batch_random = random.Random(seed)

  def fetch_random_batch() -> np.ndarray:
    batch_list = batch_random.choices(dataset, k=batch_size)
    batch_list = [np.frombuffer(seq, dtype=np.uint8) for seq in batch_list]
    return np.array(batch_list, dtype=np.uint8)

  # Initialize parameters without tracing a full training-sized attention
  # computation. Transformer parameter shapes do not depend on B or T.
  params = _initialize_parameters(model, seed)
  actual_parameter_count = hk.data_structures.tree_size(params)
  if actual_parameter_count != expected_parameter_count:
    raise ValueError(
        'Unexpected model parameter count: initialized '
        f'{actual_parameter_count:,}, expected {expected_parameter_count:,}.'
    )
  # Make gradient function.
  loss_fn = _make_loss_fn(model)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

  # Make optimizer, to apply the gradients.
  optimizer = _make_optimizer(
      name=optimizer_name,
      learning_rate=learning_rate,
      momentum=momentum,
      weight_decay=weight_decay,
      gradient_clip_norm=gradient_clip_norm,
  )
  opt_state = optimizer.init(params)

  logging.info('Initialization done, starting training...')
  last_loss = 0.0
  for step in tqdm.trange(training_steps, disable=not use_tqdm):
    batch = fetch_random_batch()
    should_log = log_every > 0 and step % log_every == 0

    params, opt_state, logs = _update_parameters(
        params=params,
        opt_state=opt_state,
        sequences=batch,
        grad_fn=grad_fn,
        optimizer=optimizer,
        normalize_gradients=normalize_gradients,
        compute_grad_norm=should_log,
    )
    if should_log:
      nats_per_byte, bits_per_byte, perplexity = _loss_statistics(
          logs['loss'], sequence_length
      )
      logging.info(
          'Step %d, loss %.6f, nats/byte %.6f, bits/byte %.6f, '
          'perplexity %.3f, grad norm %.6f',
          step,
          logs['loss'],
          nats_per_byte,
          bits_per_byte,
          perplexity,
          logs['grad_norm_unclipped'],
      )
    last_loss = logs['loss']

  return params, float(jax.device_get(last_loss))


def main(argv: list[str]) -> None:
  """Trains a language model and saves its checkpoint."""
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected positional arguments: {argv[1:]}')

  config = _resolve_model_config(
      model_size=_MODEL_SIZE.value,
      embedding_dim=_EMBEDDING_DIM.value,
      num_layers=_NUM_LAYERS.value,
      num_heads=_NUM_HEADS.value,
      widening_factor=_WIDENING_FACTOR.value,
      attention_block_size=_ATTENTION_BLOCK_SIZE.value or None,
  )
  try:
    _validate_training_arguments(
        training_steps=_TRAINING_STEPS.value,
        log_every=_LOG_EVERY.value,
        batch_size=_BATCH_SIZE.value,
        sequence_length=_SEQUENCE_LENGTH.value,
        learning_rate=_LEARNING_RATE.value,
        optimizer_name=_OPTIMIZER.value,
        seed=_SEED.value,
        momentum=_MOMENTUM.value,
        weight_decay=_WEIGHT_DECAY.value,
        gradient_clip_norm=_GRADIENT_CLIP_NORM.value,
        config=config,
    )
  except ValueError as exc:
    raise app.UsageError(str(exc)) from exc
  if not _OUTPUT_PATH.value:
    raise app.UsageError('output_path must not be empty.')
  try:
    _validate_output_path(_OUTPUT_PATH.value)
  except ValueError as exc:
    raise app.UsageError(str(exc)) from exc

  params, loss = train_transformer_decoder(
      training_steps=_TRAINING_STEPS.value,
      log_every=_LOG_EVERY.value,
      sequence_length=_SEQUENCE_LENGTH.value,
      batch_size=_BATCH_SIZE.value,
      model_config=config,
      seed=_SEED.value,
      learning_rate=_LEARNING_RATE.value,
      optimizer_name=_OPTIMIZER.value,
      momentum=_MOMENTUM.value,
      weight_decay=_WEIGHT_DECAY.value,
      gradient_clip_norm=_GRADIENT_CLIP_NORM.value,
      normalize_gradients=_NORMALIZE_GRADIENTS.value,
      use_tqdm=_USE_TQDM.value,
  )

  nats_per_byte, bits_per_byte, perplexity = _loss_statistics(
      loss, _SEQUENCE_LENGTH.value
  )
  logging.info(
      'Final loss: %.6f, nats/byte: %.6f, bits/byte: %.6f, '
      'perplexity: %.3f',
      loss,
      nats_per_byte,
      bits_per_byte,
      perplexity,
  )

  model_checkpoint.save(_OUTPUT_PATH.value, params, config)
  logging.info('Model checkpoint saved to %s', _OUTPUT_PATH.value)


if __name__ == '__main__':
  app.run(main)
