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

from collections.abc import Callable
from collections.abc import Iterator
import contextlib
import dataclasses
import functools
import math
import os
import pathlib
import random
import signal
import threading
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
    'batch_size', 32, 'Number of sequences in each training batch.'
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
    'training_steps',
    1_000_000,
    'Maximum number of parameter-update steps.',
)
_CONVERGENCE_PATIENCE = flags.DEFINE_integer(
    'convergence_patience',
    10,
    'Stop early after this many fixed-validation checks without sufficient '
    'improvement; 0 disables convergence stopping.',
)
_CONVERGENCE_MIN_DELTA = flags.DEFINE_float(
    'convergence_min_delta',
    1e-3,
    'Minimum cumulative decrease in validation nats/byte required to reset '
    'convergence patience.',
)
_CONVERGENCE_CHECK_INTERVAL = flags.DEFINE_float(
    'convergence_check_interval',
    0.25,
    'Number of Enwik8-equivalent sampled-data passes between fixed-validation '
    'checks.',
)
_CONVERGENCE_MIN_PASSES = flags.DEFINE_float(
    'convergence_min_passes',
    5.0,
    'Minimum number of Enwik8-equivalent sampled-data passes before the first '
    'fixed-validation check.',
)
_CONVERGENCE_VALIDATION_CHUNKS = flags.DEFINE_integer(
    'convergence_validation_chunks',
    512,
    'Number of deterministic, stratified Enwik8 chunks reserved for '
    'convergence validation.',
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
_POSITIONAL_ENCODING = flags.DEFINE_enum(
    'positional_encoding',
    transformer.ROTARY_POSITION_ENCODING,
    transformer.POSITIONAL_ENCODINGS,
    'Position encoding to use; rotary matches the paper, while sinusoidal '
    'reproduces checkpoints trained with the released implementation.',
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
    512,
    'Query/key tile size for exact blockwise attention; 0 uses dense '
    'attention.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', 'params.npz', 'Path for the trained model checkpoint.'
)
_USE_TQDM = flags.DEFINE_bool(
    'use_tqdm', True, 'Show a progress bar during training.'
)


@dataclasses.dataclass
class _InterruptState:
  """Records whether the CLI received a graceful interrupt request."""

  requested: bool = False


class _TrainingInterrupted(Exception):
  """Carries the last safely completed training state to the CLI."""

  def __init__(
      self,
      params: hk.Params | None,
      loss: Any | None,
      completed_steps: int,
  ):
    super().__init__('Training interrupted at a safe update boundary.')
    self.params = params
    self.loss = loss
    self.completed_steps = completed_steps


@contextlib.contextmanager
def _graceful_sigint() -> Iterator[_InterruptState]:
  """Makes the first SIGINT cooperative and the second one immediate."""
  state = _InterruptState()
  if threading.current_thread() is not threading.main_thread():
    yield state
    return

  def request_stop(_signal_number: int, _frame: Any) -> None:
    state.requested = True
    # Once the first request has reached Python, a second Ctrl-C must also work
    # if execution returns to an uninterruptible backend call.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

  previous_handler = signal.signal(signal.SIGINT, request_stop)
  try:
    yield state
  finally:
    signal.signal(signal.SIGINT, previous_handler)


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
    positional_encoding: str = transformer.ROTARY_POSITION_ENCODING,
    num_layers: int | None = None,
    num_heads: int | None = None,
    widening_factor: int | None = None,
    attention_block_size: int | None = 512,
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
  return dataclasses.replace(
      config,
      positional_encoding=positional_encoding,
      attention_block_size=attention_block_size,
  )


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
    convergence_patience: int = 0,
    convergence_min_delta: float = 1e-3,
    convergence_check_interval: float = 0.25,
    convergence_min_passes: float = 5.0,
    convergence_validation_chunks: int = 512,
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
  if convergence_patience < 0:
    raise ValueError(
        'convergence_patience must be non-negative; got '
        f'{convergence_patience}.'
    )
  if convergence_patience > 0:
    convergence_finite_values = {
        'convergence_min_delta': convergence_min_delta,
        'convergence_check_interval': convergence_check_interval,
        'convergence_min_passes': convergence_min_passes,
    }
    for name, value in convergence_finite_values.items():
      if not math.isfinite(value):
        raise ValueError(f'{name} must be finite; got {value}.')
    if convergence_min_delta < 0:
      raise ValueError(
          'convergence_min_delta must be non-negative; got '
          f'{convergence_min_delta}.'
      )
    if convergence_check_interval <= 0:
      raise ValueError(
          'convergence_check_interval must be positive; got '
          f'{convergence_check_interval}.'
      )
    if convergence_min_passes < 0:
      raise ValueError(
          'convergence_min_passes must be non-negative; got '
          f'{convergence_min_passes}.'
      )
    if convergence_validation_chunks <= 0:
      raise ValueError(
          'convergence_validation_chunks must be positive; got '
          f'{convergence_validation_chunks}.'
      )
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
  training_chunk_count = constants.ENWIK8_SIZE_BYTES // sequence_length
  if (
      convergence_patience > 0
      and convergence_validation_chunks >= training_chunk_count
  ):
    raise ValueError(
        'convergence_validation_chunks must leave at least one Enwik8 chunk '
        'for training; got '
        f'{convergence_validation_chunks} validation chunks out of '
        f'{training_chunk_count}.'
    )
  if config.embedding_dim % config.num_heads:
    raise ValueError(
        'embedding_dim must be divisible by num_heads; got '
        f'{config.embedding_dim} and {config.num_heads}.'
    )
  if config.positional_encoding not in transformer.POSITIONAL_ENCODINGS:
    valid_encodings = ', '.join(transformer.POSITIONAL_ENCODINGS)
    raise ValueError(
        f'Unknown positional encoding {config.positional_encoding!r}; '
        f'expected one of {valid_encodings}.'
    )
  head_dimension = config.embedding_dim // config.num_heads
  if (
      config.positional_encoding == transformer.ROTARY_POSITION_ENCODING
      and head_dimension % 2
  ):
    raise ValueError(
        'Rotary positional encoding requires an even head dimension; got '
        f'embedding_dim={config.embedding_dim} and '
        f'num_heads={config.num_heads} (head dimension {head_dimension}).'
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


@dataclasses.dataclass
class _ConvergenceMonitor:
  """Tracks meaningful improvements on one fixed validation set.

  ``best_validation_loss`` is used to select the checkpoint.  The separate
  ``reference_validation_loss`` is only advanced by ``min_delta`` so that a
  series of individually small improvements can accumulate and reset
  patience.  Keeping these roles separate also ensures that the lowest-loss
  parameters are retained even when the decrease is smaller than
  ``min_delta``.
  """

  patience: int
  min_delta: float
  best_validation_loss: float = math.inf
  best_step: int = 0
  reference_validation_loss: float = math.inf
  checks_without_improvement: int = 0

  def update(self, validation_loss: float, step: int) -> bool:
    """Adds a validation loss and reports whether patience is exhausted."""
    if not math.isfinite(validation_loss):
      raise ValueError(
          f'Validation loss must be finite; got {validation_loss}.'
      )

    if validation_loss < self.best_validation_loss:
      self.best_validation_loss = validation_loss
      self.best_step = step

    if self.reference_validation_loss == math.inf:
      self.reference_validation_loss = validation_loss
      self.checks_without_improvement = 0
      return False

    improvement = self.reference_validation_loss - validation_loss
    if validation_loss < self.reference_validation_loss and (
        improvement >= self.min_delta
    ):
      self.reference_validation_loss = validation_loss
      self.checks_without_improvement = 0
    else:
      self.checks_without_improvement += 1

    return self.checks_without_improvement >= self.patience


def _stratified_validation_indices(
    total_chunks: int, validation_chunks: int
) -> tuple[int, ...]:
  """Returns deterministic midpoint indices from equal-width corpus strata."""
  if total_chunks <= 1:
    raise ValueError(
        f'total_chunks must be greater than one; got {total_chunks}.'
    )
  if not 0 < validation_chunks < total_chunks:
    raise ValueError(
        'validation_chunks must be between one and total_chunks - 1; got '
        f'{validation_chunks} out of {total_chunks}.'
    )
  return tuple(
      ((2 * index + 1) * total_chunks) // (2 * validation_chunks)
      for index in range(validation_chunks)
  )


def _convergence_schedule(
    *,
    batch_size: int,
    sequence_length: int,
    min_passes: float,
    check_interval: float,
) -> tuple[int, int]:
  """Converts data-equivalent passes to update counts for this batch size."""
  bytes_per_update = batch_size * sequence_length
  first_check_step = max(
      1,
      math.ceil(
          min_passes * constants.ENWIK8_SIZE_BYTES / bytes_per_update
      ),
  )
  check_interval_steps = max(
      1,
      math.ceil(
          check_interval * constants.ENWIK8_SIZE_BYTES / bytes_per_update
      ),
  )
  return first_check_step, check_interval_steps


def _copy_params_to_host(params: hk.Params) -> hk.Params:
  """Copies donated JAX parameter buffers into an independent host snapshot."""
  return tree.map_structure(
      lambda value: np.array(jax.device_get(value), copy=True), params
  )


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


def _make_per_sequence_loss_fn(model: hk.Transformed) -> Any:
  """Returns a function producing one summed negative log-loss per sequence."""

  def per_sequence_loss_fn(
      params: hk.Params,
      sequences: jax.Array,
  ) -> jax.Array:
    conditionals = model.apply(
        params=params,
        targets=sequences,
        rng=None,
    )
    marginals = _to_marginals(conditionals, sequences)
    return -marginals

  return per_sequence_loss_fn


def _make_loss_fn(model: hk.Transformed) -> Any:
  """Returns the mean batch loss function for parameter updates."""
  per_sequence_loss_fn = _make_per_sequence_loss_fn(model)

  def loss_fn(
      params: hk.Params,
      sequences: jax.Array,
  ) -> jnp.float32:
    return jnp.mean(per_sequence_loss_fn(params, sequences))

  return loss_fn


def _evaluate_validation_losses(
    params: hk.Params,
    validation_data: np.ndarray,
    per_sequence_loss_fn: Any,
    batch_size: int,
    sequence_length: int,
) -> np.ndarray:
  """Evaluates fixed validation data in bounded-memory microbatches."""
  losses = []
  for start in range(0, len(validation_data), batch_size):
    batch = validation_data[start : start + batch_size]
    batch_losses = jax.device_get(per_sequence_loss_fn(params, batch))
    losses.append(np.asarray(batch_losses, dtype=np.float64))
  normalized_losses = np.concatenate(losses) / sequence_length
  if not np.all(np.isfinite(normalized_losses)):
    raise ValueError('Validation loss must be finite.')
  return normalized_losses


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
    batch_size: int = 32,
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
    convergence_patience: int = 0,
    convergence_min_delta: float = 1e-3,
    convergence_check_interval: float = 0.25,
    convergence_min_passes: float = 5.0,
    convergence_validation_chunks: int = 512,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[hk.Params, float]:
  """Trains a language model on Enwik8 data.

  Fixed-length byte sequences are extracted from Enwik8, and then randomly
  sampled. We train a decoder-only transformer on batches, minimizing the
  log-loss objective. The exact architecture can be modified using the
  TransformerConfig object (defined in transformer.py)

  Args:
    training_steps: Maximum number of batches to train on.
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
    convergence_patience: Stop after this many fixed-validation checks without
      a sufficient improvement. Set to 0 to disable convergence stopping.
    convergence_min_delta: Minimum nats/byte decrease that counts as an
      improvement for convergence stopping.
    convergence_check_interval: Number of Enwik8-equivalent sampled-data
      passes between validation checks.
    convergence_min_passes: Minimum number of Enwik8-equivalent sampled-data
      passes before establishing the validation baseline.
    convergence_validation_chunks: Number of deterministic, stratified Enwik8
      chunks held out from training for validation.
    stop_requested: Optional cooperative-stop predicate. When it becomes true,
      training stops only after the active donated-buffer update is safe to
      checkpoint.

  Returns:
    The selected parameters and their summed fixed-validation loss when
    convergence stopping is enabled. Otherwise, the final parameters and the
    last pre-update training-batch loss are returned.

  Raises:
    _TrainingInterrupted: The cooperative stop predicate requested a stop. The
      exception carries the last state known to be safe for checkpointing.
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
      convergence_patience=convergence_patience,
      convergence_min_delta=convergence_min_delta,
      convergence_check_interval=convergence_check_interval,
      convergence_min_passes=convergence_min_passes,
      convergence_validation_chunks=convergence_validation_chunks,
  )
  expected_parameter_count = transformer.parameter_count(config)
  logging.info(
      'Model: %s parameters, embedding_dim=%d, num_layers=%d, '
      'num_heads=%d, widening_factor=%d, positional_encoding=%s, '
      'attention=%s',
      f'{expected_parameter_count:,}',
      config.embedding_dim,
      config.num_layers,
      config.num_heads,
      config.widening_factor,
      config.positional_encoding,
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
  convergence_enabled = convergence_patience > 0
  if convergence_enabled:
    first_validation_step, validation_interval_steps = _convergence_schedule(
        batch_size=batch_size,
        sequence_length=sequence_length,
        min_passes=convergence_min_passes,
        check_interval=convergence_check_interval,
    )
    logging.info(
        'Convergence stopping: fixed validation=%d chunks, first check=%d '
        'steps (%.3f Enwik8-equivalent sampled-byte passes), interval=%d '
        'steps (%.3f passes), '
        'patience=%d checks, min_delta=%g nats/byte, maximum_steps=%d',
        convergence_validation_chunks,
        first_validation_step,
        first_validation_step * batch_size * sequence_length
        / constants.ENWIK8_SIZE_BYTES,
        validation_interval_steps,
        validation_interval_steps * batch_size * sequence_length
        / constants.ENWIK8_SIZE_BYTES,
        convergence_patience,
        convergence_min_delta,
        training_steps,
    )
    earliest_stop_step = (
        first_validation_step
        + convergence_patience * validation_interval_steps
    )
    if earliest_stop_step > training_steps:
      logging.warning(
          'The %d-step safety cap is earlier than the first possible '
          'pre-cap convergence stop at step %d; increase training_steps if '
          'automatic stopping should control this run.',
          training_steps,
          earliest_stop_step,
      )
  else:
    first_validation_step = 0
    validation_interval_steps = 0
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )

  training_chunk_count = constants.ENWIK8_SIZE_BYTES // sequence_length
  data_generator = data_loaders.get_enwik9_iterator(
      num_chunks=training_chunk_count,
      sequence_length=sequence_length,
  )
  enwik8_chunks = list(data_generator)
  if not enwik8_chunks:
    raise ValueError('The Enwik8 training dataset did not contain any chunks.')
  if any(len(chunk) != sequence_length for chunk in enwik8_chunks):
    raise ValueError('Enwik8 contained an incomplete training chunk.')
  if convergence_enabled and len(enwik8_chunks) != training_chunk_count:
    raise ValueError(
        'Enwik8 did not contain the expected number of complete training '
        'chunks.'
    )

  validation_data = None
  if convergence_enabled:
    validation_indices = _stratified_validation_indices(
        training_chunk_count, convergence_validation_chunks
    )
    validation_index_set = set(validation_indices)
    validation_data = np.stack(
        [
            np.frombuffer(enwik8_chunks[index], dtype=np.uint8)
            for index in validation_indices
        ]
    )
    dataset = [
        chunk
        for index, chunk in enumerate(enwik8_chunks)
        if index not in validation_index_set
    ]
  else:
    dataset = enwik8_chunks

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
  validation_loss_fn = (
      jax.jit(_make_per_sequence_loss_fn(model))
      if convergence_enabled
      else None
  )

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
  convergence_monitor = (
      _ConvergenceMonitor(
          patience=convergence_patience,
          min_delta=convergence_min_delta,
      )
      if convergence_enabled
      else None
  )
  next_validation_step = first_validation_step
  last_validation_step = 0
  best_params = None
  last_loss = 0.0
  stopped_for_convergence = False
  converged_at_cap = False

  progress_bar = tqdm.trange(training_steps, disable=not use_tqdm)

  def log_progress(message: str, *args: Any) -> None:
    if use_tqdm:
      progress_bar.write(message % args, file=progress_bar.fp)
    else:
      logging.info(message, *args, stacklevel=2)

  def run_validation(current_params: hk.Params, completed_steps: int) -> bool:
    """Evaluates and records one checkpoint on the fixed holdout."""
    nonlocal best_params, last_validation_step
    assert convergence_monitor is not None
    assert validation_data is not None
    assert validation_loss_fn is not None
    validation_losses = _evaluate_validation_losses(
        params=current_params,
        validation_data=validation_data,
        per_sequence_loss_fn=validation_loss_fn,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    validation_mean = float(np.mean(validation_losses))
    validation_chunk_sd = (
        float(np.std(validation_losses, ddof=1))
        if len(validation_losses) > 1
        else 0.0
    )
    previous_best = convergence_monitor.best_validation_loss
    converged = convergence_monitor.update(validation_mean, completed_steps)
    if validation_mean < previous_best:
      best_params = _copy_params_to_host(current_params)
    last_validation_step = completed_steps
    log_progress(
        'Validation after %d steps (%.3f Enwik8-equivalent sampled-byte '
        'passes): nats/byte %.6f (chunk SD %.6f), best %.6f at step %d, '
        'checks without meaningful improvement %d/%d',
        completed_steps,
        completed_steps * batch_size * sequence_length
        / constants.ENWIK8_SIZE_BYTES,
        validation_mean,
        validation_chunk_sd,
        convergence_monitor.best_validation_loss,
        convergence_monitor.best_step,
        convergence_monitor.checks_without_improvement,
        convergence_patience,
    )
    return converged

  completed_steps = 0

  def raise_if_interrupted() -> None:
    if stop_requested is None or not stop_requested():
      return
    if completed_steps == 0:
      log_progress(
          'Interrupt received before the first training step completed.'
      )
      raise _TrainingInterrupted(None, None, completed_steps)

    # The newly assigned values, rather than the donated inputs from the
    # previous step, are the only state that is safe to preserve.
    jax.block_until_ready((params, opt_state, last_loss))
    log_progress(
        'Interrupt received at a safe boundary (completed steps: %d). '
        'Press Ctrl-C again to abort immediately.',
        completed_steps,
    )
    raise _TrainingInterrupted(params, last_loss, completed_steps)

  try:
    for step in progress_bar:
      raise_if_interrupted()
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
      last_loss = logs['loss']
      completed_steps = step + 1
      raise_if_interrupted()

      if should_log:
        nats_per_byte, bits_per_byte, perplexity = _loss_statistics(
            logs['loss'], sequence_length
        )
        log_progress(
            'Step %d, loss %.6f, nats/byte %.6f, bits/byte %.6f, '
            'perplexity %.3f, grad norm %.6f',
            step,
            logs['loss'],
            nats_per_byte,
            bits_per_byte,
            perplexity,
            logs['grad_norm_unclipped'],
        )

      if (
          convergence_monitor is not None
          and completed_steps == next_validation_step
      ):
        converged = run_validation(params, completed_steps)
        next_validation_step += validation_interval_steps
        if converged:
          if completed_steps == training_steps:
            converged_at_cap = True
          else:
            log_progress(
                'Stopping early after %d steps because fixed validation loss '
                'did not improve by at least %g nats/byte for %d checks.',
                completed_steps,
                convergence_min_delta,
                convergence_patience,
            )
            stopped_for_convergence = True
            break

    raise_if_interrupted()
  finally:
    progress_bar.close()

  if convergence_monitor is not None:
    if not stopped_for_convergence:
      if last_validation_step != training_steps:
        converged_at_cap = run_validation(params, training_steps)
      if converged_at_cap:
        logging.info(
            'Reached the maximum of %d training steps as validation patience '
            'was exhausted.',
            training_steps,
        )
      else:
        logging.info(
            'Reached the maximum of %d training steps before convergence.',
            training_steps,
        )
    if best_params is not None:
      params = best_params
      last_loss = convergence_monitor.best_validation_loss * sequence_length
      logging.info(
          'Selected best fixed-validation checkpoint from step %d '
          '(%.6f nats/byte).',
          convergence_monitor.best_step,
          convergence_monitor.best_validation_loss,
      )

  return params, float(jax.device_get(last_loss))


def main(argv: list[str]) -> None:
  """Trains a language model and saves its checkpoint."""
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected positional arguments: {argv[1:]}')

  config = _resolve_model_config(
      model_size=_MODEL_SIZE.value,
      embedding_dim=_EMBEDDING_DIM.value,
      positional_encoding=_POSITIONAL_ENCODING.value,
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
        convergence_patience=_CONVERGENCE_PATIENCE.value,
        convergence_min_delta=_CONVERGENCE_MIN_DELTA.value,
        convergence_check_interval=_CONVERGENCE_CHECK_INTERVAL.value,
        convergence_min_passes=_CONVERGENCE_MIN_PASSES.value,
        convergence_validation_chunks=_CONVERGENCE_VALIDATION_CHUNKS.value,
    )
  except ValueError as exc:
    raise app.UsageError(str(exc)) from exc
  if not _OUTPUT_PATH.value:
    raise app.UsageError('output_path must not be empty.')
  try:
    _validate_output_path(_OUTPUT_PATH.value)
  except ValueError as exc:
    raise app.UsageError(str(exc)) from exc

  interrupted = False
  with _graceful_sigint() as interrupt_state:
    try:
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
          convergence_patience=_CONVERGENCE_PATIENCE.value,
          convergence_min_delta=_CONVERGENCE_MIN_DELTA.value,
          convergence_check_interval=_CONVERGENCE_CHECK_INTERVAL.value,
          convergence_min_passes=_CONVERGENCE_MIN_PASSES.value,
          convergence_validation_chunks=_CONVERGENCE_VALIDATION_CHUNKS.value,
          use_tqdm=_USE_TQDM.value,
          stop_requested=lambda: interrupt_state.requested,
      )
    except _TrainingInterrupted as exc:
      interrupted = True
      params = exc.params
      loss = exc.loss

    if params is None:
      logging.warning(
          'No training step completed; the existing checkpoint was left '
          'unchanged.'
      )
    else:
      if loss is None:
        raise RuntimeError('Completed training state did not include a loss.')
      params, loss = jax.device_get((params, loss))
      loss = float(loss)
      interrupted = interrupted or interrupt_state.requested

      nats_per_byte, bits_per_byte, perplexity = _loss_statistics(
          loss, _SEQUENCE_LENGTH.value
      )
      logging.info(
          '%s loss: %.6f, nats/byte: %.6f, bits/byte: %.6f, '
          'perplexity: %.3f',
          (
              'Last completed'
              if interrupted
              else (
                  'Best fixed-validation checkpoint'
                  if _CONVERGENCE_PATIENCE.value > 0
                  else 'Last pre-update training batch'
              )
          ),
          loss,
          nats_per_byte,
          bits_per_byte,
          perplexity,
      )

      model_checkpoint.save(_OUTPUT_PATH.value, params, config)
      logging.info('Model checkpoint saved to %s', _OUTPUT_PATH.value)

    interrupted = interrupted or interrupt_state.requested

  if interrupted or interrupt_state.requested:
    raise SystemExit(128 + signal.SIGINT)


if __name__ == '__main__':
  app.run(main)
