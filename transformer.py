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

"""Transformer model."""

from collections.abc import Mapping
import dataclasses

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


ROTARY_POSITION_ENCODING = 'rotary'
SINUSOIDAL_POSITION_ENCODING = 'sinusoidal'
POSITIONAL_ENCODINGS = (
    ROTARY_POSITION_ENCODING,
    SINUSOIDAL_POSITION_ENCODING,
)


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # Vocabulary size.
  vocab_size: int
  # The dimension of the first embedding.
  embedding_dim: int = 64
  # Number of adjacent history bytes concatenated into each attention token.
  # Following TRACE, each byte is embedded at `embedding_dim / group_size`, so
  # concatenation preserves the Transformer's hidden width. Target-aligned
  # grouping phases keep every output strictly autoregressive.
  byte_group_size: int = 4
  # The number of multi-head attention layers.
  num_layers: int = 4
  # The number of heads per layer.
  num_heads: int = 8
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # How positions are represented. The paper uses rotary encodings; the
  # sinusoidal option is retained for checkpoints produced by the released
  # implementation.
  positional_encoding: str = ROTARY_POSITION_ENCODING
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4
  # Size of the query and key tiles used by the exact blockwise attention
  # implementation, measured in byte-equivalent positions. The grouped
  # decoder divides this by `byte_group_size`. Set to None for dense attention.
  attention_block_size: int | None = 256


# The paper reports rounded parameter counts rather than complete architecture
# definitions. These presets preserve the released implementation's default of
# eight attention heads; head count does not affect parameter count. The 6.4M
# preset naturally extends the 3.2M model from four to eight layers. The 38M
# preset continues the family's width doubling (64 -> 128 -> 256 -> 512) at
# twelve layers: its non-embedding parameter count (37,804,032) rounds to the
# paper's label exactly as the smaller presets' counts round to theirs
# (198,912 / 791,040 / 3,154,944 / 6,309,888).rs.
MODEL_SIZE_PRESETS: Mapping[str, Mapping[str, int]] = {
    '200k': {
        'embedding_dim': 64,
        'num_layers': 4,
        'num_heads': 8,
        'widening_factor': 4,
    },
    '800k': {
        'embedding_dim': 128,
        'num_layers': 4,
        'num_heads': 8,
        'widening_factor': 4,
    },
    '3.2m': {
        'embedding_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'widening_factor': 4,
    },
    '6.4m': {
        'embedding_dim': 256,
        'num_layers': 8,
        'num_heads': 8,
        'widening_factor': 4,
    },
    '38m': {
        'embedding_dim': 512,
        'num_layers': 12,
        'num_heads': 8,
        'widening_factor': 4,
    },
}


def config_for_model_size(
    model_size: str,
    vocab_size: int,
) -> TransformerConfig:
  """Builds a Transformer config from a rounded parameter-count preset."""
  try:
    preset = MODEL_SIZE_PRESETS[model_size]
  except KeyError as exc:
    valid_sizes = ', '.join(MODEL_SIZE_PRESETS)
    raise ValueError(
        f'Unknown model size {model_size!r}; expected one of {valid_sizes}.'
    ) from exc
  return TransformerConfig(vocab_size=vocab_size, **preset)


def parameter_count(config: TransformerConfig) -> int:
  """Returns the exact number of trainable parameters for this architecture."""
  dim = config.embedding_dim
  group_size = config.byte_group_size
  if group_size <= 0 or dim % group_size:
    raise ValueError(
        'byte_group_size must be positive and divide embedding_dim; got '
        f'{group_size} and {dim}.'
    )
  widening = config.widening_factor
  parameters_per_layer = (
      (4 + 2 * widening) * dim**2 + (widening + 5) * dim
  )
  return (
      config.vocab_size * (dim // group_size)
      + config.vocab_size * dim
      + config.vocab_size
      + config.num_layers * parameters_per_layer
  )


def attention_block_size_in_group_tokens(
    config: TransformerConfig,
) -> int | None:
  """Converts the byte-equivalent tile budget to grouped token positions."""
  if config.attention_block_size is None:
    return None
  return max(
      1,
      (config.attention_block_size + config.byte_group_size - 1)
      // config.byte_group_size,
  )


def _dense_dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
  """Computes the released dense attention operation.

  Args:
    query: Query vectors with shape [B, Tq, H, D].
    key: Key vectors with shape [B, Tk, H, D].
    value: Value vectors with shape [B, Tk, H, Dv].
    mask: Optional mask broadcastable to [B, H, Tq, Tk].

  Returns:
    Attention output with shape [B, Tq, H, Dv].
  """
  attention = jnp.einsum('bthd,bThd->bhtT', query, key)
  attention *= 1.0 / jnp.sqrt(query.shape[-1])
  if mask is not None:
    attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)
  normalized_attention = jnn.softmax(attention)
  return jnp.einsum('bhtT,bThd->bthd', normalized_attention, value)


def _blockwise_dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    block_size: int,
    mask: jax.Array | None = None,
    is_causal: bool = False,
) -> jax.Array:
  """Computes exact attention without materializing the full score matrix.

  The softmax normalizer and weighted values are accumulated online over key
  blocks. Both scans are rematerialized during backpropagation; without that,
  reverse-mode autodiff would retain the score tiles and recover quadratic
  memory use.

  Args:
    query: Query vectors with shape [B, Tq, H, D].
    key: Key vectors with shape [B, Tk, H, D].
    value: Value vectors with shape [B, Tk, H, Dv].
    block_size: Maximum size of both the query and key blocks.
    mask: Optional mask broadcastable to [B, H, Tq, Tk]. Passing a dense mask
      naturally retains the mask's own quadratic storage; causal attention can
      use `is_causal` instead.
    is_causal: Whether a query at position t can only attend through position
      t (inclusive).

  Returns:
    Attention output with shape [B, Tq, H, Dv].
  """
  if block_size <= 0:
    raise ValueError(
        f'attention block size must be positive; got {block_size}.'
    )
  if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
    raise ValueError('query, key, and value must all have rank 4.')

  batch_size, query_length, num_heads, query_depth = query.shape
  key_batch_size, key_length, key_num_heads, key_depth = key.shape
  value_batch_size, value_length, value_num_heads, value_depth = value.shape
  if query_length == 0:
    return jnp.zeros(
        (batch_size, 0, num_heads, value_depth), dtype=value.dtype
    )
  if key_length == 0:
    raise ValueError('key and value sequences must not be empty.')
  if (
      key_batch_size != batch_size
      or value_batch_size != batch_size
      or key_num_heads != num_heads
      or value_num_heads != num_heads
      or key_depth != query_depth
      or value_length != key_length
  ):
    raise ValueError('query, key, and value shapes are incompatible.')

  # Internally use [B, H, T, D], which makes each score tile and its running
  # softmax statistics contiguous in the final dimensions.
  query = jnp.transpose(query, (0, 2, 1, 3))
  key = jnp.transpose(key, (0, 2, 1, 3))
  value = jnp.transpose(value, (0, 2, 1, 3))

  query_block_size = min(block_size, query_length)
  key_block_size = min(block_size, key_length)
  num_query_blocks = (
      query_length + query_block_size - 1
  ) // query_block_size
  num_key_blocks = (key_length + key_block_size - 1) // key_block_size
  padded_query_length = num_query_blocks * query_block_size
  padded_key_length = num_key_blocks * key_block_size

  query = jnp.pad(
      query, ((0, 0), (0, 0), (0, padded_query_length - query_length), (0, 0))
  )
  key = jnp.pad(
      key, ((0, 0), (0, 0), (0, padded_key_length - key_length), (0, 0))
  )
  value = jnp.pad(
      value,
      ((0, 0), (0, 0), (0, padded_key_length - key_length), (0, 0)),
  )

  def to_blocks(
      array: jax.Array, num_blocks: int, current_block_size: int
  ) -> jax.Array:
    blocks = jnp.reshape(
        array,
        (
            batch_size,
            num_heads,
            num_blocks,
            current_block_size,
            array.shape[-1],
        ),
    )
    return jnp.transpose(blocks, (2, 0, 1, 3, 4))

  query_blocks = to_blocks(query, num_query_blocks, query_block_size)
  key_blocks = to_blocks(key, num_key_blocks, key_block_size)
  value_blocks = to_blocks(value, num_key_blocks, key_block_size)

  # Causal attention without an explicit mask always has at least key zero
  # available to every query. Avoid tracking that fact through every tile in
  # the decoder's common path; explicit masks and non-causal attention can
  # still contain fully masked rows and need the legacy fallback below.
  track_valid_keys = not (is_causal and mask is None)
  padded_mask = None
  if mask is not None:
    mask = jnp.broadcast_to(
        jnp.asarray(mask, dtype=jnp.bool_),
        (batch_size, num_heads, query_length, key_length),
    )
    padded_mask = jnp.pad(
        mask,
        (
            (0, 0),
            (0, 0),
            (0, padded_query_length - query_length),
            (0, padded_key_length - key_length),
        ),
        constant_values=False,
    )

  key_block_indices = jnp.arange(num_key_blocks, dtype=jnp.int32)
  query_block_indices = jnp.arange(num_query_blocks, dtype=jnp.int32)
  query_offsets = jnp.arange(query_block_size, dtype=jnp.int32)
  key_offsets = jnp.arange(key_block_size, dtype=jnp.int32)
  scale = 1.0 / jnp.sqrt(query_depth)
  score_dtype = jnp.result_type(query.dtype, key.dtype)
  accumulator_dtype = jnp.result_type(score_dtype, value.dtype)
  all_masked_output = None
  if track_valid_keys:
    # This is only selected for an explicitly all-masked row, matching the
    # legacy use of a finite minimum score (whose softmax is uniform).
    all_masked_output = jnp.mean(
        value[:, :, :key_length, :].astype(accumulator_dtype), axis=2
    )

  def query_step(
      unused_carry: None,
      inputs: tuple[jax.Array, jax.Array],
  ) -> tuple[None, jax.Array]:
    query_block_index, query_block = inputs
    query_start = query_block_index * query_block_size
    query_positions = query_start + query_offsets

    running_max = jnp.full(
        (batch_size, num_heads, query_block_size),
        -jnp.inf,
        dtype=score_dtype,
    )
    running_sum = jnp.zeros_like(running_max)
    running_output = jnp.zeros(
        (batch_size, num_heads, query_block_size, value_depth),
        dtype=accumulator_dtype,
    )
    has_valid_key = (
        jnp.zeros_like(running_max, dtype=jnp.bool_)
        if track_valid_keys
        else None
    )

    def key_step(
        carry: tuple[
            jax.Array, jax.Array, jax.Array, jax.Array | None
        ],
        key_inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array | None], None
    ]:
      key_block_index, key_block, value_block = key_inputs
      key_start = key_block_index * key_block_size

      def process_block(
          current: tuple[
              jax.Array, jax.Array, jax.Array, jax.Array | None
          ],
      ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        current_max, current_sum, current_output, current_has_valid = current
        scores = jnp.einsum(
            'bhqd,bhkd->bhqk', query_block, key_block
        )
        scores *= scale

        key_positions = key_start + key_offsets
        valid = key_positions[None, :] < key_length
        valid = jnp.broadcast_to(
            valid[None, None, :, :],
            (batch_size, num_heads, query_block_size, key_block_size),
        )
        if is_causal:
          causal = key_positions[None, :] <= query_positions[:, None]
          valid = jnp.logical_and(valid, causal[None, None, :, :])
        if padded_mask is not None:
          mask_block = jax.lax.dynamic_slice(
              padded_mask,
              (0, 0, query_start, key_start),
              (batch_size, num_heads, query_block_size, key_block_size),
          )
          valid = jnp.logical_and(valid, mask_block)

        masked_scores = jnp.where(valid, scores, -jnp.inf)
        block_max = jnp.max(masked_scores, axis=-1)
        new_max = jnp.maximum(current_max, block_max)
        safe_max = jnp.where(jnp.isfinite(new_max), new_max, 0)
        old_scale = jnp.exp(
            jnp.where(
                jnp.isfinite(current_max), current_max - safe_max, -jnp.inf
            )
        )
        weights = jnp.exp(
            jnp.where(
                valid, scores - safe_max[..., None], -jnp.inf
            )
        )
        new_sum = old_scale * current_sum + jnp.sum(weights, axis=-1)
        new_output = old_scale[..., None] * current_output
        new_output += jnp.einsum(
            'bhqk,bhkd->bhqd', weights, value_block
        )
        if track_valid_keys:
          assert current_has_valid is not None
          new_has_valid = jnp.logical_or(
              current_has_valid, jnp.any(valid, axis=-1)
          )
        else:
          new_has_valid = None
        return new_max, new_sum, new_output, new_has_valid

      if is_causal:
        last_query_position = jnp.minimum(
            query_start + query_block_size - 1, query_length - 1
        )
        carry = jax.lax.cond(
            key_start <= last_query_position,
            process_block,
            lambda current: current,
            carry,
        )
      else:
        carry = process_block(carry)
      return carry, None

    rematerialized_key_step = jax.checkpoint(
        key_step, prevent_cse=False
    )
    (_, denominator, numerator, has_valid_key), _ = jax.lax.scan(
        rematerialized_key_step,
        (running_max, running_sum, running_output, has_valid_key),
        (key_block_indices, key_blocks, value_blocks),
    )
    if track_valid_keys:
      assert has_valid_key is not None
      assert all_masked_output is not None
      output = numerator / jnp.where(has_valid_key, denominator, 1)[..., None]
      output = jnp.where(
          has_valid_key[..., None],
          output,
          all_masked_output[:, :, None, :],
      )
    else:
      output = numerator / denominator[..., None]
    return unused_carry, output

  rematerialized_query_step = jax.checkpoint(
      query_step, prevent_cse=False
  )
  _, output_blocks = jax.lax.scan(
      rematerialized_query_step,
      None,
      (query_block_indices, query_blocks),
  )
  output = jnp.transpose(output_blocks, (1, 2, 0, 3, 4))
  output = jnp.reshape(
      output,
      (batch_size, num_heads, padded_query_length, value_depth),
  )
  output = output[:, :, :query_length, :]
  return jnp.transpose(output, (0, 2, 1, 3))


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      name: str | None = None,
      *,
      attention_block_size: int | None = 256,
      positional_encoding: str | None = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      name: Name of the module.
      attention_block_size: Maximum query/key tile size for blockwise
        attention, or None to use dense attention.
      positional_encoding: Positional encoding to apply inside attention.
        Rotary encoding rotates projected queries and keys; None and
        `sinusoidal` leave them unchanged.
    """
    super().__init__(name=name)
    if attention_block_size is not None and attention_block_size <= 0:
      raise ValueError(
          'attention block size must be positive or None; got '
          f'{attention_block_size}.'
      )
    if positional_encoding not in (None, *POSITIONAL_ENCODINGS):
      valid_encodings = ', '.join(POSITIONAL_ENCODINGS)
      raise ValueError(
          f'Unknown positional encoding {positional_encoding!r}; expected '
          f'one of {valid_encodings}.'
      )
    if (
        positional_encoding == ROTARY_POSITION_ENCODING
        and num_hiddens_per_head % 2
    ):
      raise ValueError(
          'Rotary positional encoding requires an even head dimension; got '
          f'{num_hiddens_per_head}.'
      )
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._attention_block_size = attention_block_size
    self._positional_encoding = positional_encoding

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_kv: jax.Array,
      mask: jax.Array | None = None,
      *,
      is_causal: bool = False,
      query_positions: jax.Array | None = None,
      key_positions: jax.Array | None = None,
  ) -> jax.Array:
    """Returns the output of the multi-head attention.

    Args:
      inputs_q: Query inputs with shape [B, Tq, E].
      inputs_kv: Key/value inputs with shape [B, Tk, E].
      mask: Optional attention mask broadcastable to [B, H, Tq, Tk].
      is_causal: Whether queries can attend only to earlier keys.
      query_positions: Optional integer positions shaped [Tq] or [B, Tq].
      key_positions: Optional integer positions shaped [Tk] or [B, Tk].
    """
    batch_size, sequence_length, embedding_size = inputs_q.shape

    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
    k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    # Keep query and key/value sequence lengths explicit. Besides supporting
    # cross-attention, this avoids an ambiguous inferred dimension for empty
    # sequences.
    q = jnp.reshape(
        q,
        (
            batch_size,
            inputs_q.shape[1],
            self._num_heads,
            self._num_hiddens_per_head,
        ),
    )
    kv_shape = (
        batch_size,
        inputs_kv.shape[1],
        self._num_heads,
        self._num_hiddens_per_head,
    )
    k = jnp.reshape(k, kv_shape)
    v = jnp.reshape(v, kv_shape)

    if self._positional_encoding == ROTARY_POSITION_ENCODING:
      if query_positions is None:
        query_positions = jnp.arange(q.shape[1])
      if key_positions is None:
        key_positions = jnp.arange(k.shape[1])
      q = apply_rotary_encoding(q, query_positions)
      k = apply_rotary_encoding(k, key_positions)

    if self._attention_block_size is None:
      if is_causal:
        query_length = q.shape[1]
        key_length = k.shape[1]
        causal_mask = np.tril(
            np.ones((batch_size, 1, query_length, key_length))
        )
        mask = causal_mask if mask is None else jnp.logical_and(
            mask, causal_mask
        )
      output = _dense_dot_product_attention(q, k, v, mask)
    else:
      output = _blockwise_dot_product_attention(
          q,
          k,
          v,
          block_size=self._attention_block_size,
          mask=mask,
          is_causal=is_causal,
      )
    output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
    return hk.Linear(embedding_size, with_bias=False)(output)


def apply_rotary_encoding(
    x: jax.Array,
    positions: jax.Array,
    max_timescale: float = 1e4,
) -> jax.Array:
  """Applies the paper's rotary position encoding to adjacent feature pairs.

  Frequencies span the complete per-head dimension, with
  `theta_i = max_timescale ** (-2 * i / D)`. At position p, each adjacent
  pair `(x_even, x_odd)` is rotated through `p * theta_i`. This matches the
  authors' public JAX implementation: queries and keys use this function while
  values remain unmodified.

  Args:
    x: An array shaped [B, T, D] or [B, T, H, D].
    positions: Zero-based positions shaped [T] or [B, T]. A leading dimension
      of one is broadcast over the input batch.
    max_timescale: Base used to construct the geometric frequencies.

  Returns:
    The rotated array, with the same shape and dtype as `x`.

  Raises:
    ValueError: If shapes are incompatible or the head dimension is odd.
  """
  if x.ndim not in (3, 4):
    raise ValueError(
        'Rotary positional encoding expects rank-3 or rank-4 inputs; got '
        f'rank {x.ndim}.'
    )
  head_dimension = x.shape[-1]
  if head_dimension % 2:
    raise ValueError(
        'Rotary positional encoding requires an even head dimension; got '
        f'{head_dimension}.'
    )
  if max_timescale <= 0:
    raise ValueError(
        f'max_timescale must be positive; got {max_timescale}.'
    )

  positions = jnp.asarray(positions)
  if positions.ndim == 1:
    positions = positions[None, :]
  if positions.ndim != 2:
    raise ValueError(
        'Rotary positions must have shape [T] or [B, T]; got '
        f'{positions.shape}.'
    )
  if positions.shape[1] != x.shape[1]:
    raise ValueError(
        'Rotary positions must match the input sequence length; got '
        f'{positions.shape[1]} and {x.shape[1]}.'
    )
  if positions.shape[0] not in (1, x.shape[0]):
    raise ValueError(
        'Rotary positions must have batch size one or match the input; got '
        f'{positions.shape[0]} and {x.shape[0]}.'
    )

  frequency_indices = jnp.arange(
      head_dimension // 2, dtype=jnp.float32
  )
  inverse_frequencies = max_timescale ** (
      -frequency_indices / (head_dimension // 2)
  )
  angles = positions[..., None] * inverse_frequencies[None, None, :]
  angles = jnp.repeat(angles, 2, axis=-1)
  if x.ndim == 4:
    angles = angles[:, :, None, :]

  pairs = jnp.reshape(x, (*x.shape[:-1], head_dimension // 2, 2))
  rotated_pairs = jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1)
  rotated = jnp.reshape(rotated_pairs, x.shape)
  cosine = jnp.cos(angles).astype(x.dtype)
  sine = jnp.sin(angles).astype(x.dtype)
  return x * cosine + rotated * sine


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> np.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.

  Returns:
    An array of shape [L, D] if `add_negative` or `keep_positive_side` is
    `False`, else [2 * L, D].
  """
  freqs = np.arange(0, hidden_size + 1, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)

  pos_seq = np.arange(start=0, stop=sequence_length)

  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  embeddings = np.concatenate(
      [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
  )
  return embeddings[:, :hidden_size]


def _group_previous_bytes(
    targets: jax.Array,
    group_size: int,
) -> jax.Array:
  """Builds target-aligned groups of preceding bytes.

  TRACE predicts one byte from a window whose adjacent history bytes are
  concatenated in groups. A single fixed grouping of a complete target
  sequence would either leak later bytes within a group or predict all bytes
  in a group from an unnecessarily stale context. Instead, target positions
  with the same residue modulo `group_size` form an independent causal phase.

  For target position `t = phase + group_index * group_size`, the token at
  `group_index` contains exactly
  `[x[t - group_size], ..., x[t - 1]]`. Negative positions are represented by
  zero-valued beginning-of-sequence padding. The result's phase dimension can
  be folded into the batch dimension before attention.

  Args:
    targets: Integer target values with shape [B, T].
    group_size: Number of adjacent history bytes in each group.

  Returns:
    Integer byte groups with shape [B, group_size, ceil(T / group_size),
    group_size].
  """
  batch_size, sequence_length = targets.shape
  num_group_positions = (sequence_length + group_size - 1) // group_size
  if sequence_length == 0:
    return jnp.zeros(
        (batch_size, group_size, 0, group_size), dtype=targets.dtype
    )

  phases = jnp.arange(group_size, dtype=jnp.int32)[:, None, None]
  group_indices = jnp.arange(
      num_group_positions, dtype=jnp.int32
  )[None, :, None]
  byte_offsets = jnp.arange(group_size, dtype=jnp.int32)[None, None, :]
  target_positions = phases + group_indices * group_size
  history_positions = target_positions - group_size + byte_offsets
  valid_history = jnp.logical_and(
      history_positions >= 0, history_positions < sequence_length
  )
  safe_history_positions = jnp.clip(
      history_positions, min=0, max=sequence_length - 1
  )
  history = jnp.take(targets, safe_history_positions, axis=1)
  return jnp.where(valid_history[None, ...], history, 0)


def embed_grouped_history(
    targets: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Embeds target-aligned byte groups for causal attention.

  Returns an array shaped [B * G, ceil(T / G), D], where G is the byte group
  size and D is the Transformer width. Folding grouping phases into the batch
  dimension lets all target bytes be trained in parallel while sharing one
  Transformer and one output head.
  """
  batch_size, sequence_length = targets.shape
  group_size = config.byte_group_size
  embedding_size = config.embedding_dim
  if group_size <= 0 or embedding_size % group_size:
    raise ValueError(
        'byte_group_size must be positive and divide embedding_dim; got '
        f'{group_size} and {embedding_size}.'
    )
  if config.positional_encoding not in POSITIONAL_ENCODINGS:
    valid_encodings = ', '.join(POSITIONAL_ENCODINGS)
    raise ValueError(
        f'Unknown positional encoding {config.positional_encoding!r}; '
        f'expected one of {valid_encodings}.'
    )

  byte_embedding_size = embedding_size // group_size
  embs_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
  embeddings_layer = hk.Embed(
      vocab_size=config.vocab_size,
      embed_dim=byte_embedding_size,
      lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
      w_init=embs_init,
  )
  grouped_history = _group_previous_bytes(targets, group_size)
  embeddings = embeddings_layer(grouped_history)
  embeddings *= jnp.sqrt(embedding_size)

  num_group_positions = grouped_history.shape[2]
  embeddings = jnp.reshape(
      embeddings,
      (
          batch_size,
          group_size,
          num_group_positions,
          embedding_size,
      ),
  )

  if config.positional_encoding == SINUSOIDAL_POSITION_ENCODING:
    # Preserve byte-level absolute positions: phase r, group position k
    # predicts byte r + kG. This reduces exactly to the original positional
    # encoding when G=1 and is invariant to how much future context is present
    # in a call. Rotary positions are instead applied to queries and keys.
    padded_sequence_length = num_group_positions * group_size
    all_pos_encodings = sinusoid_position_encoding(
        sequence_length=padded_sequence_length,
        hidden_size=embedding_size,
    )
    target_positions = (
        np.arange(group_size)[:, None]
        + group_size * np.arange(num_group_positions)[None, :]
    )
    pos_encodings = all_pos_encodings[target_positions]
    embeddings = embeddings + pos_encodings[None, ...]
  return jnp.reshape(
      embeddings,
      (
          batch_size * group_size,
          num_group_positions,
          embedding_size,
      ),
  )


def layer_norm(x: jax.Array) -> jax.Array:
  """Helper function for layer norm."""
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def _grouped_position_ids(
    batch_size: int,
    num_group_positions: int,
    group_size: int,
) -> jax.Array:
  """Returns byte-aligned positions for phases folded into the batch."""
  phase_positions = (
      jnp.arange(group_size, dtype=jnp.int32)[:, None]
      + group_size
      * jnp.arange(num_group_positions, dtype=jnp.int32)[None, :]
  )
  positions = jnp.broadcast_to(
      phase_positions[None, ...],
      (batch_size, group_size, num_group_positions),
  )
  return jnp.reshape(
      positions, (batch_size * group_size, num_group_positions)
  )


def shift_right(sequences: jax.Array) -> jax.Array:
  """Right-shift the one-hot encoded input by padding on the temporal axis."""
  bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
  padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
  return padded_sequences[:, :-1]


def transformer_decoder(
    targets: jax.Array,
    config: TransformerConfig,
) -> jax.Array:
  """Returns the transformer decoder output, shape [B, T, V].

  Args:
    targets: The integer target values, shape [B, T].
    config: The config to use for the transformer.
  """
  batch_size, sequence_length = targets.shape

  # Embed G-byte history groups in G target-aligned phases. Attention sees
  # ceil(T / G) positions per phase rather than T byte positions.
  embeddings = embed_grouped_history(targets, config)
  position_ids = _grouped_position_ids(
      batch_size,
      embeddings.shape[1],
      config.byte_group_size,
  )

  causal_mask = None
  if config.attention_block_size is None:
    attention_batch_size, attention_length = embeddings.shape[:2]
    # Preserve the released dense operation for old checkpoints.
    causal_mask = np.tril(
        np.ones(
            (attention_batch_size, 1, attention_length, attention_length),
            dtype=np.bool_,
        )
    )

  h = embeddings
  # Grouping phases are folded into the batch dimension. Shrinking each tile
  # by G ensures their combined score tile is no larger than the ungrouped tile
  # (and is G times smaller in score elements for an exact division).
  attention_block_size = attention_block_size_in_group_tokens(config)
  for _ in range(config.num_layers):
    attention_module = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        num_hiddens_per_head=config.embedding_dim // config.num_heads,
        attention_block_size=attention_block_size,
        positional_encoding=config.positional_encoding,
    )
    if config.attention_block_size is None:
      self_attention = attention_module(
          inputs_q=h,
          inputs_kv=h,
          mask=causal_mask,
          query_positions=position_ids,
          key_positions=position_ids,
      )
    else:
      self_attention = attention_module(
          inputs_q=h,
          inputs_kv=h,
          is_causal=True,
          query_positions=position_ids,
          key_positions=position_ids,
      )
    attention = layer_norm(h + self_attention)

    # Position-wise feedforward network.
    h = hk.Linear(config.embedding_dim * config.widening_factor)(attention)
    h = jnn.gelu(h)
    h = hk.Linear(config.embedding_dim)(h)
    h = layer_norm(h + attention)

  phase_logits = hk.Linear(config.vocab_size)(h)
  num_group_positions = phase_logits.shape[1]
  phase_logits = jnp.reshape(
      phase_logits,
      (
          batch_size,
          config.byte_group_size,
          num_group_positions,
          config.vocab_size,
      ),
  )
  # [B, G, ceil(T/G), V] -> byte order [B, T, V].
  logits = jnp.transpose(phase_logits, (0, 2, 1, 3))
  logits = jnp.reshape(
      logits,
      (
          batch_size,
          num_group_positions * config.byte_group_size,
          config.vocab_size,
      ),
  )
  logits = logits[:, :sequence_length, :]
  return jnn.log_softmax(logits, axis=-1)
