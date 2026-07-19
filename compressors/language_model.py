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

"""Implements a lossless compressor with language models (range coding)."""

import dataclasses
import functools
import os
from typing import Callable

import constriction
import haiku as hk
import jax
import numpy as np

from language_modeling_is_compression import constants
from language_modeling_is_compression import model_checkpoint
from language_modeling_is_compression import transformer


_CATEGORICAL_MODEL = constriction.stream.model.Categorical(perfect=False)
_LITTLE_ENDIAN_UINT32 = np.dtype('<u4')


def _retrieve_model(
    model_path: str = 'params.npz',
) -> tuple[hk.Params, transformer.TransformerConfig]:
  """Returns the trained model parameters and architecture.

  Raises:
    FileNotFoundError if the model checkpoint does not exist yet, in which case
    the user should launch a training with train.py first.
  """
  try:
    return model_checkpoint.load(
        model_path,
        default_config=transformer.TransformerConfig(
            vocab_size=constants.ALPHABET_SIZE
        ),
    )
  except FileNotFoundError as exc:
    raise FileNotFoundError(
        'You must train a model first; the model checkpoint '
        f'{model_path!r} does not exist yet.'
    ) from exc


_CONFIG_FIELD_NAMES = tuple(
    field.name for field in dataclasses.fields(transformer.TransformerConfig)
)


def _config_cache_key(
    config: transformer.TransformerConfig,
) -> tuple[object, ...]:
  """Returns a hashable representation of a Transformer configuration."""
  return tuple(getattr(config, name) for name in _CONFIG_FIELD_NAMES)


@functools.lru_cache(maxsize=4)
def _model_apply_fns(
    config_values: tuple[object, ...],
) -> tuple[Callable[..., jax.Array], Callable[..., jax.Array]]:
  """Builds eager and JIT model applications shared by equal configs.

  Parameters remain explicit arguments to the JIT function. Consequently, a
  checkpoint replacement with the same parameter shapes reuses the compiled
  executable instead of embedding the old checkpoint values in it.
  """
  config = transformer.TransformerConfig(
      **dict(zip(_CONFIG_FIELD_NAMES, config_values, strict=True))
  )
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )

  def apply_fn(params: hk.Params, inputs: np.ndarray) -> jax.Array:
    return model.apply(params, None, inputs)

  return apply_fn, jax.jit(apply_fn)


def _retrieve_predict_fn(
    params: hk.Params,
    config: transformer.TransformerConfig,
    *,
    use_jit: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns the prediction function for the trained model."""
  apply_fn, jitted_apply_fn = _model_apply_fns(_config_cache_key(config))
  # Transfer checkpoint arrays only once. On CPU this mainly avoids repeatedly
  # wrapping the same NumPy arrays; on accelerators it also avoids a transfer
  # for every compressed chunk.
  device_params = jax.device_put(params)
  return functools.partial(
      jitted_apply_fn if use_jit else apply_fn, device_params
  )


class LanguageModelCompressor:
  """Reusable language-model compressor backed by one checkpoint snapshot."""

  def __init__(self, model_path: str = 'params.npz') -> None:
    params, self.config = _retrieve_model(model_path)
    apply_fn, jitted_apply_fn = _model_apply_fns(
        _config_cache_key(self.config)
    )
    device_params = jax.device_put(params)
    self._predict_fn = functools.partial(jitted_apply_fn, device_params)
    self._eager_predict_fn = functools.partial(apply_fn, device_params)

  def compress(
      self,
      data: bytes,
      return_num_padded_bits: bool = False,
      use_slow_lossless_compression: bool = False,
  ) -> bytes | tuple[bytes, int]:
    """Compresses data using this instance's checkpoint snapshot."""
    # Prefix lengths vary in the slow path. Leaving that path eager avoids
    # populating the JIT cache with one executable per sequence length.
    predict_fn = (
        self._eager_predict_fn
        if use_slow_lossless_compression
        else self._predict_fn
    )
    return _compress_with_predict_fn(
        data,
        predict_fn=predict_fn,
        return_num_padded_bits=return_num_padded_bits,
        use_slow_lossless_compression=use_slow_lossless_compression,
    )

  def decompress(
      self,
      data: bytes,
      num_padded_bits: int = 0,
      uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
  ) -> bytes:
    """Decompresses data using this instance's checkpoint snapshot."""
    # Decompression grows the prefix one token at a time, so JIT-compiling the
    # complete model here would compile a separate executable at every step.
    return _decompress_with_predict_fn(
        data,
        predict_fn=self._eager_predict_fn,
        num_padded_bits=num_padded_bits,
        uncompressed_length=uncompressed_length,
    )


def _checkpoint_signature(path: str) -> tuple[int, ...]:
  """Returns file metadata that changes when a checkpoint is replaced."""
  stat = os.stat(path)
  return (
      stat.st_dev,
      stat.st_ino,
      stat.st_size,
      stat.st_mtime_ns,
      stat.st_ctime_ns,
  )


@functools.lru_cache(maxsize=1)
def _cached_compressor(
    absolute_path: str,
    unused_signature: tuple[int, ...],
) -> LanguageModelCompressor:
  """Loads one reusable compressor for a particular checkpoint version."""
  del unused_signature
  return LanguageModelCompressor(absolute_path)


def _get_compressor(model_path: str) -> LanguageModelCompressor:
  """Returns a cached compressor while noticing checkpoint replacements."""
  absolute_path = os.path.abspath(os.fspath(model_path))
  try:
    signature = _checkpoint_signature(absolute_path)
  except FileNotFoundError:
    # Do not cache failures, and retain _retrieve_model's actionable error.
    return LanguageModelCompressor(model_path)
  return _cached_compressor(absolute_path, signature)


def _compress_with_predict_fn(
    data: bytes,
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = False,
) -> bytes | tuple[bytes, int]:
  """Compresses data with an already-loaded prediction function."""
  # Convert the `data` into an array of integers (representing the bytes).
  sequence_array = np.frombuffer(data, dtype=np.uint8)

  if use_slow_lossless_compression:
    log_probs = list()
    for subsequence_length in range(len(sequence_array)):
      subsequence_probs = predict_fn(
          sequence_array[None, : subsequence_length + 1]
      )
      log_probs.append(subsequence_probs[0, -1])
    log_probs = np.vstack(log_probs)
  else:
    log_probs = predict_fn(sequence_array[None])[0, ...]
  probs = np.ascontiguousarray(np.exp(log_probs), dtype=np.float32)

  encoder = constriction.stream.queue.RangeEncoder()
  encoder.encode(
      sequence_array.astype(np.int32),
      _CATEGORICAL_MODEL,
      probs,
  )
  compressed_words = encoder.get_compressed()
  compressed_bytes = compressed_words.astype(
      _LITTLE_ENDIAN_UINT32, copy=False
  ).tobytes()

  if return_num_padded_bits:
    # Constriction's range coder emits complete 32-bit words and its decoder
    # does not require a separate padding count.
    return compressed_bytes, 0

  return compressed_bytes


def compress(
    data: bytes,
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = False,
    model_path: str = 'params.npz',
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using range coding and a pretrained model.

  Args:
    data: The data to be compressed.
    return_num_padded_bits: Whether to also return a padding count for API
      compatibility. Constriction streams do not require a separate padding
      count, so this value is always zero.
    use_slow_lossless_compression: Whether to compute the `pdf`s for all tokens
      in the data stream in one go or separately for every proper subsequence.
      When only compressing data (i.e., without decompression) use the first
      approach (i.e., `False`) since it has an O(n) runtime complexity, while
      the latter is O(n^2). However, the goal is to losslessly decompress the
      compressed output, use the second option (i.e., `True`) since this is what
      happens in the decoder (which iteratively reconstructs the sequence).
    model_path: Path to the trained model checkpoint.

  Returns:
    The compressed data.
  """
  return _get_compressor(model_path).compress(
      data,
      return_num_padded_bits=return_num_padded_bits,
      use_slow_lossless_compression=use_slow_lossless_compression,
  )


def _decompress_with_predict_fn(
    data: bytes,
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
) -> bytes:
  """Decompresses data with an already-loaded prediction function."""
  if num_padded_bits:
    raise ValueError(
        'Constriction streams do not use padded bits; expected '
        f'num_padded_bits=0, got {num_padded_bits}.'
    )
  if len(data) % _LITTLE_ENDIAN_UINT32.itemsize:
    raise ValueError(
        'Compressed data length must be a multiple of 4 bytes for a '
        f'Constriction stream; got {len(data)}.'
    )

  compressed_words = np.frombuffer(data, dtype=_LITTLE_ENDIAN_UINT32).astype(
      np.uint32, copy=False
  )
  decoder = constriction.stream.queue.RangeDecoder(compressed_words)

  # We need a dummy token because the language model right-shifts the sequence
  # by one when computing the conditional probabilities. Concretely, at every
  # step, we need the `pdf` of the next token given all currently decompressed
  # tokens, but without a dummy token, the last `pdf` would be that of the last
  # already decompressed token. The value of the dummy token is irrelevant.
  sequence_array = np.zeros((uncompressed_length + 1,), dtype=np.uint8)

  for idx in range(uncompressed_length):
    prefix_with_dummy = sequence_array[: idx + 1]
    probs = np.ascontiguousarray(
        np.exp(predict_fn(prefix_with_dummy[None])[0, -1]),
        dtype=np.float32,
    )
    token = decoder.decode(
        _CATEGORICAL_MODEL,
        probs[None],
    )
    sequence_array[idx] = token[0]

  return sequence_array[:uncompressed_length].tobytes()


def decompress(
    data: bytes,
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
    model_path: str = 'params.npz',
) -> bytes:
  """Decompresses the `data` using range coding and a pretrained model.

  Args:
    data: The data to be decompressed.
    num_padded_bits: Must be zero. Retained for compatibility with the previous
      arithmetic-coder API.
    uncompressed_length: The length of the original data stream (in bytes).
    model_path: Path to the trained model checkpoint.

  Returns:
    The decompressed data.
  """
  return _get_compressor(model_path).decompress(
      data,
      num_padded_bits=num_padded_bits,
      uncompressed_length=uncompressed_length,
  )
