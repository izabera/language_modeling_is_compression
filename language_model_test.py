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

"""Tests for cached language-model compression and inference."""

import functools
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
import haiku as hk
import jax
import numpy as np

from language_modeling_is_compression import model_checkpoint
from language_modeling_is_compression import transformer
from language_modeling_is_compression.compressors import language_model


class LanguageModelCompressorTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    language_model._cached_compressor.cache_clear()
    language_model._model_apply_fns.cache_clear()

  def tearDown(self):
    language_model._cached_compressor.cache_clear()
    language_model._model_apply_fns.cache_clear()
    super().tearDown()

  def _mock_model(self):
    config = transformer.TransformerConfig(
      vocab_size=256,
      embedding_dim=4,
      byte_group_size=4,
      num_layers=1,
      num_heads=1,
      widening_factor=1,
      attention_block_size=2,
    )

    def eager_apply(params, inputs):
      del params
      return np.zeros((*inputs.shape, config.vocab_size), dtype=np.float32)

    def jitted_apply(params, inputs):
      del params
      return np.zeros((*inputs.shape, config.vocab_size), dtype=np.float32)

    return config, eager_apply, jitted_apply

  def test_top_level_compress_reuses_loaded_checkpoint(self):
    config, eager_apply, jitted_apply = self._mock_model()
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'params.npz'
      path.write_bytes(b'checkpoint placeholder')
      with (
        mock.patch.object(
          language_model,
          '_retrieve_model',
          return_value=({}, config),
        ) as retrieve_model,
        mock.patch.object(
          language_model,
          '_model_apply_fns',
          return_value=(eager_apply, jitted_apply),
        ),
        mock.patch.object(
          language_model,
          '_compress_with_predict_fn',
          return_value=b'encoded',
        ) as compress_with_predict_fn,
      ):
        self.assertEqual(
          language_model.compress(b'first', model_path=str(path)),
          b'encoded',
        )
        self.assertEqual(
          language_model.compress(b'second', model_path=str(path)),
          b'encoded',
        )

    retrieve_model.assert_called_once()
    self.assertEqual(compress_with_predict_fn.call_count, 2)

  def test_top_level_cache_notices_replaced_checkpoint(self):
    config, eager_apply, jitted_apply = self._mock_model()
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'params.npz'
      path.write_bytes(b'first')
      with (
        mock.patch.object(
          language_model,
          '_retrieve_model',
          return_value=({}, config),
        ) as retrieve_model,
        mock.patch.object(
          language_model,
          '_model_apply_fns',
          return_value=(eager_apply, jitted_apply),
        ),
        mock.patch.object(
          language_model,
          '_compress_with_predict_fn',
          return_value=b'encoded',
        ),
      ):
        language_model.compress(b'data', model_path=str(path))
        path.write_bytes(b'a checkpoint with a different size')
        language_model.compress(b'data', model_path=str(path))

    self.assertEqual(retrieve_model.call_count, 2)

  def test_only_fixed_shape_compression_uses_jitted_predictor(self):
    config, eager_apply, jitted_apply = self._mock_model()
    with (
      mock.patch.object(
        language_model,
        '_retrieve_model',
        return_value=({}, config),
      ),
      mock.patch.object(
        language_model,
        '_model_apply_fns',
        return_value=(eager_apply, jitted_apply),
      ),
    ):
      compressor = language_model.LanguageModelCompressor('unused.npz')

    with mock.patch.object(
      language_model,
      '_compress_with_predict_fn',
      return_value=b'encoded',
    ) as compress_with_predict_fn:
      compressor.compress(b'data')
      fast_predict_fn = compress_with_predict_fn.call_args.kwargs['predict_fn']
      compressor.compress(b'data', use_slow_lossless_compression=True)
      slow_predict_fn = compress_with_predict_fn.call_args.kwargs['predict_fn']

    self.assertIs(fast_predict_fn.func, jitted_apply)
    self.assertIs(slow_predict_fn.func, eager_apply)

    with mock.patch.object(
      language_model,
      '_decompress_with_predict_fn',
      return_value=b'decoded',
    ) as decompress_with_predict_fn:
      self.assertEqual(compressor.decompress(b'data'), b'decoded')
    decompress_predict_fn = decompress_with_predict_fn.call_args.kwargs['predict_fn']
    self.assertIs(decompress_predict_fn.func, eager_apply)

  def test_jitted_application_accepts_dynamic_parameters(self):
    config = transformer.TransformerConfig(
      vocab_size=8,
      embedding_dim=4,
      num_layers=1,
      num_heads=1,
      widening_factor=1,
      attention_block_size=2,
    )
    model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
    )
    inputs = np.array([[1, 2, 3]], dtype=np.uint8)
    params = model.init(jax.random.PRNGKey(0), inputs)
    changed_params = jax.tree.map(lambda value: value + 0.1, params)
    eager_apply, jitted_apply = language_model._model_apply_fns(
      language_model._config_cache_key(config)
    )

    expected = eager_apply(changed_params, inputs)
    actual = jitted_apply(changed_params, inputs)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  def test_slow_compression_still_round_trips(self):
    config = transformer.TransformerConfig(
      vocab_size=256,
      embedding_dim=4,
      byte_group_size=4,
      num_layers=1,
      num_heads=1,
      widening_factor=1,
      attention_block_size=2,
    )
    model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
    )
    params = model.init(jax.random.PRNGKey(7), np.zeros((1, 1), dtype=np.uint8))
    original = bytes((0, 127, 3, 1, 4, 1, 5, 9, 2))

    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'params.npz'
      model_checkpoint.save(path, params, config)
      fast_compressed = language_model.compress(
        original,
        model_path=str(path),
      )
      compressed, padded_bits = language_model.compress(
        original,
        return_num_padded_bits=True,
        use_slow_lossless_compression=True,
        model_path=str(path),
      )
      decompressed = language_model.decompress(
        compressed,
        num_padded_bits=padded_bits,
        uncompressed_length=len(original),
        model_path=str(path),
      )

    self.assertIsInstance(fast_compressed, bytes)
    self.assertEqual(padded_bits, 0)
    self.assertEqual(len(compressed) % 4, 0)
    self.assertEqual(decompressed, original)

  def test_constriction_fast_and_slow_paths_round_trip_uniform_predictions(
      self,
  ):
    def uniform_predict(inputs):
      return np.zeros((*inputs.shape, 256), dtype=np.float32)

    original = bytes((0, 255, 1, 254, 2, 127, 128, 3, 9))
    compressed_outputs = []

    for use_slow_lossless_compression in (False, True):
      with self.subTest(slow=use_slow_lossless_compression):
        compressed, padded_bits = language_model._compress_with_predict_fn(
            original,
            predict_fn=uniform_predict,
            return_num_padded_bits=True,
            use_slow_lossless_compression=use_slow_lossless_compression,
        )
        decompressed = language_model._decompress_with_predict_fn(
            compressed,
            predict_fn=uniform_predict,
            num_padded_bits=padded_bits,
            uncompressed_length=len(original),
        )

        self.assertEqual(padded_bits, 0)
        self.assertEqual(len(compressed) % 4, 0)
        self.assertEqual(decompressed, original)
        compressed_outputs.append(compressed)

    self.assertEqual(compressed_outputs[0], compressed_outputs[1])


if __name__ == '__main__':
  absltest.main()
