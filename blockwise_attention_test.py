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

"""Tests for exact, memory-bounded blockwise attention."""

import inspect
from unittest import mock

from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import transformer


def _dense_attention(
    query, key, value, *, mask=None, is_causal=False
):
  if is_causal:
    query_positions = np.arange(query.shape[1])[:, None]
    key_positions = np.arange(key.shape[1])[None, :]
    causal_mask = key_positions <= query_positions
    mask = causal_mask if mask is None else np.logical_and(mask, causal_mask)
  return transformer._dense_dot_product_attention(query, key, value, mask)


class BlockwiseAttentionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    query_key, key_key, value_key = jax.random.split(
        jax.random.PRNGKey(12), 3
    )
    self.query = jax.random.normal(query_key, (2, 7, 2, 4))
    self.key = jax.random.normal(key_key, (2, 9, 2, 4))
    self.value = jax.random.normal(value_key, (2, 9, 2, 3))

  def test_blockwise_attention_defaults_to_256_token_tiles(self):
    config = transformer.TransformerConfig(vocab_size=8)
    preset_config = transformer.config_for_model_size('200k', vocab_size=8)
    module_default = inspect.signature(
        transformer.MultiHeadDotProductAttention.__init__
    ).parameters['attention_block_size'].default

    self.assertEqual(config.attention_block_size, 256)
    self.assertEqual(preset_config.attention_block_size, 256)
    self.assertEqual(module_default, 256)

  def test_causal_specialization_bitwise_matches_explicit_mask(self):
    query_positions = jnp.arange(self.query.shape[1])[:, None]
    key_positions = jnp.arange(self.key.shape[1])[None, :]
    causal_mask = key_positions <= query_positions

    def specialized_attention(query, key, value):
      return transformer._blockwise_dot_product_attention(
          query,
          key,
          value,
          block_size=4,
          is_causal=True,
      )

    def generic_attention(query, key, value):
      return transformer._blockwise_dot_product_attention(
          query,
          key,
          value,
          block_size=4,
          mask=causal_mask,
      )

    specialized_output = specialized_attention(
        self.query, self.key, self.value
    )
    generic_output = generic_attention(self.query, self.key, self.value)
    np.testing.assert_array_equal(specialized_output, generic_output)

    specialized_gradients = jax.grad(
        lambda query, key, value: jnp.sin(
            specialized_attention(query, key, value)
        ).sum(),
        argnums=(0, 1, 2),
    )(self.query, self.key, self.value)
    generic_gradients = jax.grad(
        lambda query, key, value: jnp.sin(
            generic_attention(query, key, value)
        ).sum(),
        argnums=(0, 1, 2),
    )(self.query, self.key, self.value)
    for specialized_gradient, generic_gradient in zip(
        specialized_gradients, generic_gradients
    ):
      np.testing.assert_array_equal(specialized_gradient, generic_gradient)

  def test_causal_specialization_omits_validity_reductions_and_fallback(self):
    with mock.patch.object(
        transformer.jnp,
        'any',
        side_effect=AssertionError('causal specialization reduced validity'),
    ), mock.patch.object(
        transformer.jnp,
        'mean',
        side_effect=AssertionError('causal specialization built fallback'),
    ):
      output = transformer._blockwise_dot_product_attention(
          self.query,
          self.key,
          self.value,
          block_size=4,
          is_causal=True,
      )

    self.assertEqual(output.shape, (2, 7, 2, 3))

  def test_causal_ragged_blocks_match_dense_forward_and_gradients(self):
    def dense_loss(query, key, value):
      output = _dense_attention(
          query, key, value, is_causal=True
      )
      return jnp.sin(output).sum()

    def blockwise_loss(query, key, value):
      output = transformer._blockwise_dot_product_attention(
          query,
          key,
          value,
          block_size=4,
          is_causal=True,
      )
      return jnp.sin(output).sum()

    dense_output = _dense_attention(
        self.query, self.key, self.value, is_causal=True
    )
    blockwise_output = transformer._blockwise_dot_product_attention(
        self.query,
        self.key,
        self.value,
        block_size=4,
        is_causal=True,
    )
    np.testing.assert_allclose(
        blockwise_output, dense_output, rtol=2e-6, atol=2e-6
    )

    dense_gradients = jax.grad(
        dense_loss, argnums=(0, 1, 2)
    )(self.query, self.key, self.value)
    blockwise_gradients = jax.grad(
        blockwise_loss, argnums=(0, 1, 2)
    )(self.query, self.key, self.value)
    for blockwise_gradient, dense_gradient in zip(
        blockwise_gradients, dense_gradients
    ):
      np.testing.assert_allclose(
          blockwise_gradient,
          dense_gradient,
          rtol=3e-6,
          atol=3e-6,
      )

  def test_explicit_integer_mask_and_all_masked_row_match_dense(self):
    mask = jax.random.bernoulli(
        jax.random.PRNGKey(21), 0.6, (2, 1, 7, 9)
    ).astype(jnp.int32)
    mask = mask.at[:, :, 3, :].set(0)

    dense_output = _dense_attention(
        self.query, self.key, self.value, mask=mask
    )
    blockwise_output = transformer._blockwise_dot_product_attention(
        self.query,
        self.key,
        self.value,
        block_size=4,
        mask=mask,
    )

    np.testing.assert_allclose(
        blockwise_output, dense_output, rtol=2e-6, atol=2e-6
    )

  def test_valid_nan_score_is_not_treated_as_an_all_masked_row(self):
    query = self.query.at[0, 3, 0, 0].set(jnp.nan)

    blockwise_output = transformer._blockwise_dot_product_attention(
        query,
        self.key,
        self.value,
        block_size=4,
        is_causal=True,
    )

    self.assertTrue(jnp.isnan(blockwise_output[0, 3, 0]).all())
    self.assertTrue(jnp.isfinite(blockwise_output[0, :3]).all())

  def test_mixed_value_dtype_matches_dense(self):
    value = self.value.astype(jnp.bfloat16)
    dense_output = _dense_attention(
        self.query, self.key, value, is_causal=True
    )
    blockwise_output = transformer._blockwise_dot_product_attention(
        self.query,
        self.key,
        value,
        block_size=4,
        is_causal=True,
    )

    self.assertEqual(blockwise_output.dtype, dense_output.dtype)
    np.testing.assert_allclose(
        blockwise_output, dense_output, rtol=2e-6, atol=2e-6
    )

  def test_block_size_one_and_larger_than_sequence_match_dense(self):
    dense_output = _dense_attention(
        self.query, self.key, self.value, is_causal=True
    )

    for block_size in (1, 64):
      with self.subTest(block_size=block_size):
        blockwise_output = transformer._blockwise_dot_product_attention(
            self.query,
            self.key,
            self.value,
            block_size=block_size,
            is_causal=True,
        )
        np.testing.assert_allclose(
            blockwise_output, dense_output, rtol=2e-6, atol=2e-6
        )

  def test_decoder_matches_dense_with_the_same_parameters(self):
    dense_config = transformer.TransformerConfig(
        vocab_size=16,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        attention_block_size=None,
    )
    blockwise_config = transformer.TransformerConfig(
        vocab_size=16,
        embedding_dim=8,
        num_layers=2,
        num_heads=2,
        attention_block_size=4,
    )
    dense_decoder = hk.without_apply_rng(
        hk.transform(
            lambda targets: transformer.transformer_decoder(
                targets, dense_config
            )
        )
    )
    blockwise_decoder = hk.without_apply_rng(
        hk.transform(
            lambda targets: transformer.transformer_decoder(
                targets, blockwise_config
            )
        )
    )
    targets = jnp.array([[1, 3, 5, 7, 9, 11, 13]], dtype=jnp.uint8)
    params = dense_decoder.init(jax.random.PRNGKey(31), targets)

    dense_output = dense_decoder.apply(params, targets)
    blockwise_output = blockwise_decoder.apply(params, targets)

    np.testing.assert_allclose(
        blockwise_output, dense_output, rtol=1e-5, atol=1e-5
    )

    dense_gradients = jax.grad(
        lambda current_params: dense_decoder.apply(
            current_params, targets
        ).sum()
    )(params)
    blockwise_gradients = jax.grad(
        lambda current_params: blockwise_decoder.apply(
            current_params, targets
        ).sum()
    )(params)
    for blockwise_gradient, dense_gradient in zip(
        jax.tree.leaves(blockwise_gradients),
        jax.tree.leaves(dense_gradients),
    ):
      np.testing.assert_allclose(
          blockwise_gradient,
          dense_gradient,
          rtol=2e-5,
          atol=2e-5,
      )

  def test_decoder_uses_causal_blocks_without_a_dense_mask(self):
    constructor_calls = []
    attention_calls = []

    class RecordingAttention:

      def __init__(self, **kwargs):
        constructor_calls.append(kwargs)

      def __call__(self, *args, **kwargs):
        attention_calls.append((args, kwargs))
        return jnp.zeros_like(kwargs['inputs_q'])

    config = transformer.TransformerConfig(
        vocab_size=8,
        embedding_dim=4,
        num_layers=2,
        num_heads=2,
        attention_block_size=3,
    )
    decoder = hk.transform(
        lambda targets: transformer.transformer_decoder(targets, config)
    )
    targets = jnp.array([[1, 2, 3]], dtype=jnp.uint8)

    with mock.patch.object(
        transformer, 'MultiHeadDotProductAttention', RecordingAttention
    ), mock.patch.object(
        transformer.np,
        'tril',
        side_effect=AssertionError('decoder constructed a dense causal mask'),
    ) as tril:
      decoder.init(jax.random.PRNGKey(0), targets)

    tril.assert_not_called()
    self.assertLen(constructor_calls, config.num_layers)
    self.assertLen(attention_calls, config.num_layers)
    for constructor_kwargs in constructor_calls:
      self.assertEqual(
          constructor_kwargs['attention_block_size'],
          transformer.attention_block_size_in_group_tokens(config),
      )
    for positional_args, call_kwargs in attention_calls:
      self.assertEmpty(positional_args)
      self.assertNotIn('mask', call_kwargs)
      self.assertTrue(call_kwargs['is_causal'])

  def test_compiled_gradient_uses_much_less_temporary_memory(self):
    if jax.default_backend() != 'cpu':
      self.skipTest('This regression specifically measures the JAX CPU path.')

    shape = (1, 512, 8, 8)
    array_spec = jax.ShapeDtypeStruct(shape, jnp.float32)

    def dense_attention(query, key, value):
      positions = jnp.arange(query.shape[1])
      causal_mask = (
          positions[None, None, :, None]
          >= positions[None, None, None, :]
      )
      return transformer._dense_dot_product_attention(
          query, key, value, causal_mask
      )

    def blockwise_attention(query, key, value):
      return transformer._blockwise_dot_product_attention(
          query,
          key,
          value,
          block_size=128,
          is_causal=True,
      )

    def compiled_gradient_temporary_bytes(attention_fn):
      gradient_fn = jax.grad(
          lambda query, key, value: jnp.sin(
              attention_fn(query, key, value)
          ).sum(),
          argnums=(0, 1, 2),
      )
      executable = jax.jit(gradient_fn).lower(
          array_spec, array_spec, array_spec
      ).compile()
      return executable.memory_analysis().temp_size_in_bytes

    dense_bytes = compiled_gradient_temporary_bytes(dense_attention)
    blockwise_bytes = compiled_gradient_temporary_bytes(blockwise_attention)

    self.assertLess(blockwise_bytes, dense_bytes // 4)


if __name__ == '__main__':
  absltest.main()
