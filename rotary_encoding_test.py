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

"""Tests for rotary positional encodings."""

from unittest import mock

from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from language_modeling_is_compression import transformer


def _reference_rotary(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
  """Independent NumPy implementation of adjacent-pair RoPE."""
  head_dimension = x.shape[-1]
  inverse_frequencies = 10_000.0 ** (
      -np.arange(0, head_dimension, 2, dtype=np.float32) / head_dimension
  )
  angles = positions[..., None, None] * inverse_frequencies[None, None, :]
  cosine = np.cos(angles)
  sine = np.sin(angles)
  expected = np.empty_like(x)
  expected[..., ::2] = x[..., ::2] * cosine - x[..., 1::2] * sine
  expected[..., 1::2] = x[..., 1::2] * cosine + x[..., ::2] * sine
  return expected


class RotaryEncodingTest(absltest.TestCase):

  def test_rotary_is_the_default_for_new_models(self):
    config = transformer.TransformerConfig(vocab_size=256)

    self.assertEqual(
        config.positional_encoding, transformer.ROTARY_POSITION_ENCODING
    )

  def test_matches_adjacent_pair_numpy_reference(self):
    x = np.linspace(-2.0, 3.0, num=2 * 3 * 2 * 6, dtype=np.float32)
    x = x.reshape(2, 3, 2, 6)
    positions = np.array([[0, 2, 5], [1, 4, 7]], dtype=np.int32)

    actual = transformer.apply_rotary_encoding(
        jnp.asarray(x), jnp.asarray(positions)
    )
    expected = _reference_rotary(x, positions)

    self.assertEqual(actual.shape, x.shape)
    self.assertEqual(actual.dtype, jnp.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual[0, 0], x[0, 0])

  def test_common_position_offset_preserves_attention_logits(self):
    query = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 2, 8))
    key = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 2, 8))
    query_positions = jnp.array([0, 4, 8])
    key_positions = jnp.array([1, 5, 9, 13])

    def scores(offset):
      rotated_query = transformer.apply_rotary_encoding(
          query, query_positions + offset
      )
      rotated_key = transformer.apply_rotary_encoding(
          key, key_positions + offset
      )
      return jnp.einsum('bthd,bThd->bhtT', rotated_query, rotated_key)

    np.testing.assert_allclose(scores(0), scores(37), rtol=2e-5, atol=2e-5)

  def test_rejects_odd_head_dimension(self):
    with self.assertRaisesRegex(ValueError, 'even head dimension'):
      transformer.apply_rotary_encoding(
          jnp.zeros((1, 2, 1, 3)), jnp.arange(2)
      )

    attention = hk.transform(
        lambda inputs: transformer.MultiHeadDotProductAttention(
            num_heads=1,
            num_hiddens_per_head=3,
            positional_encoding=transformer.ROTARY_POSITION_ENCODING,
        )(inputs, inputs)
    )
    with self.assertRaisesRegex(ValueError, 'even head dimension'):
      attention.init(jax.random.PRNGKey(0), jnp.zeros((1, 2, 3)))

  def test_attention_rotates_queries_and_keys_but_not_values(self):
    def attention(positional_encoding):
      def forward(inputs, positions):
        return transformer.MultiHeadDotProductAttention(
            num_heads=1,
            num_hiddens_per_head=4,
            attention_block_size=None,
            positional_encoding=positional_encoding,
        )(
            inputs,
            inputs,
            query_positions=positions,
            key_positions=positions,
        )

      return hk.without_apply_rng(hk.transform(forward))

    inputs = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)
    positions = jnp.array([0, 4, 8])
    rotary_attention = attention(transformer.ROTARY_POSITION_ENCODING)
    plain_attention = attention(None)
    params = rotary_attention.init(jax.random.PRNGKey(3), inputs, positions)

    def projected_vectors(model):
      recorded = {}

      def record_attention(query, key, value, mask):
        del mask
        recorded.update(query=query, key=key, value=value)
        return value

      with mock.patch.object(
          transformer,
          '_dense_dot_product_attention',
          side_effect=record_attention,
      ):
        model.apply(params, inputs, positions)
      return recorded

    plain = projected_vectors(plain_attention)
    rotary = projected_vectors(rotary_attention)

    np.testing.assert_allclose(
        rotary['query'],
        transformer.apply_rotary_encoding(plain['query'], positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rotary['key'],
        transformer.apply_rotary_encoding(plain['key'], positions),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(rotary['value'], plain['value'])

  def test_rotary_mode_does_not_add_absolute_encodings_to_embeddings(self):
    config = transformer.TransformerConfig(
        vocab_size=16,
        embedding_dim=8,
        byte_group_size=4,
        positional_encoding=transformer.ROTARY_POSITION_ENCODING,
    )
    embed = hk.without_apply_rng(
        hk.transform(
            lambda targets: transformer.embed_grouped_history(targets, config)
        )
    )
    targets = jnp.zeros((1, 9), dtype=jnp.uint8)
    params = embed.init(jax.random.PRNGKey(4), targets[:, :1])

    embeddings = embed.apply(params, targets)

    expected = jnp.broadcast_to(embeddings[0, 0], embeddings.shape)
    np.testing.assert_array_equal(embeddings, expected)

  def test_grouped_positions_retain_byte_spacing(self):
    positions = transformer._grouped_position_ids(
        batch_size=2,
        num_group_positions=3,
        group_size=4,
    )
    expected_for_one_batch = np.array(
        [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(positions[:4], expected_for_one_batch)
    np.testing.assert_array_equal(positions[4:], expected_for_one_batch)


if __name__ == '__main__':
  absltest.main()
