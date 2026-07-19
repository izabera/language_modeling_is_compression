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

"""Tests for TRACE-style target-aligned byte grouping."""

import functools
from unittest import mock

from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from language_modeling_is_compression import transformer


class ByteGroupingTest(absltest.TestCase):
  def _decoder(self, *, group_size=4, block_size=2):
    config = transformer.TransformerConfig(
      vocab_size=16,
      embedding_dim=8,
      byte_group_size=group_size,
      num_layers=2,
      num_heads=2,
      widening_factor=2,
      attention_block_size=block_size,
    )
    decoder = hk.without_apply_rng(
      hk.transform(functools.partial(transformer.transformer_decoder, config=config))
    )
    return config, decoder

  def test_default_uses_the_papers_robust_group_size(self):
    config = transformer.TransformerConfig(vocab_size=256)
    self.assertEqual(config.byte_group_size, 4)
    self.assertEqual(
      transformer.attention_block_size_in_group_tokens(config), 64
    )
    self.assertEqual(
      transformer.config_for_model_size('200k', vocab_size=256).byte_group_size,
      4,
    )

  def test_groups_are_target_aligned_preceding_bytes(self):
    targets = jnp.array([[10, 11, 12, 13, 14, 15]], dtype=jnp.uint8)

    groups = transformer._group_previous_bytes(targets, group_size=4)

    expected = np.array(
      [
        [
          [[0, 0, 0, 0], [10, 11, 12, 13]],
          [[0, 0, 0, 10], [11, 12, 13, 14]],
          [[0, 0, 10, 11], [12, 13, 14, 15]],
          [[0, 10, 11, 12], [13, 14, 15, 0]],
        ]
      ],
      dtype=np.uint8,
    )
    np.testing.assert_array_equal(groups, expected)

  def test_group_size_one_is_the_original_right_shift(self):
    targets = jnp.array([[7, 3, 9, 2]], dtype=jnp.uint8)

    groups = transformer._group_previous_bytes(targets, group_size=1)

    np.testing.assert_array_equal(groups[:, 0, :, 0], [[0, 7, 3, 9]])
    np.testing.assert_array_equal(groups[:, 0, :, 0], transformer.shift_right(targets))

  def test_attention_length_is_reduced_and_phases_fold_into_batch(self):
    recorded_shapes = []

    class RecordingAttention:
      def __init__(self, **unused_kwargs):
        pass

      def __call__(self, *unused_args, **kwargs):
        recorded_shapes.append(kwargs['inputs_q'].shape)
        return jnp.zeros_like(kwargs['inputs_q'])

    config = transformer.TransformerConfig(
      vocab_size=16,
      embedding_dim=8,
      byte_group_size=4,
      num_layers=1,
      num_heads=2,
      widening_factor=2,
      attention_block_size=2,
    )
    decoder = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
    )
    targets = jnp.arange(18, dtype=jnp.uint8).reshape(2, 9)

    with mock.patch.object(
      transformer, 'MultiHeadDotProductAttention', RecordingAttention
    ):
      params = decoder.init(jax.random.PRNGKey(0), targets)
      output = decoder.apply(params, None, targets)

    self.assertEqual(recorded_shapes, [(8, 3, 8), (8, 3, 8)])
    self.assertEqual(output.shape, (2, 9, 16))

  def test_current_and_future_bytes_cannot_change_a_logit(self):
    _, decoder = self._decoder()
    targets = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.uint8)
    params = decoder.init(jax.random.PRNGKey(3), targets[:, :1])
    changed = targets.at[:, 5:].set(jnp.array([[15, 14, 13, 12, 11]], dtype=jnp.uint8))

    original_output = decoder.apply(params, targets)
    changed_output = decoder.apply(params, changed)

    # Position 5 predicts targets[5], so it may only depend on positions < 5.
    np.testing.assert_array_equal(original_output[:, :6], changed_output[:, :6])
    self.assertGreater(
      float(jnp.max(jnp.abs(original_output[:, 6:] - changed_output[:, 6:]))),
      1e-7,
    )

  def test_full_sequence_matches_every_causal_prefix(self):
    _, decoder = self._decoder()
    targets = jnp.array([[2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4]], dtype=jnp.uint8)
    params = decoder.init(jax.random.PRNGKey(5), targets[:, :1])

    full_output = decoder.apply(params, targets)

    for target_index in range(targets.shape[1]):
      with self.subTest(target_index=target_index):
        prefix_output = decoder.apply(params, targets[:, : target_index + 1])
        np.testing.assert_allclose(
          full_output[:, target_index],
          prefix_output[:, -1],
          rtol=2e-5,
          atol=2e-5,
        )

  def test_parameter_formula_matches_grouped_parameter_tree(self):
    for group_size in (1, 2, 4):
      with self.subTest(group_size=group_size):
        config, decoder = self._decoder(group_size=group_size)
        params = decoder.init(jax.random.PRNGKey(7), jnp.zeros((1, 1), dtype=jnp.uint8))
        self.assertEqual(
          hk.data_structures.tree_size(params),
          transformer.parameter_count(config),
        )

  def test_group_size_must_be_positive_and_divide_model_width(self):
    for group_size in (0, 3):
      with self.subTest(group_size=group_size):
        config = transformer.TransformerConfig(
          vocab_size=16,
          embedding_dim=8,
          byte_group_size=group_size,
          num_layers=1,
          num_heads=2,
        )
        with self.assertRaisesRegex(
          ValueError, 'byte_group_size must be positive and divide embedding_dim'
        ):
          transformer.parameter_count(config)

  def test_empty_and_ragged_sequences_keep_byte_aligned_output(self):
    _, decoder = self._decoder()
    params = decoder.init(jax.random.PRNGKey(11), jnp.zeros((1, 1), dtype=jnp.uint8))

    for sequence_length in (0, 1, 3, 4, 5, 7):
      with self.subTest(sequence_length=sequence_length):
        targets = jnp.arange(sequence_length, dtype=jnp.uint8)[None, :]
        output = decoder.apply(params, targets)
        self.assertEqual(output.shape, (1, sequence_length, 16))


if __name__ == '__main__':
  absltest.main()
