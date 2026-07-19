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

"""Tests for Transformer attention backend selection and causal masking."""

from unittest import mock

from absl.testing import absltest
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import transformer


class TransformerAttentionTest(absltest.TestCase):

  def _attention(self, *, use_flash_attention: bool):
    def forward(inputs, mask, *, is_causal):
      return transformer.MultiHeadDotProductAttention(
          num_heads=2,
          num_hiddens_per_head=8,
          use_flash_attention=use_flash_attention,
      )(inputs, inputs, mask=mask, is_causal=is_causal)

    return hk.without_apply_rng(hk.transform(forward))

  def _record_attention_primitive_call(
      self,
      *,
      use_flash_attention: bool,
      mask=None,
      is_causal: bool,
  ):
    recorded = {}

    def fake_dot_product_attention(
        query, key, value, bias=None, mask=None, **kwargs
    ):
      recorded.update(
          query=query,
          key=key,
          value=value,
          bias=bias,
          mask=mask,
          kwargs=kwargs,
      )
      return value

    inputs = jnp.arange(48, dtype=jnp.float32).reshape(1, 3, 16)
    attention = self._attention(use_flash_attention=use_flash_attention)
    with mock.patch.object(
        transformer.jnn,
        'dot_product_attention',
        side_effect=fake_dot_product_attention,
    ):
      params = attention.init(
          jax.random.PRNGKey(0), inputs, mask, is_causal=is_causal
      )
      recorded.clear()
      output = attention.apply(
          params, inputs, mask, is_causal=is_causal
      )

    return recorded, output

  def test_flash_attention_is_disabled_by_default(self):
    config = transformer.TransformerConfig(vocab_size=8)

    self.assertFalse(config.use_flash_attention)

  def test_cuda_backend_detection_distinguishes_rocm(self):
    cases = (
        ('cuda 12.9', True),
        ('rocm 7.0', False),
        ('cpu', False),
    )

    for platform_version, expected in cases:
      with self.subTest(platform_version=platform_version), mock.patch.object(
          transformer.jax_backend,
          'get_backend',
          return_value=mock.Mock(platform_version=platform_version),
      ):
        self.assertEqual(transformer.has_cuda_backend(), expected)

  def test_flash_attention_rejects_unsupported_head_dimensions(self):
    cases = (
        (7, 'must be a multiple of 8'),
        (136, 'must be at most 128'),
    )
    inputs = jnp.zeros((1, 2, 16), dtype=jnp.float32)

    for head_dimension, message in cases:
      with self.subTest(head_dimension=head_dimension):
        attention = hk.transform(
            lambda x: transformer.MultiHeadDotProductAttention(
                num_heads=1,
                num_hiddens_per_head=head_dimension,
                use_flash_attention=True,
            )(x, x)
        )
        with self.assertRaisesRegex(ValueError, message):
          attention.init(jax.random.PRNGKey(0), inputs)

  def test_standard_attention_preserves_legacy_manual_path(self):
    mask = jnp.array(
        [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=jnp.int32
    )
    inputs = jnp.arange(48, dtype=jnp.float32).reshape(1, 3, 16)
    attention = self._attention(use_flash_attention=False)

    with mock.patch.object(
        transformer.jnn,
        'dot_product_attention',
        side_effect=AssertionError('legacy path called fused attention'),
    ) as fused_attention:
      params = attention.init(
          jax.random.PRNGKey(0), inputs, mask, is_causal=False
      )
      output = attention.apply(
          params, inputs, mask, is_causal=False
      )
      boolean_mask_output = attention.apply(
          params, inputs, mask.astype(jnp.bool_), is_causal=False
      )

    fused_attention.assert_not_called()
    self.assertEqual(output.shape, (1, 3, 16))
    self.assertEqual(output.dtype, jnp.float32)
    np.testing.assert_array_equal(output, boolean_mask_output)

  def test_flash_attention_uses_cudnn_and_bfloat16(self):
    recorded, output = self._record_attention_primitive_call(
        use_flash_attention=True,
        is_causal=True,
    )

    self.assertEqual(output.shape, (1, 3, 16))
    self.assertEqual(output.dtype, jnp.float32)
    self.assertEqual(recorded['query'].dtype, jnp.bfloat16)
    self.assertEqual(recorded['key'].dtype, jnp.bfloat16)
    self.assertEqual(recorded['value'].dtype, jnp.bfloat16)
    self.assertIsNone(recorded['mask'])
    self.assertEqual(recorded['kwargs']['implementation'], 'cudnn')
    self.assertTrue(recorded['kwargs']['is_causal'])

  def test_flash_decoder_rejects_cpu_backend(self):
    config = transformer.TransformerConfig(
        vocab_size=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=1,
        use_flash_attention=True,
    )
    decoder = hk.transform(
        lambda targets: transformer.transformer_decoder(targets, config)
    )
    targets = jnp.array([[1, 2, 3]], dtype=jnp.uint8)

    with mock.patch.object(
        transformer, 'has_cuda_backend', return_value=False
    ), self.assertRaisesRegex(
        RuntimeError, 'requires a CUDA-enabled JAX GPU backend'
    ):
      decoder.init(jax.random.PRNGKey(0), targets)

  def test_decoder_requests_causal_attention_without_dense_mask(self):
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
        use_flash_attention=True,
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
    ) as tril, mock.patch.object(
        transformer, 'has_cuda_backend', return_value=True
    ):
      decoder.init(jax.random.PRNGKey(0), targets)

    tril.assert_not_called()
    self.assertLen(constructor_calls, config.num_layers)
    self.assertLen(attention_calls, config.num_layers)
    for constructor_kwargs in constructor_calls:
      self.assertTrue(constructor_kwargs['use_flash_attention'])
    for positional_args, call_kwargs in attention_calls:
      self.assertEmpty(positional_args)
      self.assertNotIn('mask', call_kwargs)
      self.assertTrue(call_kwargs['is_causal'])


if __name__ == '__main__':
  absltest.main()
