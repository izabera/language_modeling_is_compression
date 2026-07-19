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

"""Tests for Transformer training updates and stopping controls."""

from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tree

from language_modeling_is_compression import train
from language_modeling_is_compression import transformer


class TrainSpeedupsTest(absltest.TestCase):

  def _make_update_inputs(self):
    params = {'w': jnp.array([0.5, -0.25, 1.0], dtype=jnp.float32)}
    sequences = jnp.array([[1.0, 2.0, -1.0]], dtype=jnp.float32)

    def loss_fn(current_params, current_sequences):
      difference = current_params['w'] - current_sequences[0]
      return jnp.sum(jnp.square(difference))

    grad_fn = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    return params, opt_state, sequences, grad_fn, optimizer

  def test_one_token_initialization_preserves_full_input_parameter_tree(self):
    config = transformer.TransformerConfig(
        vocab_size=16,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        attention_block_size=4,
    )
    initialization_inputs = []

    def decoder(targets):
      initialization_inputs.append((targets.shape, targets.dtype))
      return transformer.transformer_decoder(targets, config)

    model = hk.transform(decoder)
    compact_params = train._initialize_parameters(model, seed=7)
    self.assertEqual(initialization_inputs[-1][0], (1, 1))
    self.assertEqual(initialization_inputs[-1][1], jnp.uint8)

    full_params = model.init(
        jax.random.PRNGKey(7), np.zeros((3, 9), dtype=np.uint8)
    )
    self.assertEqual(
        jax.tree.structure(compact_params), jax.tree.structure(full_params)
    )
    for compact, full in zip(
        jax.tree.leaves(compact_params), jax.tree.leaves(full_params)
    ):
      self.assertEqual(compact.shape, full.shape)
      self.assertEqual(compact.dtype, full.dtype)
      np.testing.assert_array_equal(compact, full)

  def test_nonlogging_update_skips_norm_and_reuses_compilation(self):
    params, opt_state, sequences, grad_fn, optimizer = (
        self._make_update_inputs()
    )
    traced_norm_calls = []
    original_global_norm = optax.global_norm

    def recording_global_norm(updates):
      traced_norm_calls.append(None)
      return original_global_norm(updates)

    with mock.patch.object(
        train.optax, 'global_norm', side_effect=recording_global_norm
    ):
      for _ in range(2):
        params, opt_state, logs = train._update_parameters(
            params,
            opt_state,
            sequences,
            grad_fn=grad_fn,
            optimizer=optimizer,
            compute_grad_norm=False,
        )
        jax.block_until_ready((params, opt_state, logs))
        self.assertTrue(jnp.isnan(logs['grad_norm_unclipped']))

      _, expected_grad = grad_fn(params, sequences)
      expected_grad = tree.map_structure(
          lambda gradient: gradient / sequences.shape[1], expected_grad
      )
      expected_norm = original_global_norm(expected_grad)

      first_logging_norm = None
      for update_index in range(2):
        params, opt_state, logs = train._update_parameters(
            params,
            opt_state,
            sequences,
            grad_fn=grad_fn,
            optimizer=optimizer,
            compute_grad_norm=True,
        )
        jax.block_until_ready((params, opt_state, logs))
        self.assertTrue(jnp.isfinite(logs['grad_norm_unclipped']))
        if update_index == 0:
          first_logging_norm = logs['grad_norm_unclipped']

      # Both predicate values share one trace and one stable output pytree.
      self.assertLen(traced_norm_calls, 1)
      np.testing.assert_allclose(
          first_logging_norm, expected_norm, rtol=1e-6
      )

  def test_compiled_update_donates_parameter_and_optimizer_buffers(self):
    params, opt_state, sequences, grad_fn, optimizer = (
        self._make_update_inputs()
    )
    executable = train._update_parameters.lower(
        params,
        opt_state,
        sequences,
        grad_fn=grad_fn,
        optimizer=optimizer,
        compute_grad_norm=False,
    ).compile()
    memory = executable.memory_analysis()
    if memory is None:
      self.skipTest('The current JAX backend does not report memory aliases.')

    donated_bytes = sum(
        leaf.nbytes for leaf in jax.tree.leaves((params, opt_state))
    )
    self.assertGreaterEqual(memory.alias_size_in_bytes, donated_bytes)

  def test_training_only_requests_gradient_norm_on_logging_steps(self):
    config = transformer.TransformerConfig(
        vocab_size=8,
        embedding_dim=4,
        num_layers=1,
        num_heads=2,
        widening_factor=2,
        attention_block_size=2,
    )
    norm_requests = []

    def fake_update(params, opt_state, **kwargs):
      compute_grad_norm = kwargs['compute_grad_norm']
      norm_requests.append(compute_grad_norm)
      logs = {
          'loss': jnp.asarray(float(len(norm_requests))),
          'grad_norm_unclipped': jnp.asarray(
              1.0 if compute_grad_norm else jnp.nan
          ),
      }
      return params, opt_state, logs

    with mock.patch.object(
        train.data_loaders,
        'get_enwik9_iterator',
        return_value=iter([b'\x01\x02']),
    ), mock.patch.object(
        train, '_update_parameters', side_effect=fake_update
    ):
      _, last_loss = train.train_transformer_decoder(
          training_steps=5,
          log_every=2,
          batch_size=1,
          sequence_length=2,
          use_tqdm=False,
          model_config=config,
      )

    self.assertEqual(norm_requests, [True, False, True, False, True])
    self.assertEqual(last_loss, 5.0)


class ConvergenceStoppingTest(absltest.TestCase):

  def test_cli_defaults_use_paper_training_horizon(self):
    self.assertEqual(train._BATCH_SIZE.default, 32)
    self.assertEqual(train._TRAINING_STEPS.default, 1_000_000)
    self.assertEqual(train._CONVERGENCE_PATIENCE.default, 5)

  def test_monitor_stops_after_patience(self):
    monitor = train._ConvergenceMonitor(patience=2, min_delta=0.1)

    self.assertFalse(monitor.update(3.0))
    self.assertFalse(monitor.update(3.0))
    self.assertEqual(monitor.windows_without_improvement, 1)
    self.assertTrue(monitor.update(3.0))

  def test_improvement_at_min_delta_resets_patience(self):
    monitor = train._ConvergenceMonitor(patience=2, min_delta=0.25)

    self.assertFalse(monitor.update(4.0))
    self.assertFalse(monitor.update(3.875))
    self.assertEqual(monitor.windows_without_improvement, 1)
    self.assertFalse(monitor.update(3.75))
    self.assertEqual(monitor.windows_without_improvement, 0)
    self.assertEqual(monitor.best_mean_loss, 3.75)

  def test_monitor_rejects_nonfinite_loss(self):
    monitor = train._ConvergenceMonitor(patience=1, min_delta=0.0)

    for loss in (float('nan'), float('inf'), -float('inf')):
      with self.subTest(loss=loss):
        with self.assertRaisesRegex(ValueError, 'must be finite'):
          monitor.update(loss)

  def test_validation_rejects_invalid_convergence_settings(self):
    config = transformer.TransformerConfig(
        vocab_size=256,
        embedding_dim=4,
        byte_group_size=1,
        num_layers=1,
        num_heads=2,
        widening_factor=2,
        attention_block_size=2,
    )
    valid_arguments = dict(
        training_steps=10,
        log_every=1,
        batch_size=1,
        sequence_length=2,
        learning_rate=1e-4,
        optimizer_name='adam',
        seed=0,
        momentum=0.9,
        weight_decay=0.0,
        gradient_clip_norm=0.0,
        config=config,
        convergence_patience=1,
        convergence_min_delta=0.01,
        convergence_window=2,
    )
    invalid_settings = (
        ('convergence_patience', -1, 'must be non-negative'),
        ('convergence_min_delta', -0.1, 'must be non-negative'),
        ('convergence_min_delta', float('nan'), 'must be finite'),
        ('convergence_window', 0, 'must be positive'),
        ('training_steps', 3, 'must be at least 4'),
    )

    for name, value, error_pattern in invalid_settings:
      with self.subTest(name=name, value=value):
        arguments = dict(valid_arguments)
        arguments[name] = value
        with self.assertRaisesRegex(ValueError, error_pattern):
          train._validate_training_arguments(**arguments)

  def test_training_stops_early_and_returns_last_observed_loss(self):
    config = transformer.TransformerConfig(
        vocab_size=8,
        embedding_dim=4,
        num_layers=1,
        num_heads=2,
        widening_factor=2,
        attention_block_size=2,
    )
    losses = iter([8.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 6.0])
    update_count = 0

    def fake_update(params, opt_state, **kwargs):
      del kwargs
      nonlocal update_count
      update_count += 1
      logs = {
          'loss': jnp.asarray(next(losses)),
          'grad_norm_unclipped': jnp.asarray(jnp.nan),
      }
      return params, opt_state, logs

    with mock.patch.object(
        train.data_loaders,
        'get_enwik9_iterator',
        return_value=iter([b'\x01\x02']),
    ), mock.patch.object(
        train, '_update_parameters', side_effect=fake_update
    ):
      _, last_loss = train.train_transformer_decoder(
          training_steps=10,
          log_every=0,
          batch_size=1,
          sequence_length=2,
          use_tqdm=False,
          model_config=config,
          convergence_patience=2,
          convergence_min_delta=0.1,
          convergence_window=2,
      )

    self.assertEqual(update_count, 8)
    self.assertEqual(last_loss, 7.0)

  @flagsaver.flagsaver(
      training_steps=20,
      convergence_patience=3,
      convergence_min_delta=0.02,
      convergence_window=4,
      output_path='unused-checkpoint.npz',
      use_tqdm=False,
  )
  def test_main_forwards_training_controls(self):
    config = transformer.TransformerConfig(
        vocab_size=256,
        embedding_dim=4,
        byte_group_size=1,
        num_layers=1,
        num_heads=2,
        widening_factor=2,
        attention_block_size=2,
    )

    flags_were_parsed = flags.FLAGS.is_parsed()
    if not flags_were_parsed:
      flags.FLAGS.mark_as_parsed()
    try:
      with mock.patch.object(
          train, '_resolve_model_config', return_value=config
      ), mock.patch.object(
          train, '_validate_output_path'
      ), mock.patch.object(
          train, 'train_transformer_decoder', return_value=({}, 2.0)
      ) as train_model, mock.patch.object(train.model_checkpoint, 'save'):
        train.main(['train.py'])
    finally:
      if not flags_were_parsed:
        flags.FLAGS.unparse_flags()

    training_arguments = train_model.call_args.kwargs
    self.assertEqual(training_arguments['training_steps'], 20)
    self.assertEqual(training_arguments['convergence_patience'], 3)
    self.assertEqual(training_arguments['convergence_min_delta'], 0.02)
    self.assertEqual(training_arguments['convergence_window'], 4)


if __name__ == '__main__':
  absltest.main()
