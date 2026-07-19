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

"""Tests for Transformer checkpoint configuration compatibility."""

import dataclasses
import json
import pathlib
import tempfile

from absl.testing import absltest
import numpy as np

from language_modeling_is_compression import model_checkpoint
from language_modeling_is_compression import transformer


class ModelCheckpointTest(absltest.TestCase):
  def test_new_checkpoint_preserves_architecture_options(self):
    config = transformer.TransformerConfig(
      vocab_size=8,
      attention_block_size=64,
      byte_group_size=2,
      positional_encoding=transformer.ROTARY_POSITION_ENCODING,
    )
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'params.npz'
      model_checkpoint.save(path, {}, config)
      _, loaded_config = model_checkpoint.load(
        path, transformer.TransformerConfig(vocab_size=8)
      )

    self.assertEqual(loaded_config.attention_block_size, 64)
    self.assertEqual(loaded_config.byte_group_size, 2)
    self.assertEqual(
      loaded_config.positional_encoding,
      transformer.ROTARY_POSITION_ENCODING,
    )

  def test_old_serialized_config_keeps_dense_attention(self):
    config_values = dataclasses.asdict(
      transformer.TransformerConfig(vocab_size=8)
    )
    del config_values['attention_block_size']
    del config_values['byte_group_size']
    del config_values['positional_encoding']
    serialized_config = json.dumps(config_values, sort_keys=True)

    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'params.npz'
      with open(path, 'wb') as checkpoint_file:
        np.savez(
          checkpoint_file,
          __transformer_config__=np.asarray(serialized_config),
        )
      _, loaded_config = model_checkpoint.load(
        path, transformer.TransformerConfig(vocab_size=8)
      )

    self.assertIsNone(loaded_config.attention_block_size)
    self.assertEqual(loaded_config.byte_group_size, 1)
    self.assertEqual(
      loaded_config.positional_encoding,
      transformer.SINUSOIDAL_POSITION_ENCODING,
    )

  def test_parameter_only_checkpoint_keeps_dense_attention(self):
    with tempfile.TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / 'params.npz'
      with open(path, 'wb') as checkpoint_file:
        np.savez(checkpoint_file)
      _, loaded_config = model_checkpoint.load(
        path,
        transformer.TransformerConfig(vocab_size=8, attention_block_size=128),
      )

    self.assertIsNone(loaded_config.attention_block_size)
    self.assertEqual(loaded_config.byte_group_size, 1)
    self.assertEqual(
      loaded_config.positional_encoding,
      transformer.SINUSOIDAL_POSITION_ENCODING,
    )


if __name__ == '__main__':
  absltest.main()
