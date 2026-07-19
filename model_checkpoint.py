"""Saves and loads model parameters together with their architecture."""

import dataclasses
import json
import os

import haiku as hk
import numpy as np

from language_modeling_is_compression import transformer


_CONFIG_KEY = '__transformer_config__'


def save(
    path: str | os.PathLike[str],
    params: hk.Params,
    config: transformer.TransformerConfig,
) -> None:
  """Saves model parameters and the config required to use them."""
  serialized_config = json.dumps(dataclasses.asdict(config), sort_keys=True)
  # Passing a file handle prevents numpy from silently appending ".npz" when a
  # caller deliberately chooses a different checkpoint suffix.
  with open(path, 'wb') as checkpoint_file:
    np.savez(
        checkpoint_file,
        **params,
        **{_CONFIG_KEY: np.asarray(serialized_config)},
    )


def load(
    path: str | os.PathLike[str],
    default_config: transformer.TransformerConfig,
) -> tuple[hk.Params, transformer.TransformerConfig]:
  """Loads parameters and config, including legacy parameter-only files."""
  with np.load(path, allow_pickle=True) as data:
    params = {
        key: data[key].item() for key in data.files if key != _CONFIG_KEY
    }
    if _CONFIG_KEY in data.files:
      serialized_config = str(data[_CONFIG_KEY].item())
      config_values = json.loads(serialized_config)
      # Checkpoints written before blockwise attention must keep the released
      # dense operation order, which can affect arithmetic-coded bitstreams at
      # floating-point rounding boundaries.
      config_values.setdefault('attention_block_size', None)
      # The former byte-level decoder is exactly the group-size-one case. Its
      # parameter names and shapes remain valid under that configuration.
      config_values.setdefault('byte_group_size', 1)
      config = transformer.TransformerConfig(**config_values)
    else:
      config = dataclasses.replace(
          default_config,
          attention_block_size=None,
          byte_group_size=1,
      )
  return params, config
