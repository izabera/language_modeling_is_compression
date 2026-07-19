"""Saves and loads model parameters together with their architecture."""

import dataclasses
import json
import os
import tempfile

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
  path = os.fspath(path)
  directory = os.path.dirname(path) or '.'
  temporary_path: str | None = None
  try:
    # Keeping the temporary file beside the checkpoint makes os.replace atomic.
    # Passing a file handle also prevents numpy from silently appending ".npz"
    # when a caller deliberately chooses a different checkpoint suffix.
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=directory,
        prefix='.checkpoint-',
        suffix='.tmp',
        delete=False,
    ) as checkpoint_file:
      temporary_path = checkpoint_file.name
      np.savez(
          checkpoint_file,
          **params,
          **{_CONFIG_KEY: np.asarray(serialized_config)},
      )
      checkpoint_file.flush()
      os.fsync(checkpoint_file.fileno())
    os.replace(temporary_path, path)
    temporary_path = None
  finally:
    if temporary_path is not None:
      try:
        os.unlink(temporary_path)
      except FileNotFoundError:
        pass


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
      # Before rotary support, embeddings always received fixed sinusoidal
      # encodings. Preserve those numerics for inference and arithmetic
      # decoding when loading an older serialized configuration.
      config_values.setdefault(
          'positional_encoding', transformer.SINUSOIDAL_POSITION_ENCODING
      )
      config = transformer.TransformerConfig(**config_values)
    else:
      config = dataclasses.replace(
          default_config,
          attention_block_size=None,
          positional_encoding=transformer.SINUSOIDAL_POSITION_ENCODING,
      )
  return params, config
