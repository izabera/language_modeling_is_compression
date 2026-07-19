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

"""Tests for periodic compression-ratio reporting."""

from unittest import mock

from absl.testing import absltest

from language_modeling_is_compression import compress


class _FakeProgressBar:

  def __init__(self, iterable):
    self._iterable = iterable
    self.postfixes = []

  def __iter__(self):
    return iter(self._iterable)

  def set_postfix(self, postfix, refresh):
    self.postfixes.append((postfix, refresh))


class CompressTest(absltest.TestCase):

  def test_chunked_compression_reports_cumulative_ratio(self):
    progress_bar = _FakeProgressBar([b'ab', b'cdefgh'])

    with mock.patch.object(
        compress.tqdm, 'tqdm', return_value=progress_bar
    ) as tqdm_mock:
      ratio, _ = compress.evaluate_compressor_chunked(
          compress_fn=lambda _: b'x',
          get_data_generator_fn=lambda: iter([b'ab', b'cdefgh']),
          num_chunks=2,
          count_header_only_once=False,
      )

    tqdm_mock.assert_called_once()
    self.assertEqual(
        progress_bar.postfixes,
        [
            ({'ratio': '50.0%'}, False),
            ({'ratio': '25.0%'}, False),
        ],
    )
    self.assertEqual(ratio, 0.25)

  def test_reported_ratio_includes_mask_and_header_adjustments(self):
    progress_bar = _FakeProgressBar([b'abcd', b'efgh'])

    def mask_fn(data):
      return data, 1

    with mock.patch.object(compress.tqdm, 'tqdm', return_value=progress_bar):
      ratio, _ = compress.evaluate_compressor_chunked(
          compress_fn=lambda _: b'xyz',
          get_data_generator_fn=lambda: iter([b'abcd', b'efgh']),
          num_chunks=2,
          count_header_only_once=True,
          mask_fn=mask_fn,
      )

    self.assertEqual(progress_bar.postfixes[-1], ({'ratio': '39.9%'}, False))
    self.assertAlmostEqual(ratio, (6 * 64 / 62 - 3) / 8)


if __name__ == '__main__':
  absltest.main()
