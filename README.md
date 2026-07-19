# Language Modeling is Compression

<p align="center">
  <img src="https://raw.githubusercontent.com/google-deepmind/language_modeling_is_compression/master/overview.png" alt="Overview figure"/>
</p>

This repository provides an implementation of our ICLR 2024 paper [Language Modeling is Compression](https://arxiv.org/abs/2309.10668).

> It has long been established that predictive models can be transformed into lossless compressors and
vice versa. Incidentally, in recent years, the machine learning community has focused on training
increasingly large and powerful self-supervised (language) models. Since these large language models
exhibit impressive predictive capabilities, they are well-positioned to be strong compressors. In this
work, we advocate for viewing the prediction problem through the lens of compression and evaluate
the compression capabilities of large (foundation) models. We show that large language models are
powerful general-purpose predictors and that the compression viewpoint provides novel insights into
scaling laws, tokenization, and in-context learning. For example, Chinchilla 70B, while trained primarily
on text, compresses ImageNet patches to 43.4% and LibriSpeech samples to 16.4% of their raw size,
beating domain-specific compressors like PNG (58.5%) or FLAC (30.3%), respectively. Finally, we show
that the prediction-compression equivalence allows us to use any compressor (like gzip) to build a
conditional generative model.

It contains all the code necessary to reproduce the experiments, including the
training of small Transformer language models on enwik8 to retrieve the neural
networks' weights. Chinchilla's weights are not provided.


## Content

```
.
├── compressors
|   ├── compressor.py      - Defines a protocol for compressors.
|   ├── flac.py            - Lossless audio compressor FLAC (Coalson, 2008).
|   ├── language_model.py  - Interface for language models, and compression function using arithmetic coding.
|   └── png.py             - Lossless image compressor PNG (Boutell, 1997).
├── arithmetic_coder.py    - Arithmetic Encoder and Decoder (Pasco, 1977).
├── compress.py            - Script to compress data.
├── constants.py           - Various constants like sequence length, alphabet size etc.
├── data_loaders.py        - Defines all our datasets.
├── README.md
├── requirements.txt       - Dependencies.
├── train.py               - Script to train a language model on Enwik8.
├── transformer.py         - Code for the Transformer model (Vaswani, 2017).
└── utils.py               - Utilities like converting a sequence of bits to bytes.
```

`compressors` contains all our compressors, either classical (like PNG or FLAC), or combining a predictor and an arithmetic coder (language models).
They all follow the protocol `Compressor`, defined in `compressors/compressor.py`.


## Installation

`pip install -r requirements.txt` will install all required dependencies.
This is best done inside a [conda environment](https://www.anaconda.com/).
To that end, install [Anaconda](https://www.anaconda.com/download#downloads).

Then, run the following commands:

```bash
# Clone the source code into a local directory:
git clone https://github.com/google-deepmind/language_modeling_is_compression.git
cd language_modeling_is_compression

# Create and activate the conda environment:
conda create --name lmic
conda activate lmic

# Install `pip` and use it to install all the dependencies:
conda install pip
pip install -r requirements.txt
```

If you have a GPU available (highly recommended for fast training), then you can install JAX with CUDA support.
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that the jax version must correspond to the existing CUDA installation you wish to use (CUDA 12 in the example above).
Please see the [JAX documentation](https://github.com/google/jax#installation) for more details.

## Usage

Before running any code, make sure to activate the conda environment and set the `PYTHONPATH`:

```bash
conda activate lmic
export PYTHONPATH=$(pwd)/..
```

If you want to compress with a language model, you need to train it first using:
```bash
python train.py
```

Training options are available as command-line flags. For example:

```bash
python train.py \
  --model_size=800k \
  --batch_size=32 \
  --training_steps=1000000 \
  --learning_rate=1e-4 \
  --optimizer=adam \
  --seed=0
```

`--training_steps` is a safety cap rather than a target. The CLI defaults to
one million optimizer updates, matching the authors' reported
[training configuration](https://github.com/google-deepmind/language_modeling_is_compression/issues/15#issuecomment-2231329302),
and stops sooner when the loss plateaus:

```bash
python train.py \
  --training_steps=1000000 \
  --batch_size=32 \
  --convergence_patience=10 \
  --convergence_min_delta=1e-3 \
  --convergence_min_passes=5 \
  --convergence_check_interval=0.25 \
  --convergence_validation_chunks=512
```

When convergence stopping is enabled, 512 chunks are selected deterministically
across Enwik8 and removed from the training sampler. The same held-out chunks
are evaluated every time, avoiding the lucky-minibatch-record behavior of a
training-loss stopper. Checks begin after five Enwik8-equivalent passes of
sampled training bytes and then run every quarter pass. Defining the schedule
in data exposure instead of optimizer steps gives each batch size the same
volume of sampled training data between checks; optimization itself remains
batch-size dependent.

Patience resets when the fixed validation loss has cumulatively improved by at
least `1e-3` nats per byte. Training stops after ten checks without such an
improvement, and the lowest-validation-loss parameters are saved rather than
the parameters from the stopping step. Validation chunk-loss spread, best step,
and sampled-data exposure are included in the logs. `training_steps` remains a
safety cap, and the checkpoint at that cap is always evaluated before the best
parameters are selected.

This is a practical plateau heuristic, not proof that the optimizer has found
a global or asymptotic optimum. For comparisons against a fixed-step published
recipe, keep automatic stopping disabled.

Use `--convergence_patience=0` to disable validation stopping, train on every
Enwik8 chunk, and run for exactly `training_steps` updates. This is the setting
to use when reproducing a fixed-step training recipe exactly.

Attention uses an exact blockwise softmax by default. Query and key tiles are
256 tokens wide, so the model never creates the full score and probability
arrays. The causal mask is generated one tile at a time as well.
This is especially useful for the default 2048-byte sequences on CPU, where
JAX's XLA attention implementation is not a memory-efficient FlashAttention
kernel. The result is mathematically the same as dense attention, subject to
the usual small floating-point differences from a different summation order.

Use `--attention_block_size=N` to tune the memory/speed tradeoff. Smaller tiles
use less peak memory; `--attention_block_size=0` restores the released dense
path. The selected value is stored in new checkpoints. Checkpoints predating
blockwise attention are automatically kept on the dense path to preserve their
numerics.

New models use rotary positional encodings (RoPE), matching the paper's
positional-encoding choice. RoPE rotates every adjacent pair across the full
projected query and key head dimension, with a base of 10,000; values are not
rotated. Use `--positional_encoding=sinusoidal` to reproduce the released
implementation's fixed additive encodings.

Use `python train.py --help` to see all options, including architecture
overrides, gradient clipping, logging frequency, sequence length, and output
path. The available model-size presets have the following exact parameter
counts in this implementation:

| Preset | Embedding dimension | Layers | Heads | Parameters |
| --- | ---: | ---: | ---: | ---: |
| `200k` | 64 | 4 | 8 | 231,936 |
| `800k` | 128 | 4 | 8 | 856,832 |
| `3.2m` | 256 | 4 | 8 | 3,286,272 |
| `6.4m` | 256 | 8 | 8 | 6,441,216 |
| `38m` | 512 | 12 | 8 | 38,066,432 |

The labels describe parameter scale and preserve the released implementation's
default of eight attention heads. The
[authors' paper configuration notes](https://github.com/google-deepmind/language_modeling_is_compression/issues/15#issuecomment-2231329302)
specify four heads for the 200K and 800K experiments. Use `--num_heads=4` for
the paper's small-model head count. The
`6.4m` preset is the natural eight-layer extension of the four-layer `3.2m`
configuration.

The default batch size is 32, matching the authors' small-model experiments. It
is intended for the default 200K model; reduce `--batch_size` when using a
larger model or a memory-constrained device. The convergence schedule adjusts
its optimizer-step cadence to preserve sampled-byte exposure across batch
sizes. Checkpoints include the model configuration, so non-default model sizes
can be loaded for compression.

To evaluate the compression rates, use:
```bash
python compress.py
```

Pass `--model_path` to `compress.py` when the training checkpoint was written
somewhere other than the default `params.npz`.


## Citing This Work

```bibtex
@inproceedings{deletang2024language,
  author       = {Gr{\'{e}}goire Del{\'{e}}tang and
                  Anian Ruoss and
                  Paul{-}Ambroise Duquenne and
                  Elliot Catt and
                  Tim Genewein and
                  Christopher Mattern and
                  Jordi Grau{-}Moya and
                  Li Kevin Wenliang and
                  Matthew Aitchison and
                  Laurent Orseau and
                  Marcus Hutter and
                  Joel Veness},
  title        = {Language Modeling Is Compression},
  booktitle    = {{ICLR}},
  year         = {2024}
}
```


## License and Disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
