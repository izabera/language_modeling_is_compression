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
  --batch_size=1 \
  --training_steps=1000 \
  --learning_rate=1e-4 \
  --optimizer=adam \
  --seed=0
```

Attention uses an exact blockwise softmax by default. Query and key tiles are
256 tokens wide, so the model never creates the full
`batch_size * num_heads * sequence_length * sequence_length` score and
probability arrays. The causal mask is generated one tile at a time as well.
This is especially useful for the default 2048-byte sequences on CPU, where
JAX's XLA attention implementation is not a memory-efficient FlashAttention
kernel. The result is mathematically the same as dense attention, subject to
the usual small floating-point differences from a different summation order.

Use `--attention_block_size=N` to tune the memory/speed tradeoff. Smaller tiles
use less peak memory; `--attention_block_size=0` restores the released dense
path. The selected value is stored in new checkpoints. Checkpoints created by
older versions are automatically kept on dense attention to preserve their
numerics.

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
specify four heads for the 200K and 800K experiments as well as rotary
positional encodings, whereas this repository implements fixed sinusoidal
encodings. Use `--num_heads=4` for the paper's small-model head count. The
`6.4m` preset is the natural eight-layer extension of the four-layer `3.2m`
configuration.

The default batch size is 1 to keep memory use modest. Checkpoints include the
model configuration, so non-default model sizes can be loaded for compression.

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
