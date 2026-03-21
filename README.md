# LM-TTS: Language modeling based Text-to-Speech

## Introduction
This repository contains code for training a Text-to-Speech (TTS) model based on language modeling techniques. The model is trained to generate discrete audio tokens conditioned on text input, following which the generated tokens are decoded back into audio waveforms using a neural audio codec.

## Architecture

The TTS model consists of two main components:

1. **Language Model**: A transformer language model that takes text input and autoregressively generates discrete audio tokens. The model is trained using a cross-entropy loss function to predict the next token in the sequence given the previous tokens and the text input.

2. **Audio Tokenizer**: The audio tokenizer used in this project is [Neucodec](https://github.com/neuphonic/neucodec), which is a Finite Scalar Quantisation (FSQ) based 0.8kbps audio codec. Neucodec is used to tokenize the audio data into discrete tokens that are used for training the language model. During inference, the generated tokens from the language model are decoded back into audio waveforms using Neucodec.

### Language model architecture

The language model is a transformer language model, with the following features different from a vanilla transformer:
- RoPE instead of learned positional encodings
- RMSNorm everywhere instead of LayerNorm
- QK Normalization
- Untied embedding / unembedding weights
- $\text{ReLU}^2$ activations instead of GeLU or SwiGLU
- Logit softcapping
- Value Embeddings
- Per-layer residual scalars
- Sliding window attention
- Flash Attention 3 with fallback to PyTorch SDPA on unsupported hardware

The implementation is based on the language modeling implementation in [karpathy/nanochat](https://github.com/karpathy/nanochat)

## Training
### Optimizer design
The optimizer used for training the language model is a split optimizer design, where the parameters of the model are divided into two groups: those that are updated using AdamW and those that are updated using Muon. The split is based on the type of parameters, with embeddings and scalars updated using AdamW and weight matrices updated using Muon.

## Usage

**COMING SOON**

## References
- [Neucodec](https://github.com/neuphonic/neucodec)
- [karpathy/nanochat](https://github.com/karpathy/nanochat)