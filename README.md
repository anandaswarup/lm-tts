# Introduction

LM-TTS is a single stage auto-regressive transformer model capable of generating speech samples conditioned on text prompts. The text prompts are passed through a frozen text encoder model to obtain a sequence of hidden-state representations. LM-TTS is then trained to predict discrete audio tokens, or audio codes, conditioned on these hidden-states. These audio tokens are then decoded using an audio compression model, such as EnCodec, to recover the audio waveform.

## Model structure

The LM-TTS model can be de-composed into three distinct stages:

1. Text encoder: maps the text inputs to a sequence of hidden-state representations. LM-TTS uses a frozen text encoder from T5.
2. LM-TTS decoder: a language model (LM) that auto-regressively generates audio tokens (or codes) conditional on the encoder hidden-state representations.
3. Audio decoder: used to recover the audio waveform from the audio tokens predicted by the decoder.ß