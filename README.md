# lm-tts
Decoder-only language model for end-to-end text-to-speech using neural audio codec tokens.

## Usage
### Copy Synthesis with NeuCodec
To perform copy synthesis using the NeuCodec neural audio codec
```python
python scripts/copy_synthesis_neucodec.py \
    --input_wav <Path to the input speech wav file> \
    --output_wav <Path to save the reconstructed wav file> \
    --model_name <Name of the NeuCodec model to use. Options: neucodec, distill-neucodec>
```