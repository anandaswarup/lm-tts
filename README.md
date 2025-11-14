# LM-TTS

## Overview

Language model based Text to Speech synthesis.

## Usage
All scripts in this repository will use Modal's cloud GPU infrastructure. Create a Modal account if you don't have one,
and follow the instructions to setup / configure Modal on your machine.

### Tokenizing LibriTTS_R dataset with Neucodec and uploading to HuggingFace datasets
#### Deploy the App on Modal
```bash
modal deploy tokenize_libritts_r.py
```
This uploads the `tokeinize_libritts_r.py` script to Modal. 

#### Run the Script
```bash
modal run tokenize_libritts_r.py::app.process_dataset \
    --subset all \
    --repo-name your-username/libritts_r_neucodec \
    --batch-size 32
```