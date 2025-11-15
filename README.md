# LM-TTS

## Overview

Language model based Text to Speech synthesis.

## Usage

### Tokenizing LibriTTS_R dataset with Neucodec and uploading to HuggingFace

The tokenization script processes the clean subsets (`train.clean.360`, `train.clean.100`, `dev.clean`, `test.clean`) of LibriTTS_R using the Neucodec audio codec and uploads the tokenized dataset to HuggingFace.

#### Option 1: Run Locally (with your GPU/CPU)

```bash
# Set HuggingFace token
export HF_TOKEN=your_huggingface_token

# Run the script
python scripts/tokenize_and_upload.py \
    --repo-name your-username/libritts_r_neucodec \
    --batch-size batch_size_value
```

#### Option 2: Run on Modal Cloud GPU

Modal provides cloud GPU infrastructure. Create a Modal account if you don't have one, and follow the instructions to setup/configure Modal on your machine.

**Deploy the App on Modal:**
```bash
modal deploy scripts/run_tokenize_and_upload_on_modal.py
```

**Run the Script:**
```bash
modal run --detach scripts/run_tokenize_and_upload_on_modal.py::app.run \
    --repo-name your-username/libritts_r_neucodec \
    --batch-size batch_size_value
```

**Check logs:**
```bash
modal app logs libritts-r-tokenizer
```