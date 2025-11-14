"""
Script to tokenize LibriTTS_R dataset using Neucodec audio codec. Runs on Modal with A100-80GB GPU and uploads to 
HuggingFace datasets.

Usage:
 Deploy and run
    - modal deploy tokenize_libritts_r.py
    - modal run tokenize_libritts_r.py::app.process_dataset --subset all --repo-name user/dataset
"""

import os
from typing import Any, Dict

import modal
import torch
import torchaudio
from datasets import DatasetDict, load_dataset
from neucodec import NeuCodec

# Modal setup
app = modal.App("libritts-r-tokenizer")

# Create Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasets[audio]>=4.2.0",
    "torch>=2.9.0",
    "torchaudio>=2.9.0",
    "neucodec>=0.0.4",
    "transformers>=4.57.1",
    "huggingface-hub>=0.27.0",
)

# Create Modal volume for caching
volume = modal.Volume.from_name("libritts-cache", create_if_missing=True)

# HuggingFace secrets for authentication
HF_TOKEN = modal.Secret.from_name("HF_TOKEN")


def encode_audio_with_neucodec(
    batch: Dict[str, Any],
    codec: NeuCodec,
    target_sample_rate: int = 24000,
) -> Dict[str, Any]:
    """
    Encode audio samples in a batch using Neucodec.

    Args:
        batch: Batch from HuggingFace dataset containing audio samples
        codec: Initialized NeuCodec model
        target_sample_rate: Expected sample rate (24000 Hz for LibriTTS-R)

    Returns:
        Batch with 'codecs' field replacing 'audio' field
    """
    device = next(codec.parameters()).device

    # Process each audio sample in the batch
    codecs_list = []

    for audio_dict in batch["audio"]:
        # Load audio array and sample rate
        audio_array = audio_dict["array"]
        sample_rate = audio_dict["sampling_rate"]

        # Convert to torch tensor
        waveform = torch.tensor(audio_array, dtype=torch.float32)

        # Add channel dimension if needed (Neucodec expects [channels, samples])
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if necessary
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            )
            waveform = resampler(waveform)

        # Encode with Neucodec
        # Neucodec expects shape [B, 1, T] at 16kHz for encode_code
        # But we're working at 24kHz, so we need to resample to 16kHz first
        if target_sample_rate != 16000:
            resampler_16k = torchaudio.transforms.Resample(
                orig_freq=target_sample_rate, new_freq=16000
            )
            waveform_16k = resampler_16k(waveform)
        else:
            waveform_16k = waveform

        # Add batch dimension if needed
        if waveform_16k.ndim == 2:
            waveform_16k = waveform_16k.unsqueeze(0)

        # Encode with Neucodec - returns FSQ codes
        with torch.no_grad():
            # encode_code expects [B, 1, T] at 16kHz
            codes = codec.encode_code(waveform_16k.to(device))
            # codes shape is [B, 1, F] where F is the number of frames

            # Convert to list for storage in dataset
            codes_list = codes.squeeze(0).cpu().tolist()

        codecs_list.append(codes_list)

    # Remove audio field and add codecs field
    batch["codecs"] = codecs_list
    del batch["audio"]

    return batch


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    volumes={"/cache": volume},
    secrets=[HF_TOKEN],
    timeout=86400,  # 24 hours timeout
)
def tokenize_split(
    split_name: str,
    subset: str = "all",
    batch_size: int = 32,
) -> tuple[str, Any]:
    """
    Tokenize a single split of the LibriTTS_R dataset.

    Args:
        split_name: Name of the split (e.g., 'train.clean.100', 'dev.clean')
        subset: Subset configuration ('all', 'clean', 'other', 'dev')
        batch_size: Batch size for processing
        num_proc: Number of processes for parallel processing
    """
    print(f"Processing split: {split_name}")

    # Set cache directory
    cache_dir = "/cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)

    # Load the dataset split
    print(f"Loading dataset split: {split_name}")
    dataset = load_dataset(
        "mythicinfinity/libritts_r",
        subset,
        split=split_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Print dataset info (handle both Dataset and IterableDataset)
    try:
        print(f"Split {split_name} loaded with {len(dataset)} samples")  # type: ignore
    except TypeError:
        print(f"Split {split_name} loaded (streaming mode)")

    # Initialize Neucodec model
    print("Initializing Neucodec model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec = codec.to(device)
    codec.eval()

    print(f"Neucodec model loaded on {device}")

    # Process the dataset
    print(f"Tokenizing audio in split: {split_name}")
    tokenized_dataset = dataset.map(
        lambda batch: encode_audio_with_neucodec(batch, codec),
        batched=True,
        batch_size=batch_size,
        remove_columns=["audio"],
    )

    # Update features to include codecs instead of audio
    # Codecs will be stored as nested lists of integers
    try:
        print(f"Split {split_name} tokenized. Total samples: {len(tokenized_dataset)}")  # type: ignore
    except TypeError:
        print(f"Split {split_name} tokenized (streaming mode)")

    # Commit volume changes
    volume.commit()

    return split_name, tokenized_dataset


@app.function(
    image=image,
    secrets=[HF_TOKEN],
    timeout=3600,
)
def upload_to_huggingface(
    dataset_dict: DatasetDict,
    repo_name: str,
    private: bool = False,
) -> None:
    """
    Upload the tokenized dataset to HuggingFace Hub.

    Args:
        dataset_dict: DatasetDict containing all splits
        repo_name: Repository name on HuggingFace (e.g., 'username/libritts_r_neucodec')
        private: Whether to make the dataset private
    """
    print(f"Uploading dataset to HuggingFace: {repo_name}")

    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN")  # Push to hub
    dataset_dict.push_to_hub(
        repo_name,
        token=hf_token,
        private=private,
    )

    print(f"Dataset successfully uploaded to {repo_name}")


@app.function(
    image=image,
    secrets=[HF_TOKEN],
    timeout=86400,
)
def process_dataset(
    subset: str = "all",
    repo_name: str | None = None,
    batch_size: int = 32,
    splits: str | None = None,
):
    """
    Main function for tokenizing LibriTTS_R dataset (runs on Modal cloud).

    Args:
        subset: Which subset to process ('all', 'clean', 'other', 'dev')
        repo_name: HuggingFace repo name to upload to (e.g., 'username/libritts_r_neucodec')
        batch_size: Batch size for processing
        splits: Comma-separated list of splits to process (e.g., 'dev.clean,test.clean')
                If None, processes all splits for the subset
    """
    # Define available splits for each subset
    split_mapping = {
        "dev": ["dev.clean"],
        "clean": ["dev.clean", "test.clean", "train.clean.100", "train.clean.360"],
        "other": ["dev.other", "test.other", "train.other.500"],
        "all": [
            "dev.clean",
            "dev.other",
            "test.clean",
            "test.other",
            "train.clean.100",
            "train.clean.360",
            "train.other.500",
        ],
    }

    # Determine which splits to process
    if splits:
        split_list = [s.strip() for s in splits.split(",")]
    else:
        split_list = split_mapping.get(subset, split_mapping["all"])

    print(f"Starting tokenization for subset '{subset}'")
    print(f"Splits to process: {split_list}")

    # Process each split in parallel
    results = []
    for split_name in split_list:
        print(f"\n{'=' * 60}")
        print(f"Processing split: {split_name}")
        print(f"{'=' * 60}\n")

        result = tokenize_split.remote(
            split_name=split_name,
            subset=subset,
            batch_size=batch_size,
        )
        results.append((split_name, result))

    # Collect all processed splits into a DatasetDict
    print("\n" + "=" * 60)
    print("All splits processed. Preparing DatasetDict...")
    print("=" * 60 + "\n")

    dataset_dict = DatasetDict()
    for split_name, dataset in results:
        dataset_dict[split_name] = dataset

    # Upload to HuggingFace if repo_name is provided
    if repo_name:
        print(f"\nUploading to HuggingFace: {repo_name}")
        upload_to_huggingface.remote(dataset_dict, repo_name)
    else:
        print("\nNo repo_name provided. Skipping upload to HuggingFace.")
        print("To upload, run with --repo-name argument")

    print("\n" + "=" * 60)
    print("Tokenization complete!")
    print("=" * 60)


@app.local_entrypoint()
def main(
    subset: str = "all",
    repo_name: str | None = None,
    batch_size: int = 32,
    splits: str | None = None,
):
    """
    Local entrypoint - calls the remote process_dataset function.
    This allows the script to run entirely on Modal's cloud.
    """
    # Call the remote function - this runs on Modal's infrastructure
    process_dataset.remote(subset, repo_name, batch_size, splits)
