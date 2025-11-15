"""
Script to tokenize LibriTTS_R dataset using Neucodec audio codec and upload to HuggingFace datasets.

Usage: python tokenize_and_upload.py --repo-name username/dataset --batch-size batch_size_value
"""

import argparse
import os
from typing import Any, Dict

import torch
import torchaudio
from datasets import DatasetDict, load_dataset
from neucodec import NeuCodec


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
            # encode_code expects [B, 1, T] at 16kHz on CPU
            # The model internally handles moving to GPU for encoding
            codes = codec.encode_code(waveform_16k)
            # codes shape is [B, 1, F] where F is the number of frames

            # Convert to list for storage in dataset
            codes_list = codes.squeeze(0).cpu().tolist()

        codecs_list.append(codes_list)

    # Remove audio field and add codecs field
    batch["codecs"] = codecs_list
    del batch["audio"]

    return batch


def tokenize_dataset(
    repo_name: str,
    batch_size: int = 32,
    private: bool = False,
    cache_dir: str | None = None,
) -> None:
    """
    Tokenize LibriTTS_R clean subsets and upload to HuggingFace.

    Args:
        repo_name: HuggingFace repo name (e.g., 'username/libritts_r_neucodec')
        batch_size: Batch size for processing
        private: Whether to make the dataset private
        cache_dir: Directory for caching datasets
    """
    # Splits to process (clean subsets only)
    splits = ["train.clean.360", "train.clean.100", "dev.clean", "test.clean"]

    # Initialize Neucodec model
    print("Initializing Neucodec model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec = codec.to(device)
    codec.eval()
    print("Neucodec model loaded successfully\n")

    # Process each split
    dataset_dict = DatasetDict()

    for split_name in splits:
        print("=" * 60)
        print(f"Processing split: {split_name}")
        print("=" * 60)

        # Load the dataset split
        print(f"Loading dataset split: {split_name}")
        dataset = load_dataset(
            "mythicinfinity/libritts_r",
            "all",
            split=split_name,
            cache_dir=cache_dir,
        )

        try:
            print(f"Split {split_name} loaded with {len(dataset)} samples")  # type: ignore
        except TypeError:
            print(f"Split {split_name} loaded (streaming mode)")

        # Process the dataset
        print(f"Tokenizing audio in split: {split_name}")
        tokenized_dataset = dataset.map(
            lambda batch: encode_audio_with_neucodec(batch, codec),
            batched=True,
            batch_size=batch_size,
            remove_columns=["audio"],
        )

        try:
            num_samples = len(tokenized_dataset)  # type: ignore
            print(f"Split {split_name} tokenized. Total samples: {num_samples}\n")
        except TypeError:
            print(f"Split {split_name} tokenized (streaming mode)\n")

        # Add to dataset dict
        dataset_dict[split_name] = tokenized_dataset  # type: ignore

    # Upload to HuggingFace
    print("=" * 60)
    print(f"Uploading to HuggingFace: {repo_name}")
    print(f"Splits: {list(dataset_dict.keys())}")
    print(f"Private: {private}")
    print("=" * 60)

    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set it with: export HF_TOKEN=your_token"
        )

    # Push to hub
    dataset_dict.push_to_hub(
        repo_name,
        token=hf_token,
        private=private,
    )

    print("\n" + "=" * 60)
    print("Upload complete!")
    print(f"Dataset available at: https://huggingface.co/datasets/{repo_name}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize LibriTTS_R dataset with Neucodec and upload to HuggingFace"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="HuggingFace repo name (e.g., 'username/libritts_r_neucodec')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching datasets",
    )

    args = parser.parse_args()

    tokenize_dataset(
        repo_name=args.repo_name,
        batch_size=args.batch_size,
        private=args.private,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
