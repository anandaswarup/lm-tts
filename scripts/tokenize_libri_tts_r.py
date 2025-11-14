"""
Tokenize LibriTTS_R dataset using Neucodec audio codec and upload to Hugging Face.

This script processes the LibriTTS_R dataset by:
    1. Loading audio files and metadata from the LibriTTS_R directory structure
    2. Encoding audio using Neucodec to discrete codes (single codebook)
    3. Creating a dataset with id, audio_duration, text_normalized, text_original, and codes (flat list)
    4. Uploading the tokenized dataset to Hugging Face Hub

Usage:
    python tokenize_libri_tts_r.py \
        --dataset_root /path/to/LibriTTS_R \
        --hf_dataset_id your-username/dataset-name \
        --split train-clean-100 \
        --batch_size 32

Note:
    - Audio is resampled to 16kHz for Neucodec encoding
    - Codes are stored as 1D arrays (flat lists) since Neucodec uses a single codebook
    - Requires authentication with Hugging Face Hub
"""

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from datasets import Dataset
from neucodec import NeuCodec
from torchaudio import transforms as T
from tqdm import tqdm


def load_libri_tts_r_metadata(dataset_root: Path, split: str) -> List[Dict]:
    """
    Load metadata from LibriTTS_R dataset directory structure.

    LibriTTS_R structure:
        dataset_root/
            {split}/
                {speaker_id}/
                    {chapter_id}/
                        {speaker_id}_{chapter_id}_{utterance_id}.wav
                        {speaker_id}_{chapter_id}_{utterance_id}.normalized.txt
                        {speaker_id}_{chapter_id}_{utterance_id}.original.txt

    Args:
        dataset_root: Root directory of LibriTTS_R dataset
        split: Split name (e.g., 'train-clean-100', 'dev-clean', 'test-clean')

    Returns:
        List of dictionaries containing metadata for each sample
    """
    split_path = dataset_root / split
    if not split_path.exists():
        raise ValueError(f"Split directory not found: {split_path}")

    samples = []

    # Iterate through speaker directories
    for speaker_dir in sorted(split_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        # Iterate through chapter directories
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            # Find all wav files in chapter directory
            for wav_file in sorted(chapter_dir.glob("*.wav")):
                # Extract sample ID from filename
                sample_id = wav_file.stem

                # Construct paths for text files
                normalized_txt = wav_file.with_suffix(".normalized.txt")
                original_txt = wav_file.with_suffix(".original.txt")

                # Read text content
                text_normalized = (
                    normalized_txt.read_text().strip()
                    if normalized_txt.exists()
                    else ""
                )
                text_original = (
                    original_txt.read_text().strip() if original_txt.exists() else ""
                )

                samples.append(
                    {
                        "id": sample_id,
                        "audio_path": str(wav_file),
                        "text_normalized": text_normalized,
                        "text_original": text_original,
                    }
                )

    return samples


def encode_audio_to_codes(
    audio_path: str,
    model: NeuCodec,
    device: str,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Encode audio file to discrete codes using Neucodec.

    Args:
        audio_path: Path to audio file
        model: Pre-initialized Neucodec model
        device: Device to use for inference
        target_sample_rate: Target sample rate for encoding (default: 16000)

    Returns:
        1D tensor of shape (num_frames,) containing discrete codes (single codebook)
    """
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to target sample rate if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Move to device and add batch dimension if needed
    waveform = waveform.to(device)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1, 1, num_samples]

    # Encode to discrete codes
    with torch.no_grad():
        codes = model.encode_code(waveform)  # [batch, num_codebooks, num_frames]

    # Remove batch dimension and flatten to 1D (single codebook)
    codes = codes.squeeze(0).squeeze(0).cpu()  # [num_frames]

    return codes


def process_dataset(
    samples: List[Dict],
    model: NeuCodec,
    device: str,
    batch_size: int = 32,
) -> List[Dict]:
    """
    Process all samples in the dataset to extract codes and audio duration.

    Args:
        samples: List of sample metadata dictionaries
        model: Pre-initialized Neucodec model
        device: Device to use for inference
        batch_size: Batch size for processing (currently processes one at a time)

    Returns:
        List of processed samples with codes (1D arrays) and audio_duration
    """
    processed_samples = []

    for sample in tqdm(samples, desc="Tokenizing audio"):
        try:
            # Encode audio to codes
            codes = encode_audio_to_codes(
                sample["audio_path"],
                model,
                device,
            )

            # Calculate audio duration
            waveform, sample_rate = torchaudio.load(sample["audio_path"])
            duration = waveform.shape[-1] / sample_rate

            # Create processed sample
            processed_sample = {
                "id": sample["id"],
                "audio_duration": float(duration),
                "text_normalized": sample["text_normalized"],
                "text_original": sample["text_original"],
                "codes": codes.numpy(),  # Flat list of codes (single codebook)
            }

            processed_samples.append(processed_sample)

        except Exception as e:
            print(f"Error processing {sample['id']}: {e}")
            continue

    return processed_samples


def create_hf_dataset(processed_samples: List[Dict]) -> Dataset:
    """
    Create Hugging Face Dataset from processed samples.

    Args:
        processed_samples: List of processed sample dictionaries with flat code arrays

    Returns:
        Hugging Face Dataset object with columns: id, audio_duration, text_normalized, text_original, codes
    """
    # Convert list of dicts to dict of lists
    data_dict = {
        "id": [s["id"] for s in processed_samples],
        "audio_duration": [s["audio_duration"] for s in processed_samples],
        "text_normalized": [s["text_normalized"] for s in processed_samples],
        "text_original": [s["text_original"] for s in processed_samples],
        "codes": [s["codes"] for s in processed_samples],
    }

    # Create dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize LibriTTS_R dataset using Neucodec and upload to HuggingFace"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of LibriTTS_R dataset",
    )
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        required=True,
        help="Hugging Face dataset ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train-clean-100",
        help="Dataset split to process (default: train-clean-100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded dataset private",
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    dataset_root = Path(args.dataset_root)

    # Validate dataset root exists
    if not dataset_root.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    print(f"Loading LibriTTS_R metadata from {dataset_root}/{args.split}...")
    samples = load_libri_tts_r_metadata(dataset_root, args.split)
    print(f"Found {len(samples)} samples")

    # Initialize Neucodec model
    print(f"Initializing Neucodec model on {args.device}...")
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model = model.to(args.device)
    model.eval()

    # Process dataset
    print("Processing dataset...")
    processed_samples = process_dataset(
        samples,
        model,
        args.device,
        args.batch_size,
    )
    print(f"Successfully processed {len(processed_samples)} samples")

    # Create Hugging Face dataset
    print("Creating Hugging Face dataset...")
    dataset = create_hf_dataset(processed_samples)

    # Print dataset info
    print("\nDataset info:")
    print(dataset)
    print("\nFirst sample:")
    print(dataset[0])

    # Upload to Hugging Face Hub
    print(f"\nUploading dataset to {args.hf_dataset_id}...")
    dataset.push_to_hub(
        args.hf_dataset_id,
        private=args.private,
        split=args.split,
    )

    print(f"✓ Dataset uploaded successfully to {args.hf_dataset_id}")


if __name__ == "__main__":
    main()
