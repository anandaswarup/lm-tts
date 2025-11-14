"""
Modal GPU script to tokenize LibriTTS_R dataset using Neucodec and upload to Hugging Face.

This script runs the tokenization process on Modal's GPU infrastructure:
    1. Mounts the local LibriTTS_R dataset to the Modal container
    2. Processes audio files using Neucodec on GPU
    3. Uploads the tokenized dataset to Hugging Face Hub

Usage:
    modal run tokenize_libritts_r_modal.py \
        --dataset-root /path/to/LibriTTS_R \
        --hf-dataset-id your-username/dataset-name

Note:
    - Requires Modal account and authentication (modal setup)
    - Requires HF_TOKEN secret configured in Modal
    - Uses A100 GPU by default
"""

import modal

# Create Modal app
app = modal.App("tokenize-libritts-r")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.9.0",
    "torchaudio>=2.9.0",
    "neucodec>=0.0.4",
    "datasets[audio]>=4.2.0",
    "tqdm>=4.66.0",
    "huggingface-hub>=0.20.0",
)


# Main tokenization function
@app.function(
    image=image,
    gpu="A100",  # Use A100 GPU (can change to "T4", "A100", etc.)
    timeout=86400,  # 24 hours timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF token for upload
)
def tokenize_split(
    dataset_root: str,
    hf_dataset_id: str,
    split: str,
    private: bool = False,
):
    """
    Tokenize a single split of LibriTTS_R dataset.

    Args:
        dataset_root: Path to LibriTTS_R dataset root (in Modal volume)
        hf_dataset_id: Hugging Face dataset ID
        split: Dataset split to process
        private: Whether to make the dataset private
    """
    import os
    from pathlib import Path
    from typing import Dict, List

    import torch
    import torchaudio
    from datasets import Dataset
    from neucodec import NeuCodec
    from torchaudio import transforms as T
    from tqdm import tqdm

    def load_libri_tts_r_metadata(dataset_root: Path, split: str) -> List[Dict]:
        """Load metadata from LibriTTS_R dataset directory structure."""
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
                    sample_id = wav_file.stem
                    normalized_txt = wav_file.with_suffix(".normalized.txt")
                    original_txt = wav_file.with_suffix(".original.txt")

                    text_normalized = (
                        normalized_txt.read_text().strip()
                        if normalized_txt.exists()
                        else ""
                    )
                    text_original = (
                        original_txt.read_text().strip()
                        if original_txt.exists()
                        else ""
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
        """Encode audio file to discrete codes using Neucodec."""
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != target_sample_rate:
            resampler = T.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(device)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            codes = model.encode_code(waveform)

        codes = codes.squeeze(0).squeeze(0).cpu()
        return codes

    def process_dataset(
        samples: List[Dict],
        model: NeuCodec,
        device: str,
    ) -> List[Dict]:
        """Process all samples in the dataset."""
        processed_samples = []

        for sample in tqdm(samples, desc="Tokenizing audio"):
            try:
                codes = encode_audio_to_codes(
                    sample["audio_path"],
                    model,
                    device,
                )

                waveform, sample_rate = torchaudio.load(sample["audio_path"])
                duration = waveform.shape[-1] / sample_rate

                processed_sample = {
                    "id": sample["id"],
                    "audio_duration": float(duration),
                    "text_normalized": sample["text_normalized"],
                    "text_original": sample["text_original"],
                    "codes": codes.numpy(),
                }

                processed_samples.append(processed_sample)

            except Exception as e:
                print(f"Error processing {sample['id']}: {e}")
                continue

        return processed_samples

    def create_hf_dataset(processed_samples: List[Dict]) -> Dataset:
        """Create Hugging Face Dataset from processed samples."""
        data_dict = {
            "id": [s["id"] for s in processed_samples],
            "audio_duration": [s["audio_duration"] for s in processed_samples],
            "text_normalized": [s["text_normalized"] for s in processed_samples],
            "text_original": [s["text_original"] for s in processed_samples],
            "codes": [s["codes"] for s in processed_samples],
        }

        dataset = Dataset.from_dict(data_dict)
        return dataset

    # Main execution
    print(f"Starting tokenization for split: {split}")
    print(f"Dataset root: {dataset_root}")
    print(f"Target HF dataset: {hf_dataset_id}")

    dataset_root_path = Path(dataset_root)
    if not dataset_root_path.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    # Load metadata
    print(f"Loading LibriTTS_R metadata from {dataset_root_path}/{split}...")
    samples = load_libri_tts_r_metadata(dataset_root_path, split)
    print(f"Found {len(samples)} samples")

    # Initialize Neucodec model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Neucodec model on {device}...")
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Process dataset
    print("Processing dataset...")
    processed_samples = process_dataset(samples, model, device)
    print(f"Successfully processed {len(processed_samples)} samples")

    # Create Hugging Face dataset
    print("Creating Hugging Face dataset...")
    dataset = create_hf_dataset(processed_samples)

    print("\nDataset info:")
    print(dataset)
    print("\nFirst sample:")
    print(dataset[0])

    # Upload to Hugging Face Hub
    print(f"\nUploading dataset to {hf_dataset_id}...")
    dataset.push_to_hub(
        hf_dataset_id,
        private=private,
        split=split,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"✓ Dataset uploaded successfully to {hf_dataset_id}")
    return {
        "split": split,
        "num_samples": len(processed_samples),
        "hf_dataset_id": hf_dataset_id,
    }


@app.local_entrypoint()
def main(
    dataset_root: str,
    hf_dataset_id: str,
    private: bool = False,
):
    """
    Local entrypoint to run tokenization on Modal.

    Args:
        dataset_root: Path to local LibriTTS_R dataset root
        hf_dataset_id: Hugging Face dataset ID
        private: Whether to make the dataset private
    """
    from pathlib import Path

    # Validate local dataset path
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")

    # Discover available splits
    available_splits = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            available_splits.append(item.name)

    available_splits = sorted(available_splits)

    if not available_splits:
        raise ValueError(f"No splits found in {dataset_root}")

    print("=" * 60)
    print("LibriTTS_R Tokenization on Modal GPU")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Splits found: {', '.join(available_splits)}")
    print(f"HF dataset ID: {hf_dataset_id}")
    print(f"Private: {private}")
    print("=" * 60)

    # Create a Modal volume and mount the dataset
    volume = modal.Volume.from_name("libritts-r-data", create_if_missing=True)

    # Copy dataset to Modal volume (this happens once)
    print("\nUploading dataset to Modal volume...")
    print("(This may take a while for the first run)")

    with volume.batch_upload() as batch:
        batch.put_directory(str(dataset_path), "/data")

    print("Dataset uploaded to Modal volume")

    # Run tokenization for each split
    results = []
    for i, split in enumerate(available_splits, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing split {i}/{len(available_splits)}: {split}")
        print(f"{'=' * 60}")

        result = tokenize_split.remote(
            dataset_root="/data",
            hf_dataset_id=hf_dataset_id,
            split=split,
            private=private,
        )
        results.append(result)

        print(f"✓ Split '{split}' completed: {result['num_samples']} samples")

    # Print final summary
    print("\n" + "=" * 60)
    print("All splits tokenization complete!")
    print("=" * 60)
    for result in results:
        print(f"  {result['split']}: {result['num_samples']} samples")
    print(f"\nDataset: https://huggingface.co/datasets/{hf_dataset_id}")
    print("=" * 60)
