"""
Modal GPU script to tokenize LibriTTS_R dataset using Neucodec and upload to HuggingFace datasets.

This script runs the tokenization process entirely in Modal's cloud infrastructure
and can run independently after you close your laptop:
    1. Upload dataset to Modal volume (one-time)
    2. Call the deployed function which runs completely in the cloud
    3. Processes audio files using Neucodec on GPU
    4. Uploads the tokenized dataset to Hugging Face Hub

Setup:
    1. First, upload your dataset to Modal volume:
       modal run tokenize_libritts_r_modal.py::upload_dataset \
           --local-path /path/to/LibriTTS_R

    2. Deploy the tokenization function:
       modal deploy tokenize_libritts_r_modal.py
    3. Trigger the tokenization (runs in background):
       modal run tokenize_libritts_r_modal.py::trigger_tokenization \
           --hf-dataset-id your-username/dataset-name

    4. Check status anytime:
       modal app logs tokenize-libritts-r

Note:
    - Requires Modal account and authentication (modal setup)
    - Requires HF_TOKEN secret configured in Modal
    - Uses A100 GPU by default
"""

import modal

# Create Modal app
app = modal.App("tokenize-libritts-r")

# Create persistent volume for dataset storage
volume = modal.Volume.from_name("libritts-r-data", create_if_missing=True)

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
    gpu="A100",
    timeout=86400,  # 24 hours timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
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
        """
        Load metadata from LibriTTS_R dataset directory structure.
        """
        split_path = dataset_root / split
        if not split_path.exists():
            raise ValueError(f"Split directory not found: {split_path}")

        samples = []

        for speaker_dir in sorted(split_path.iterdir()):
            if not speaker_dir.is_dir():
                continue

            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue

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
        """
        Encode audio file to discrete codes using Neucodec.
        """
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
        """
        Process all samples in the dataset.
        """
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
        """
        Create Hugging Face Dataset from processed samples.
        """
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


@app.function(
    image=image,
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
)
def process_all_splits(hf_dataset_id: str, private: bool = False):
    """
    Process all available splits in the dataset.
    """
    from pathlib import Path

    dataset_root = Path("/data")

    # Discover available splits
    available_splits = []
    for item in dataset_root.iterdir():
        if item.is_dir():
            available_splits.append(item.name)

    available_splits = sorted(available_splits)

    if not available_splits:
        raise ValueError(f"No splits found in {dataset_root}")

    print("=" * 60)
    print("LibriTTS_R Tokenization on Modal GPU")
    print("=" * 60)
    print(f"Splits found: {', '.join(available_splits)}")
    print(f"HF dataset ID: {hf_dataset_id}")
    print(f"Private: {private}")
    print("=" * 60)

    # Process each split
    results = []
    for i, split in enumerate(available_splits, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing split {i}/{len(available_splits)}: {split}")
        print(f"{'=' * 60}")

        result = tokenize_split.local(
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

    return results

@app.local_entrypoint()
def upload_dataset(local_path: str):
    """
    Upload local LibriTTS_R dataset to Modal volume.

    Args:
        local_path: Path to local LibriTTS_R dataset
    """
    from pathlib import Path

    dataset_path = Path(local_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path not found: {local_path}")

    # Count total files for progress tracking
    print("Scanning dataset directory...")
    all_files = list(dataset_path.rglob("*"))
    file_count = sum(1 for f in all_files if f.is_file())
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())

    print("=" * 60)
    print("Uploading LibriTTS_R Dataset to Modal Volume")
    print("=" * 60)
    print(f"Local path: {local_path}")
    print(f"Files to upload: {file_count:,}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print("=" * 60)

    print("\nUploading dataset to Modal volume...")
    print("(This may take a while depending on your connection)")
    print("Progress will be shown by Modal...")

    with volume.batch_upload() as batch:
        batch.put_directory(str(dataset_path), "/")

    print("\n✓ Dataset uploaded successfully to Modal volume")
    print("You can now trigger tokenization with:")
    print("  modal run tokenize_libritts_r_modal.py::trigger_tokenization \\")
    print("      --hf-dataset-id your-username/dataset-name")


@app.local_entrypoint()
def trigger_tokenization(hf_dataset_id: str, private: bool = False):
    """
    Trigger tokenization job that runs completely in the cloud.

    Args:
        hf_dataset_id: Hugging Face dataset ID
        private: Whether to make the dataset private
    """
    print("=" * 60)
    print("Triggering LibriTTS_R Tokenization (Background Mode)")
    print("=" * 60)
    print(f"HF dataset ID: {hf_dataset_id}")
    print(f"Private: {private}")
    print("=" * 60)
    print("\nStarting background job on Modal...")
    print("You can safely close your laptop after this starts.")
    print("\nTo check logs later, run:")
    print("  modal app logs tokenize-libritts-r")
    print("=" * 60)

    # Spawn the job asynchronously
    call = process_all_splits.spawn(hf_dataset_id=hf_dataset_id, private=private)

    print(f"\n✓ Job started with ID: {call.object_id}")
    print("\nThe job is now running in the cloud.")
    print("You can disconnect and check status later with:")
    print("  modal app logs tokenize-libritts-r")
