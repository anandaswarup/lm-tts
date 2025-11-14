"""
Copy synthesis script using Neucodec audio codec.

This script performs copy synthesis on a directory of audio files by:
    1. Loading all wav files from an input directory
    2. Resampling to 16kHz if necessary (Neucodec's expected input rate)
    3. Encoding the audio using Neucodec to discrete codes
    4. Decoding the codes back to audio (24kHz output)
    5. Saving the reconstructed audio to an output directory

The script can be used to evaluate the quality of the Neucodec codec by comparing
the input audio with the reconstructed output.

Usage:
    python copy_synthesis.py --input_dir /path/to/input --output_dir /path/to/output

Note:
    - Input audio is resampled to 16kHz for encoding
    - Output audio is saved at 24kHz (Neucodec's native output rate)
    - Model is instantiated once for efficiency
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from neucodec import NeuCodec
from torchaudio import transforms as T


def process_single_file(
    input_path: Path,
    output_path: Path,
    model: NeuCodec,
    device: str,
    target_sample_rate: int = 16000,
) -> None:
    """
    Perform copy synthesis on a single audio file.

    Args:
        input_path: Path to the input wav file
        output_path: Path to save the reconstructed wav file
        model: Pre-initialized Neucodec model
        device: Device to use for inference
        target_sample_rate: Target sample rate for encoding (default: 16000)
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(str(input_path))

    # Resample if necessary (Neucodec expects 16kHz input)
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # Ensure waveform is in the correct shape (B, 1, T)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(
            0
        )  # Add batch and channel dimensions
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # Add batch dimension

    # Move to device
    waveform = waveform.to(device)

    # Perform encoding and decoding (copy synthesis)
    with torch.no_grad():
        # Encode to discrete codes
        fsq_codes = model.encode_code(waveform)

        # Decode back to audio (outputs at 24kHz)
        reconstructed = model.decode_code(fsq_codes)

    # Move back to CPU for saving
    reconstructed = reconstructed.cpu()

    # Remove batch dimension
    reconstructed = reconstructed.squeeze(0)

    # Save the reconstructed audio (24kHz output)
    torchaudio.save(
        str(output_path),
        reconstructed,
        sample_rate=24000,  # Neucodec outputs 24kHz audio
    )


def copy_synthesis_batch(input_dir: str, output_dir: str, device: str = "cpu") -> None:
    """
    Perform copy synthesis on all wav files in a directory using Neucodec.

    Args:
        input_dir: Path to directory containing input wav files
        output_dir: Path to directory where reconstructed wav files will be saved
        device: Device to use for inference ('cpu' or 'cuda')
    """
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input directory exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all wav files in input directory
    wav_files = sorted(input_path.glob("*.wav"))

    if not wav_files:
        print(f"No wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} wav files to process")

    # Initialize Neucodec model once
    print("Initializing Neucodec model...")
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # Process each wav file
    for i, wav_file in enumerate(wav_files, 1):
        output_file = output_path / wav_file.name
        print(f"\n[{i}/{len(wav_files)}] Processing: {wav_file.name}")

        try:
            process_single_file(wav_file, output_file, model, device)
            print(f"✓ Saved to: {output_file}")
        except Exception as e:
            print(f"✗ Error processing {wav_file.name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("Copy synthesis completed!")
    print(f"Processed {len(wav_files)} files")
    print(f"Output directory: {output_path}")
    print(f"{'=' * 60}")


def main():
    """
    Main method to parse arguments and run copy synthesis.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Perform copy synthesis on directory of wav files using Neucodec"
    )
    parser.add_argument(
        "--input_dir", type=str, help="Path to directory containing input wav files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory where reconstructed wav files will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cuda if available, else cpu)",
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        copy_synthesis_batch(args.input_dir, args.output_dir, args.device)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
