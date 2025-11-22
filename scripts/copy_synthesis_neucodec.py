"""
Script to perform copy synthesis using the NeuCodec neural audio codec. This script takes speech wav files as
input, encodes them using the NeuCodec codec, and then decodes them back (copy synthesis), for testing the codec's
reconstruction quality for different languages.
"""

import argparse
import logging
import time

import torch
import torchaudio
from einops import rearrange
from neucodec import DistillNeuCodec, NeuCodec

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

INPUT_SR = 16000  # NeuCodec expects input at 16kHz sample rate
OUTPUT_SR = 24000  # NeuCodec outputs at 24kHz sample rate
MODEL_NAME = "neuphonic/distill-neucodec"  # Options: neuphonic/neucodec, neuphonic/distill-neucodec
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_process_audio(input_wav: str) -> torch.Tensor:
    """
    Load the input wav file and convert to mono + resample it to 16kHz if necessary.

    Args:
        input_wav (str): Path to the input wav file.

    Returns:
        torch.Tensor: audio tensor of shape [1, 1, num_samples] sampled at 16kHz. First dimension is batch size,
        second dimension is channel, third dimension is number of samples.
    """
    # Load audio
    y, sr = torchaudio.load(input_wav, normalize=True)

    # Convert to mono if input is stereo
    if y.size(0) > 1:
        y = torch.mean(y, dim=0, keepdim=True)

    # Resample to 16kHz if necessary
    if sr != INPUT_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=INPUT_SR)
        y = resampler(y)

    # Add batch dimension
    y = rearrange(y, "num_channels num_samples -> 1 num_channels num_samples")

    return y


def main() -> None:
    """
    Main method to parse arguments and perform copy synthesis using NeuCodec.
    """
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Perform copy synthesis using NeuCodec neural audio codec."
    )
    parser.add_argument(
        "--input_wav",
        type=str,
        required=True,
        help="Path to the input speech wav file.",
    )
    parser.add_argument(
        "--output_wav",
        type=str,
        required=True,
        help="Path to save the reconstructed wav file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="neucodec",
        help="Name of the NeuCodec model to use. Options: neucodec, distill-neucodec",
    )

    # Parse the arguments
    args = parser.parse_args()
    input_wav = args.input_wav
    output_wav = args.output_wav
    model_name = args.model_name

    # Load NeuCodec model and place it on the appropriate device
    try:
        if model_name == "neucodec":
            model = NeuCodec.from_pretrained("neuphonic/neucodec")
        elif model_name == "distill-neucodec":
            model = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec")
        else:
            raise ValueError(
                f"Invalid model_name: {model_name}. Choose from 'neucodec' or 'distill-neucodec'."
            )
    except Exception as e:
        logger.error(f"Error loading NeuCodec model: {e}")
        return

    model = model.to(DEVICE)
    model.eval()
    logger.info(
        f"NeuCodec model: {model_name} loaded successfully on device: {DEVICE}."
    )

    # Load and process input audio
    y = load_and_process_audio(input_wav)
    y = y.to(DEVICE)
    logger.info(f"Audio loaded and processed. Shape: {y.shape}, Device: {DEVICE}")

    # Encode the audio using NeuCodec
    logger.info("Encoding audio waveform into codes ...")
    encode_start = time.time()
    with torch.no_grad():
        codes = model.encode_code(y)
    encode_end = time.time()
    encode_time = encode_end - encode_start

    if codes is None:
        logger.error("Encoding failed: encoded_codes is None")
    logger.info(f"Encoded codes shape: {codes.shape}")
    logger.info(f"Encoded codes device: {codes.device}")
    logger.info(f"Encoding time: {encode_time:.4f}s")

    # Log some statistics about the codes
    logger.info(f"Code sequence length: {codes.shape[-1]}")
    logger.info(f"Code range: [{codes.min().item():.0f}, {codes.max().item():.0f}]")

    # Calculate compression ratio
    compression_ratio = y.shape[2] / codes.numel() if codes.numel() > 0 else 0
    logger.info(
        f"Compression ratio: {compression_ratio:.1f} ({y.shape[2]} samples -> {codes.numel()} codes)"
    )

    # Decode the codes back to audio using NeuCodec
    logger.info("Decoding codes back to audio waveform ...")
    decode_start = time.time()
    with torch.no_grad():
        y_hat = model.decode_code(codes)
    decode_end = time.time()
    decode_time = decode_end - decode_start
    logger.info(f"Decoding time: {decode_time:.4f}s")

    # Log total coding time
    total_coding_time = encode_time + decode_time
    logger.info(f"Total encoding + decoding time: {total_coding_time:.4f}s")

    # Calculate metrics
    logger.info("Calculating quality metrics...")

    # Resample original to 24kHz to match reconstructed output
    resampler = torchaudio.transforms.Resample(orig_freq=INPUT_SR, new_freq=OUTPUT_SR)
    y_24k = resampler(y)

    # Handle length differences (common with codecs)
    min_len = min(y_24k.shape[-1], y_hat.shape[-1])
    original_trimmed = y_24k[..., :min_len]
    reconstructed_trimmed = y_hat[..., :min_len]

    # Simple MSE calculation using torch
    mse = torch.mean((original_trimmed - reconstructed_trimmed) ** 2).item()

    if y_24k.shape[-1] != y_hat.shape[-1]:
        logger.info(
            f"Audio length difference: Original {y_24k.shape[-1]} samples, Reconstructed {y_hat.shape[-1]} samples"
        )

    logger.info(f"MSE (first {min_len} samples at 24kHz): {mse:.6f}")

    # Calculate Signal-to-Noise Ratio (SNR) using torch
    signal_power = torch.mean(original_trimmed**2).item()
    noise_power = mse
    if noise_power > 0:
        snr_db = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
        logger.info(f"SNR: {snr_db:.2f} dB")

    # Save the reconstructed audio
    logger.info(f"Saving reconstructed audio to {output_wav} ...")
    torchaudio.save(
        output_wav,
        rearrange(
            y_hat.cpu(), "1 num_channels num_samples -> num_channels num_samples"
        ),
        OUTPUT_SR,
    )


if __name__ == "__main__":
    main()
